# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.layer import \
    FusedMoeWeightScaleSupported
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.awq import (AWQConfig,
                                                         AWQLinearMethod)
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.process_weights.moe_weights import \
    FusedMoEWeights
from tpu_inference.layers.common.quant_methods import AWQ
from tpu_inference.layers.common.quantization import awq_u32_unpack_u4
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import (
    reorder_concatenated_tensor_for_sharding,
    slice_sharded_tensor_for_concatenation)
from tpu_inference.layers.vllm.moe import (
    MoEBackend, select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedLinearMethod
from tpu_inference.logger import init_logger
from tpu_inference.utils import align_to, get_mesh_shape_product

P = PartitionSpec

logger = init_logger(__name__)


def _awq_unpack_int8_and_scale(
    qweight: jax.Array,
    qzeros: jax.Array,
    scales: jax.Array,
    group_size: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Unpacks AWQ weights to int8 (centered) and returns them alongside bf16 scales.
    Does NOT dequantize to bf16 to save memory.
    """
    w = awq_u32_unpack_u4(qweight)
    z = awq_u32_unpack_u4(qzeros)

    w = w.astype(jnp.int8)
    z = z.astype(jnp.int8)

    E, in_feat, out_feat = w.shape
    w = w.reshape(E, -1, group_size, out_feat)

    z = jnp.expand_dims(z, 2)
    w = w - z  # int8 centered weights

    # Reshape w back to (E, in_feat, out_feat)
    w = w.reshape(E, in_feat, out_feat)

    # Scales are (E, in_feat // group_size, out_feat)
    s = scales.astype(jnp.bfloat16)

    return w, s


def _awq_dequant_and_format_moe_weights(
    w13_qw: jax.Array,
    w13_qz: jax.Array,
    w13_s: jax.Array,
    w2_qw: jax.Array,
    w2_qz: jax.Array,
    w2_s: jax.Array,
    group_size: int,
    moe_backend: MoEBackend,
    w13_interleave: bool,
    w13_reorder_size: int,
    mesh: Mesh,
) -> FusedMoEWeights:
    # Unpack to int8 and get bf16 scales, but do NOT multiply yet
    w13, w13_scale = _awq_unpack_int8_and_scale(w13_qw, w13_qz, w13_s,
                                                group_size)
    w2, w2_scale = _awq_unpack_int8_and_scale(w2_qw, w2_qz, w2_s, group_size)

    E = w13.shape[0]
    H = w13.shape[1]
    two_I = w13.shape[2]
    I = two_I // 2

    if w13_interleave:
        # Interleave weights (int8)
        w1 = w13[:, :, ::2]
        w3 = w13[:, :, 1::2]
        w13 = jnp.concatenate([w1, w3], axis=2)

        # Interleave scales (bf16)
        # Scales shape: (E, H//G, 2*I)
        w1_s = w13_scale[:, :, ::2]
        w3_s = w13_scale[:, :, 1::2]
        w13_scale = jnp.concatenate([w1_s, w3_s], axis=2)

    match moe_backend:
        case MoEBackend.FUSED_MOE:
            # Process Weights (int8)
            # ----------------------
            w13 = w13.reshape(E, H, 2, I)
            w13 = jnp.swapaxes(w13, 1, 2)  # (E, 2, H, I)

            pad_H = align_to(H, 256) - H
            pad_I = align_to(I, 256) - I

            if pad_H > 0 or pad_I > 0:
                w13 = jnp.pad(w13, ((0, 0), (0, 0), (0, pad_H), (0, pad_I)))
                w2 = jnp.pad(w2, ((0, 0), (0, pad_I), (0, pad_H)))

            # Process Scales (bf16)
            # ---------------------
            # w13_scale expected shape for kernel: (E, 2, H_blocks, 1, I)
            # Current w13_scale: (E, H_blocks, 2*I)
            H_blocks = w13_scale.shape[1]
            w13_scale = w13_scale.reshape(E, H_blocks, 2, I)
            w13_scale = jnp.swapaxes(w13_scale, 1, 2)  # (E, 2, H_blocks, I)
            w13_scale = jnp.expand_dims(w13_scale, 3)  # (E, 2, H_blocks, 1, I)

            # w2_scale expected shape for kernel: (E, I_blocks, 1, H)
            # Current w2_scale: (E, I_blocks, H)
            w2_scale = jnp.expand_dims(w2_scale, 2)  # (E, I_blocks, 1, H)

            # Pad scales if necessary
            pad_H_blocks = pad_H // group_size
            if pad_H > 0 or pad_I > 0:
                w13_scale = jnp.pad(w13_scale,
                                    ((0, 0), (0, 0), (0, pad_H_blocks), (0, 0),
                                     (0, pad_I)))

                pad_I_blocks = pad_I // group_size
                w2_scale = jnp.pad(w2_scale, ((0, 0), (0, pad_I_blocks),
                                              (0, 0), (0, pad_H)))

        case MoEBackend.GMM_EP:
            # For GMM kernels, we generally expect 4D scales: (E, Blocks, 1, Dim)
            # w13_scale: (E, H//G, 2*I) -> (E, H//G, 1, 2*I)
            w13_scale = jnp.expand_dims(w13_scale, 2)
            # w2_scale: (E, I//G, H) -> (E, I//G, 1, H)
            w2_scale = jnp.expand_dims(w2_scale, 2)

        case MoEBackend.GMM_TP:
            output_sizes = [I, I]
            # Reorder Weights
            w13 = reorder_concatenated_tensor_for_sharding(w13,
                                                           output_sizes,
                                                           w13_reorder_size,
                                                           dim=2)
            # Reorder Scales (dim 2 is features/output)
            w13_scale = reorder_concatenated_tensor_for_sharding(
                w13_scale, output_sizes, w13_reorder_size, dim=2)

            # Expand scales to 4D for GMM kernel: (E, Blocks, 1, Dim)
            w13_scale = jnp.expand_dims(w13_scale, 2)
            w2_scale = jnp.expand_dims(w2_scale, 2)

            # Sharding Constraints
            w13 = jax.lax.with_sharding_constraint(
                w13,
                NamedSharding(mesh, P(None, None,
                                      ShardingAxisName.MLP_TENSOR)))
            w2 = jax.lax.with_sharding_constraint(
                w2,
                NamedSharding(mesh, P(None, ShardingAxisName.MLP_TENSOR,
                                      None)))

            # Scales sharding constraints (aligned with weights)
            w13_scale = jax.lax.with_sharding_constraint(
                w13_scale,
                NamedSharding(mesh,
                              P(None, None, None,
                                ShardingAxisName.MLP_TENSOR)))

            w2_scale = jax.lax.with_sharding_constraint(
                w2_scale,
                NamedSharding(mesh,
                              P(None, ShardingAxisName.MLP_TENSOR, None,
                                None)))

    return FusedMoEWeights(
        w13_weight=w13,
        w13_weight_scale=w13_scale,
        w13_bias=None,
        w2_weight=w2,
        w2_weight_scale=w2_scale,
        w2_bias=None,
    )


@register_quantization_config(AWQ)
class VllmAWQConfig(AWQConfig, VllmQuantConfig):

    @classmethod
    def get_name(cls):
        return AWQ

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # NOTE: AWQ checkpoint was quantized with float16. But on TPUs, using
        # bfloat16 is significantly preferred over float16. This might lead to
        # some numeric output change.
        return [torch.bfloat16]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[Union["LinearMethodBase", "QuantizeMethodBase"]]:
        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            if is_layer_skipped(prefix, self.modules_to_not_convert):
                return VllmUnquantizedLinearMethod(linear_config)
            return VllmAWQLinearMethod(self, linear_config)
        elif isinstance(layer, FusedMoE):
            layer.moe_config = self.get_moe_config(layer)
            return VllmAWQMoEMethod(self, layer, self.mesh)
        return None


class VllmAWQLinearMethod(AWQLinearMethod):

    def __init__(self, quant_config: VllmAWQConfig,
                 linear_config: VllmQuantLinearConfig):
        super().__init__(quant_config)
        self.linear_config = linear_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert layer.qweight.packed_dim == layer.qweight.ndim - 1
        weight = t2j(layer.qweight, use_dlpack=False)
        delattr(layer, "qweight")

        weight_scale = t2j(layer.scales, use_dlpack=False)
        delattr(layer, "scales")

        assert layer.qzeros.packed_dim == layer.qzeros.ndim - 1
        zero_point = t2j(layer.qzeros, use_dlpack=False)
        delattr(layer, "qzeros")

        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")
        else:
            bias = None

        @jax.jit
        def process_awq_linear_weights(
            weight: jax.Array,
            weight_scale: jax.Array,
            zero_point: jax.Array,
            bias: jax.Array | None,
        ) -> LinearWeights:
            weight = awq_u32_unpack_u4(weight)
            group_size = self.quant_config.group_size
            weight = weight.reshape((-1, group_size, weight.shape[-1]))

            zero_point = awq_u32_unpack_u4(zero_point)

            return process_linear_weights(
                LinearWeights(
                    weight=weight,
                    weight_scale=weight_scale,
                    zero_point=zero_point,
                    bias=bias,
                ),
                fused=self.linear_config.fuse_matmuls,
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
                transposed=False,
            )

        weights = process_awq_linear_weights(weight, weight_scale, zero_point,
                                             bias)
        weights = torch_view(
            shard_linear_weights(
                weights,
                mesh=self.linear_config.mesh,
                weight_p_spec=self.linear_config.weight_sharding,
                bias_p_spec=self.linear_config.bias_sharding,
                transposed=False,
            ))

        if self.linear_config.fuse_matmuls:
            layer.qweight = Parameter(weights.weight, requires_grad=False)
            layer.scales = Parameter(weights.weight_scale, requires_grad=False)
            layer.qzeros = Parameter(weights.zero_point, requires_grad=False)
            if bias is not None:
                layer.bias = Parameter(weights.bias, requires_grad=False)
        else:
            layer.qweight = to_parameter_list(weights.weight)
            layer.scales = to_parameter_list(weights.weight_scale)
            layer.qzeros = to_parameter_list(weights.zero_point)
            if bias is not None:
                layer.bias = to_parameter_list(weights.bias)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        with jax.named_scope(layer._get_name()):
            if self.linear_config.fuse_matmuls:
                out = self._apply_fused(layer, x, bias)
            else:
                out = self._apply_split(layer, x, bias)

        return out

    def _apply_fused(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_jax = jax_view(x)

        qweight = jax_view(layer.qweight)
        qzeros = jnp.expand_dims(jax_view(layer.qzeros), 1)
        scales = jnp.expand_dims(jax_view(layer.scales), 1)

        qweight = qweight.astype(jnp.int8)
        qzeros = qzeros.astype(jnp.int8)

        weight = (qweight - qzeros) * scales
        weight = weight.reshape((-1, weight.shape[-1]))
        outs = jnp.einsum("bd,df->bf", x_jax, weight)

        if bias is not None and not layer.skip_bias_add:
            outs += bias.jax()

        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)

    def _apply_split(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert isinstance(layer.qweight, torch.nn.ParameterList)

        x_jax = jax_view(x)
        params = zip(layer.qweight, layer.qzeros, layer.scales)
        outs = []
        for i, (qweight, qzeros, scales) in enumerate(params):
            qweight = jax_view(qweight)
            scales = jnp.expand_dims(jax_view(scales), 1)
            qzeros = jnp.expand_dims(jax_view(qzeros), 1)

            qweight = qweight.astype(jnp.int8)
            qzeros = qzeros.astype(jnp.int8)

            weight = (qweight - qzeros) * scales
            weight = weight.reshape((-1, weight.shape[-1]))
            out = jnp.einsum("bd,df->bf", x_jax, weight)

            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])

            outs.append(out)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)


class VllmAWQMoEMethod(FusedMoEMethodBase):

    def __init__(
        self,
        quant_config: VllmAWQConfig,
        layer: torch.nn.Module,
        mesh: Mesh,
        ep_axis_name: str = "model",
    ):
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        self.quant_config = quant_config

        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)

        self.extra_backend_kwargs = {}
        if self.moe_backend == MoEBackend.FUSED_MOE:
            self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name)

        self._w13_interleave = layer.activation == "swigluoai"
        self._w13_reorder_size = get_mesh_shape_product(
            self.mesh, ShardingAxisName.MLP_TENSOR)

    @property
    def is_monolithic(self) -> bool:
        return True

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> None:
        return None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        extra_weight_attrs.update({
            "is_transposed":
            True,
            "quant_method":
            FusedMoeWeightScaleSupported.GROUP.value,
        })

        w13_qweight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                2 * intermediate_size_per_partition //
                self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.quant_config.group_size
        num_groups_w2 = intermediate_size_per_partition // self.quant_config.group_size

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        w13_scales = Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                intermediate_size_per_partition * 2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = Parameter(
            torch.empty(num_experts,
                        num_groups_w2,
                        hidden_size,
                        dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        # WEIGHT_ZERO_POINT
        # Allocate 2 zero points for w1 and w3 respectively.
        w13_qzeros = Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition //
                self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, FusedMoE)

        assert not self.moe.has_bias

        w13_qweight = t2j(layer.w13_qweight, use_dlpack=False)
        delattr(layer, "w13_qweight")

        w2_qweight = t2j(layer.w2_qweight, use_dlpack=False)
        delattr(layer, "w2_qweight")

        w13_scales = t2j(layer.w13_scales, use_dlpack=False)
        delattr(layer, "w13_scales")

        w2_scales = t2j(layer.w2_scales, use_dlpack=False)
        delattr(layer, "w2_scales")

        w13_qzeros = t2j(layer.w13_qzeros, use_dlpack=False)
        delattr(layer, "w13_qzeros")

        w2_qzeros = t2j(layer.w2_qzeros, use_dlpack=False)
        delattr(layer, "w2_qzeros")

        if self.moe_backend in MoEBackend.expert_sharded_backends():
            sharding = NamedSharding(self.mesh, P(ShardingAxisName.EXPERT))
        else:
            sharding = NamedSharding(self.mesh, P())

        w13_qweight = jax.device_put(w13_qweight, sharding)
        w2_qweight = jax.device_put(w2_qweight, sharding)

        w13_scales = jax.device_put(w13_scales, sharding)
        w2_scales = jax.device_put(w2_scales, sharding)

        w13_qzeros = jax.device_put(w13_qzeros, sharding)
        w2_qzeros = jax.device_put(w2_qzeros, sharding)

        layer.w13_qweight = Parameter(torch_view(w13_qweight),
                                      requires_grad=False)
        layer.w2_qweight = Parameter(torch_view(w2_qweight),
                                     requires_grad=False)

        layer.w13_scales = Parameter(torch_view(w13_scales),
                                     requires_grad=False)
        layer.w2_scales = Parameter(torch_view(w2_scales), requires_grad=False)

        layer.w13_qzeros = Parameter(torch_view(w13_qzeros),
                                     requires_grad=False)
        layer.w2_qzeros = Parameter(torch_view(w2_qzeros), requires_grad=False)

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        (w13_qw, w13_qz, w13_s, w2_qw, w2_qz,
         w2_s) = jax.lax.optimization_barrier(
             (jax_view(layer.w13_qweight), jax_view(layer.w13_qzeros),
              jax_view(layer.w13_scales), jax_view(layer.w2_qweight),
              jax_view(layer.w2_qzeros), jax_view(layer.w2_scales)))

        weights = _awq_dequant_and_format_moe_weights(
            w13_qw,
            w13_qz,
            w13_s,
            w2_qw,
            w2_qz,
            w2_s,
            group_size=self.quant_config.group_size,
            moe_backend=self.moe_backend,
            w13_interleave=self._w13_interleave,
            w13_reorder_size=self._w13_reorder_size,
            mesh=self.mesh,
        )

        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits)
