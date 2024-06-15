from dataclasses import dataclass
from typing import Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from models.decoder_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
    get_down_block,
    get_up_block,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor

    
class NoiseMapping(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        act_fn: str = "silu",
        num_fcs: int = None,
        post_act_fn: Optional[str] = None,
    ):
        super().__init__()
        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "mish":
            self.act = nn.Mish()
        elif act_fn == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"{act_fn} does not exist. Make sure to define one of 'silu', 'mish', or 'gelu'")

        linears = []
        for i in range(num_fcs):
            if i == 0:
                linears.extend([nn.Linear(in_channels, out_channels), self.act])
            elif i == num_fcs - 1:
                linears.extend([nn.Linear(out_channels, out_channels)])
            else:
                linears.extend([nn.Linear(out_channels, out_channels), self.act])

        self.linears = nn.Sequential(*linears)

        if post_act_fn is None:
            self.post_act = None
        elif post_act_fn == "silu":
            self.post_act = nn.SiLU()
        elif post_act_fn == "mish":
            self.post_act = nn.Mish()
        elif post_act_fn == "gelu":
            self.post_act = nn.GELU()
        else:
            raise ValueError(f"{post_act_fn} does not exist. Make sure to define one of 'silu', 'mish', or 'gelu'")

    def forward(self, sample):
        sample = self.linears(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample
    

class TriplaneDecoder(ModelMixin, ConfigMixin):
    r"""
    UNet2DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a timestep
    and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        num_class_embeds: Optional[int] = None,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        down_blocks_no_crossatt_out_channels = [int(block_out_channels[0] / 4), int(block_out_channels[0] / 2)] # 80, 160
        up_blocks_no_crossatt_out_channels = list(reversed(down_blocks_no_crossatt_out_channels)) # 160, 80
        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, padding=(1, 1))

        # time
        # self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        # timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(768*2, time_embed_dim)
        # self.time_embedding = NoiseMapping(768*2, time_embed_dim, num_fcs=8)

        # class embedding
        # df_if num_class_embeds is not None:
        #     self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        # self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        
        # resnet_time_scale_shift = "default"
        resnet_time_scale_shift = "ada_group"
        # resnet_time_scale_shift = "scale_shift"

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            add_upsample = True
            self.num_upsamplers += 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                resnet_time_scale_shift=resnet_time_scale_shift,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # up blocks with no cross attention
        self.up_blocks_no_crossatt_1 = UpBlock2D(
            num_layers=layers_per_block + 1,
            in_channels=up_blocks_no_crossatt_out_channels[0], # 160
            out_channels=reversed_block_out_channels[-1], # 320
            prev_output_channel=prev_output_channel,
            temb_channels=time_embed_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_upsample=True,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=int(norm_num_groups / 2) if int(norm_num_groups / 2) > 0 else 1,
        )
        self.num_upsamplers += 1
        prev_output_channel = reversed_block_out_channels[-1]

        self.up_blocks_no_crossatt_2 = UpBlock2D(
            num_layers=layers_per_block + 1,
            in_channels=up_blocks_no_crossatt_out_channels[1], # 80
            out_channels=up_blocks_no_crossatt_out_channels[0], # 160
            prev_output_channel=prev_output_channel,
            temb_channels=time_embed_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_upsample=True,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=int(norm_num_groups / 4) if int(norm_num_groups / 4) > 0 else 1,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=up_blocks_no_crossatt_out_channels[0],
                                          num_groups=int(norm_num_groups / 2) if int(norm_num_groups / 2) > 0 else 1, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(up_blocks_no_crossatt_out_channels[0], out_channels, kernel_size=3, padding=1)

    def set_attention_slice(self, slice_size):
        head_dims = self.config.attention_head_dim
        head_dims = [head_dims] if isinstance(head_dims, int) else head_dims
        if slice_size is not None and any(dim % slice_size != 0 for dim in head_dims):
            raise ValueError(
                f"Make sure slice_size {slice_size} is a common divisor of "
                f"the number of heads used in cross_attention: {head_dims}"
            )
        if slice_size is not None and slice_size > min(head_dims):
            raise ValueError(
                f"slice_size {slice_size} has to be smaller or equal to "
                f"the lowest number of heads used in cross_attention: min({head_dims}) = {min(head_dims)}"
            )

        # for block in self.down_blocks:
        #     df_if hasattr(block, "attentions") and block.attentions is not None:
        #         block.set_attention_slice(slice_size)

        self.mid_block.set_attention_slice(slice_size)

        for block in self.up_blocks:
            if hasattr(block, "attentions") and block.attentions is not None:
                block.set_attention_slice(slice_size)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        # for block in self.down_blocks:
        #     df_if hasattr(block, "attentions") and block.attentions is not None:
        #         block.set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)

        self.mid_block.set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)

        for block in self.up_blocks:
            if hasattr(block, "attentions") and block.attentions is not None:
                block.set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        encoder_hidden_states: torch.Tensor,
        t_emb: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, channel, height, width) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] df_if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly df_if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        encoder_hidden_states = encoder_hidden_states.to(sample.dtype)

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 0. center input df_if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        # timesteps = timestep
        # df_if not torch.is_tensor(timesteps):
        #     # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors df_if you can
        #     timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        # elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
        #     timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # timesteps = timesteps.expand(sample.shape[0])

        # t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # df_if self.config.num_class_embeds is not None:
        #     df_if class_labels is None:
        #         raise ValueError("class_labels should be provided when num_class_embeds > 0")
        #     class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
        #     emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # # 3. down
        # res_samples_down_no_crossatt = (sample, )
        # sample, res_samples = self.down_blocks_no_crossatt_1(hidden_states=sample, temb=emb)
        # res_samples_down_no_crossatt += res_samples
        # sample, res_samples = self.down_blocks_no_crossatt_2(hidden_states=sample, temb=emb)
        # res_samples_down_no_crossatt += res_samples
        #
        # res_samples_down_no_crossatt = res_samples_down_no_crossatt[:-1]

        # down_block_res_samples = (sample,)
        # for downsample_block in self.down_blocks:
        #     df_if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
        #         sample, res_samples = downsample_block(
        #             hidden_states=sample,
        #             temb=emb,
        #             encoder_hidden_states=encoder_hidden_states,
        #         )
        #     else:
        #         sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
        #
        #     down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            # res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            res_samples = None
            # down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # df_if we have not reached the final block and need to forward the
            # upsample size, we do it here
            # df_if not is_final_block and forward_upsample_size:
            #     upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # res_samples = res_samples_down_no_crossatt[-len(self.up_blocks_no_crossatt_1.resnets):]
        res_samples = None
        # res_samples_down_no_crossatt = res_samples_down_no_crossatt[:-len(self.up_blocks_no_crossatt_1.resnets)]
        sample = self.up_blocks_no_crossatt_1(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples,
                                              upsample_size=None)

        # res_samples = res_samples_down_no_crossatt[-len(self.up_blocks_no_crossatt_2.resnets):]
        res_samples = None
        sample = self.up_blocks_no_crossatt_2(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples,
                                              upsample_size=None)
        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)