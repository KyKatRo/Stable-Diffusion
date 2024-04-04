from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter


def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    def fix_state_dict(s_dict):
        new_state_dict = {}
        for key, value in s_dict.items():
            # Replace incorrect key parts with the correct ones
            new_key = key

            new_key = key.replace('residual_layer', 'residual')
            new_key = new_key.replace('groupnorm_', 'groupnorm')
            new_key = new_key.replace('conv_', 'conv')
            # new_key = new_key.replace('residual_', 'residual')
            new_key = new_key.replace('attention_', 'attention.')

            # Fix the missing dots between the block number and the layer name
            parts = new_key.split('.')
            new_parts = []
            for part in parts:
                if part.isnumeric() and new_parts:
                    new_parts[-1] = new_parts[-1] + '.' + part
                else:
                    new_parts.append(part)
            new_key = '.'.join(new_parts)

            new_state_dict[new_key] = value

        return new_state_dict

    encoder = VAE_Encoder().to(device)
    encoder_state_dict = fix_state_dict(state_dict["encoder"])
    encoder.load_state_dict(encoder_state_dict, strict=True)

    decoder = VAE_Decoder().to(device)
    decoder_state_dict = fix_state_dict(state_dict["decoder"])
    decoder.load_state_dict(decoder_state_dict, strict=True)

    def fix_state_dict_diffusion(s_dict):
        new_state_dict = {}
        for key, value in s_dict.items():
            new_key = key.replace('_', '.')  # Initial replacement to convert underscores to dots
            # Specific replacements based on the observed naming patterns
            new_key = new_key.replace('linear.time', 'linear_time')
            new_key = new_key.replace('conv.feature', 'conv1_feature')
            new_key = new_key.replace('conv.merged', 'conv2_merged')
            new_key = new_key.replace('linear.linear_geglu', 'linear_geglu')
            new_key = new_key.replace('attention.out.proj', 'attention1.out_proj')
            new_key = new_key.replace('attention.in.proj', 'attention1.in_proj')
            new_key = new_key.replace('attention.q.proj', 'attention2.q_proj')
            new_key = new_key.replace('attention.k.proj', 'attention2.k_proj')
            new_key = new_key.replace('attention.v.proj', 'attention2.v_proj')
            new_key = new_key.replace('residual.layer', 'residual')

            new_state_dict[new_key] = value

        return new_state_dict

    diffusion = Diffusion().to(device)
    diffusion_state_dict = fix_state_dict_diffusion(state_dict["diffusion"])
    diffusion.load_state_dict(diffusion_state_dict, strict=True)

    clip = CLIP().to(device)
    clip_state_dict = fix_state_dict(state_dict["clip"])
    clip.load_state_dict(clip_state_dict, strict=True)

    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
    }
