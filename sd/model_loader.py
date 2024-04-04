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
            # Fixing naming for group normalization and convolutional layers
            new_key = key.replace('groupnorm_', 'groupnorm')
            new_key = new_key.replace('conv_', 'conv')

            # Fixing naming for residual layers
            new_key = new_key.replace('residuallayer', 'residual')
            new_key = new_key.replace('residual_layer', 'residual')  # Adjusting this line to match the expected keys

            # Fixing naming for attention layers
            new_key = new_key.replace('inproj', 'in_proj')
            new_key = new_key.replace('outproj', 'out_proj')

            new_state_dict[new_key] = value
        return new_state_dict

    encoder = VAE_Encoder().to(device)
    encoder_state_dict = fix_state_dict(state_dict["encoder"])
    encoder.load_state_dict(encoder_state_dict, strict=True)

    decoder = VAE_Decoder().to(device)
    decoder_state_dict = fix_state_dict(state_dict["decoder"])
    decoder.load_state_dict(decoder_state_dict, strict=True)

    diffusion = Diffusion().to(device)
    diffusion_state_dict = fix_state_dict(state_dict["diffusion"])
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

