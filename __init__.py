from .decoder_noise import DecoderNoise
from .tiled_vae_decode import LTTiledVAEDecode

NODE_CLASS_MAPPINGS = {
    "DecoderNoise": DecoderNoise,
    "LTTiledVAEDecode": LTTiledVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DecoderNoise": "🅛🅣 VAE Decoder Noise",
    "LTTiledVAEDecode": "🅛🅣 Tiled VAE Decode",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
