from .tiled_vae_decode import LTTiledVAEDecode

NODE_CLASS_MAPPINGS = {
    "LTTiledVAEDecode": LTTiledVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTTiledVAEDecode": "🅛🅣 Tiled VAE Decode",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
