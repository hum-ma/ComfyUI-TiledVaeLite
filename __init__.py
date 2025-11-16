from .decoder_noise import DecoderNoise
from .nodes_registry import NODE_CLASS_MAPPINGS as RUNTIME_NODE_CLASS_MAPPINGS
from .nodes_registry import (
    NODE_DISPLAY_NAME_MAPPINGS as RUNTIME_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_registry import NODES_DISPLAY_NAME_PREFIX, camel_case_to_spaces
from .tiled_vae_decode import LTTiledVAEDecode

# Static node mappings, required for ComfyUI-Manager mapping to work
NODE_CLASS_MAPPINGS = {
    "Set VAE Decoder Noise": DecoderNoise,
    "TiledVAEDecode": LTTiledVAEDecode,
}

# Consistent display names between static and dynamic node mappings in nodes_registry.py,
# to prevent ComfyUI initializing them with default display names.
NODE_DISPLAY_NAME_MAPPINGS = {
    name: f"{NODES_DISPLAY_NAME_PREFIX} {camel_case_to_spaces(name)}"
    for name in NODE_CLASS_MAPPINGS.keys()
}

# Update with runtime mappings (these will override static mappings if there are any differences)
NODE_CLASS_MAPPINGS.update(RUNTIME_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(RUNTIME_NODE_DISPLAY_NAME_MAPPINGS)

# Export so that ComfyUI can pick them up.
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
