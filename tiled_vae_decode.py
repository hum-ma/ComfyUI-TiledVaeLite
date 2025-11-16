import torch
import time
import logging

from .nodes_registry import comfy_node
from comfy.utils import ProgressBar


@comfy_node(
    name="TiledVAEDecode",
)
class LTTiledVAEDecode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "latents": ("LATENT",),
                "horizontal_tiles": ("INT", {"default": 2, "min": 1, "max": 6}),
                "vertical_tiles": ("INT", {"default": 2, "min": 1, "max": 6}),
                "overlap": ("INT", {"default": 4, "min": 1, "max": 8}),
                "last_frame_fix": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "decode"

    CATEGORY = "latent"

    def decode(
        self,
        vae,
        latents,
        horizontal_tiles,
        vertical_tiles,
        overlap,
        last_frame_fix,
    ):
        # Get the latent samples
        samples = latents["samples"]

        if last_frame_fix:
            # Repeat the last frame along dimension 2 (frames)
            # samples: [batch, channels, frames, height, width]
            last_frame = samples[
                :, :, -1:, :, :
            ]  # shape: [batch, channels, 1, height, width]
            samples = torch.cat([samples, last_frame], dim=2)

        batch, channels, frames, height, width = samples.shape
        time_scale_factor, width_scale_factor, height_scale_factor = (
            vae.downscale_index_formula
        )
        image_frames = 1 + (frames - 1) * time_scale_factor

        # Calculate output image dimensions
        output_height = height * height_scale_factor
        output_width = width * width_scale_factor

        # Calculate tile sizes with overlap
        base_tile_height = (height + (vertical_tiles - 1) * overlap) // vertical_tiles
        base_tile_width = (width + (horizontal_tiles - 1) * overlap) // horizontal_tiles

        # Initialize output tensor and weight tensor
        # VAE decode returns images in format [batch, height, width, channels]
        output = None
        weights = None

        pbar = ProgressBar(horizontal_tiles * vertical_tiles)

        mult = 8 # to display sizes in pixels
        time_init = time.perf_counter()
        # Process each tile
        for v in range(vertical_tiles):
            for h in range(horizontal_tiles):
                # Calculate tile boundaries
                h_start = h * (base_tile_width - overlap)
                v_start = v * (base_tile_height - overlap)

                # Adjust end positions for edge tiles
                h_end = (
                    min(h_start + base_tile_width, width)
                    if h < horizontal_tiles - 1
                    else width
                )
                v_end = (
                    min(v_start + base_tile_height, height)
                    if v < vertical_tiles - 1
                    else height
                )

                # Calculate actual tile dimensions
                tile_height = v_end - v_start
                tile_width = h_end - h_start

                logging.info(f"Processing VAE decode tile at row {v}, col {h}:  Position: ({v_start*mult}:{v_end*mult}, {h_start*mult}:{h_end*mult}), Size: {tile_height*mult}x{tile_width*mult}")
                time_before = time.perf_counter()

                # Extract tile
                tile = samples[:, :, :, v_start:v_end, h_start:h_end]

                # Create tile latents dict
                tile_latents = {"samples": tile}

                # Decode the tile
                decoded_tile = vae.decode(tile_latents["samples"])

                # Initialize output tensors on first tile
                if output is None:
                    output = torch.zeros(
                        (
                            batch,
                            image_frames,
                            output_height,
                            output_width,
                            decoded_tile.shape[-1],
                        ),
                        device=decoded_tile.device,
                        dtype=decoded_tile.dtype,
                    )
                    weights = torch.zeros(
                        (batch, image_frames, output_height, output_width, 1),
                        device=decoded_tile.device,
                        dtype=decoded_tile.dtype,
                    )

                # Calculate output tile boundaries
                out_h_start = v_start * height_scale_factor
                out_h_end = v_end * height_scale_factor
                out_w_start = h_start * width_scale_factor
                out_w_end = h_end * width_scale_factor

                # Create weight mask for this tile
                tile_out_height = out_h_end - out_h_start
                tile_out_width = out_w_end - out_w_start
                tile_weights = torch.ones(
                    (batch, image_frames, tile_out_height, tile_out_width, 1),
                    device=decoded_tile.device,
                    dtype=decoded_tile.dtype,
                )

                # Calculate overlap regions in output space
                overlap_out_h = overlap * height_scale_factor
                overlap_out_w = overlap * width_scale_factor

                # Apply horizontal blending weights
                if h > 0:  # Left overlap
                    h_blend = torch.linspace(
                        0, 1, overlap_out_w, device=decoded_tile.device
                    )
                    tile_weights[:, :, :, :overlap_out_w, :] *= h_blend.view(
                        1, 1, 1, -1, 1
                    )
                if h < horizontal_tiles - 1:  # Right overlap
                    h_blend = torch.linspace(
                        1, 0, overlap_out_w, device=decoded_tile.device
                    )
                    tile_weights[:, :, :, -overlap_out_w:, :] *= h_blend.view(
                        1, 1, 1, -1, 1
                    )

                # Apply vertical blending weights
                if v > 0:  # Top overlap
                    v_blend = torch.linspace(
                        0, 1, overlap_out_h, device=decoded_tile.device
                    )
                    tile_weights[:, :, :overlap_out_h, :, :] *= v_blend.view(
                        1, 1, -1, 1, 1
                    )
                if v < vertical_tiles - 1:  # Bottom overlap
                    v_blend = torch.linspace(
                        1, 0, overlap_out_h, device=decoded_tile.device
                    )
                    tile_weights[:, :, -overlap_out_h:, :, :] *= v_blend.view(
                        1, 1, -1, 1, 1
                    )

                # Add weighted tile to output
                output[:, :, out_h_start:out_h_end, out_w_start:out_w_end, :] += (
                    decoded_tile * tile_weights
                )

                # Add weights to weight tensor
                weights[
                    :, :, out_h_start:out_h_end, out_w_start:out_w_end, :
                ] += tile_weights
                time_elapsed = time.perf_counter()-time_before
                logging.info("  Time: {:.2f} seconds".format(time_elapsed))

                pbar.update(1)

        # Normalize by weights
        output = output / (weights + 1e-8)

        # Reshape output to match expected format [batch * frames, height, width, channels]
        output = output.view(
            batch * image_frames, output_height, output_width, output.shape[-1]
        )

        if last_frame_fix:
            output = output[:-time_scale_factor, :, :]

        time_total = time.perf_counter()-time_init
        logging.info("VAE total decode time: {:.2f} seconds".format(time_total))
        return (output,)
