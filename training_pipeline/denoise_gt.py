"""
Denoise path-traced ground truth textures.
Removes Monte Carlo variance from participating media rendering
while preserving edges and fine detail.

Uses non-local means denoising which is ideal for this type of noise:
it finds similar patches across the image and averages them,
exploiting the self-similarity of rendered content to separate
signal from stochastic variance.

Input:  ./ground_truth_raw/*.png
Output: ./ground_truth/*.png
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# --- Configuration ---
# Non-local means parameters (tuned for path-tracing noise)
# h: filter strength. Higher removes more noise but risks losing detail.
#    For path-traced participating media, 6-10 is usually the sweet spot.
# templateWindowSize: patch size for comparison (must be odd)
# searchWindowSize: search area for similar patches (must be odd)
NLM_H = 1.8
NLM_TEMPLATE = 7  # 7x7 patches capture enough local structure
NLM_SEARCH = 21  # 21x21 search window


def denoise_image(img):
    """Denoise a single image using non-local means."""

    if img is None:
        return None

    # NLM works on 8-bit images natively
    # For 16-bit PNGs, process at higher precision
    is_16bit = img.dtype == np.uint16

    if is_16bit:
        # Convert to float, denoise, convert back
        img_float = img.astype(np.float32) / 65535.0
        img_8bit = (img_float * 255).clip(0, 255).astype(np.uint8)
    else:
        img_8bit = img

    if len(img_8bit.shape) == 3 and img_8bit.shape[2] == 3:
        # Color image - use color-aware NLM
        # denoised_8bit = cv2.fastNlMeansDenoisingColored(
        #     img_8bit,
        #     None,
        #     h=NLM_H,
        #     hForColorComponents=NLM_H,
        #     templateWindowSize=NLM_TEMPLATE,
        #     searchWindowSize=NLM_SEARCH
        # )
        denoised_8bit = cv2.fastNlMeansDenoisingColored(
            img_8bit, None, NLM_H, NLM_H, NLM_TEMPLATE, NLM_SEARCH
        )
    elif len(img_8bit.shape) == 3 and img_8bit.shape[2] == 4:
        # RGBA - denoise RGB, keep alpha untouched
        rgb = img_8bit[:, :, :3]
        alpha = img[:, :, 3:]  # keep original precision alpha

        # rgb_denoised = cv2.fastNlMeansDenoisingColored(
        #     rgb,
        #     None,
        #     h=NLM_H,
        #     hForColorComponents=NLM_H,
        #     templateWindowSize=NLM_TEMPLATE,
        #     searchWindowSize=NLM_SEARCH
        # )
        rgb_denoised = cv2.fastNlMeansDenoisingColored(
            rgb, None, NLM_H, NLM_H, NLM_TEMPLATE, NLM_SEARCH
        )

        if is_16bit:
            rgb_out = (rgb_denoised.astype(np.float32) / 255.0 * 65535).clip(0, 65535).astype(np.uint16)
            return np.concatenate([rgb_out, alpha], axis=2)
        else:
            return np.concatenate([rgb_denoised, alpha], axis=2)
    else:
        # Grayscale
        denoised_8bit = cv2.fastNlMeansDenoising(
            img_8bit,
            None,
            h=NLM_H,
            templateWindowSize=NLM_TEMPLATE,
            searchWindowSize=NLM_SEARCH
        )

    if is_16bit:
        # Map denoised 8-bit back to 16-bit range
        denoised = (denoised_8bit.astype(np.float32) / 255.0 * 65535).clip(0, 65535).astype(np.uint16)
        return denoised

    return denoised_8bit

def denoise_image_bilateral(img):
    if len(img.shape) == 3 and img.shape[2] == 4:
        rgb = img[:, :, :3]
        alpha = img[:, :, 3:]
        rgb_denoised = cv2.bilateralFilter(rgb, d=5, sigmaColor=18, sigmaSpace=5)
        return np.concatenate([rgb_denoised, alpha], axis=2)
    else:
        return cv2.bilateralFilter(img, d=5, sigmaColor=18, sigmaSpace=5)

def main():
    script_dir = Path(__file__).parent
    input_dir = script_dir / "ground_truth_raw"
    output_dir = script_dir / "ground_truth_extend"

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(exist_ok=True)

    png_files = sorted(input_dir.glob("*.png"))

    if not png_files:
        print(f"No PNG files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(png_files)} PNG files")
    print(f"NLM params: h={NLM_H}, template={NLM_TEMPLATE}x{NLM_TEMPLATE}, search={NLM_SEARCH}x{NLM_SEARCH}")
    print()

    for i, filepath in enumerate(png_files):
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"[{i + 1}/{len(png_files)}] FAILED to read: {filepath.name}")
            continue

        denoised = denoise_image(img)
        #denoised = denoise_image_bilateral(img)

        out_path = output_dir / filepath.name
        cv2.imwrite(str(out_path), denoised)

        print(f"[{i + 1}/{len(png_files)}] {filepath.name} ({img.shape[1]}x{img.shape[0]}, {img.dtype})")

    print(f"\nDone. Denoised images saved to {output_dir}")


if __name__ == "__main__":
    main()