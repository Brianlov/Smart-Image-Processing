"""Standalone 4-step document morphology pipeline (fixed ksize=3).

Sequence:
1) Grayscale (Color Processing)
2) Grayscale Erosion (Morphological Processing)
3) Otsu’s Binarization (Segmentation)
4) Binary Closing (Morphological Processing)

Usage:
  python morph_seq.py <input_image> --out outputs
"""
from __future__ import annotations

import os
from typing import Dict

import cv2
import numpy as np


KSIZE = 2
ITERATIONS = 1


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    # Convert BGR to RGB for internal consistency if needed
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path: str, img: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    to_write = img
    if img.ndim == 3 and img.shape[2] == 3:
        to_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, to_write)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def grayscale_erosion(gray: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KSIZE, KSIZE))
    return cv2.erode(gray, kernel, iterations=ITERATIONS)


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 3:
        gray = to_grayscale(gray)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 


def binary_closing(bin_img: np.ndarray) -> np.ndarray:
    if bin_img.ndim == 3:
        bin_img = to_grayscale(bin_img)
    # ensure binary
    _, b = cv2.threshold(bin_img, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KSIZE, KSIZE))
    return cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=ITERATIONS)


def process_morph_seq(input_path: str, out_dir: str = "outputs", save_intermediate: bool = True) -> Dict[str, np.ndarray]:
    ensure_dir(out_dir)
    color_img = load_image(input_path)
    results: Dict[str, np.ndarray] = {"original": color_img}

    # 1) Grayscale
    gray = to_grayscale(color_img)
    results["step1_gray"] = gray
    if save_intermediate:
        save_image(os.path.join(out_dir, "morphseq_01_gray.png"), gray)

    # 2) Grayscale Erosion
    eroded = grayscale_erosion(gray)
    results["step2_eroded"] = eroded
    if save_intermediate:
        save_image(os.path.join(out_dir, "morphseq_02_eroded.png"), eroded)

    # 3) Otsu’s Binarization
    binary = otsu_binarize(eroded)
    results["step3_otsu"] = binary
    if save_intermediate:
        save_image(os.path.join(out_dir, "morphseq_03_otsu.png"), binary)

    # 4) Binary Closing
    closed = binary_closing(binary)
    results["step4_closed"] = closed
    if save_intermediate:
        save_image(os.path.join(out_dir, "morphseq_04_closed.png"), closed)

    # Compressed PNG of final binary (compression level 3)
    cv2.imwrite(
        os.path.join(out_dir, "morphseq_closed.png"),
        closed,
        [int(cv2.IMWRITE_PNG_COMPRESSION), int(3)],
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="4-step document morphology pipeline (ksize=3)")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--out", default="outputs", help="Output directory")
    args = parser.parse_args()

    process_morph_seq(args.input, args.out, save_intermediate=True)
    print(f"Done. Results saved to {args.out}")
