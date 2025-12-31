"""Smart Image Processing utilities for a document scanner demo.

Implements:
- Image Enhancement: histogram equalization, contrast stretching
- Image Restoration: denoising (Gaussian, median), non-local means
- Geometric Transformations: rotate, translate, scale
- Image Segmentation: Otsu thresholding, Canny edge detection
- Compression helpers: save as JPEG/PNG with compression options
- Color Processing: color space conversions, saturation enhancement
- Feature Extraction: Sobel, Canny, Local Binary Patterns (LBP)

Usage: run this file as a script or import functions in a notebook.
"""
from __future__ import annotations

import os
from typing import Tuple, Optional

import numpy as np
import cv2

try:
    from skimage import exposure, color
    from skimage.feature import local_binary_pattern
except Exception:
    exposure = None
    color = None
    local_binary_pattern = None

try:
    from skimage.restoration import denoise_nl_means, estimate_sigma
except Exception:
    denoise_nl_means = None
    estimate_sigma = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_image(path: str, color: bool = True) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(path: str, img: np.ndarray) -> None:
    # Convert RGB to BGR for OpenCV write
    dirname = os.path.dirname(path)
    if dirname:
        ensure_dir(dirname)
    if img.ndim == 3 and img.shape[2] == 3:
        to_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        to_write = img
    cv2.imwrite(path, to_write)


# ------------------ Enhancement ------------------
def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Perform histogram equalization. Works for grayscale and RGB (by Y channel)."""
    if img.ndim == 2:
        eq = cv2.equalizeHist(img)
        return eq
    # for color image: convert to YCrCb and equalize Y
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)


def contrast_stretching(img: np.ndarray, in_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Simple linear contrast stretching to full [0,255]."""
    arr = img.astype(np.float32)
    if in_range is None:
        vmin, vmax = arr.min(), arr.max()
    else:
        vmin, vmax = in_range
    if vmax == vmin:
        return img.copy()
    stretched = (arr - vmin) * (255.0 / (vmax - vmin))
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    return stretched


def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction. gamma>1 brightens backgrounds; <1 darkens.
    Works on grayscale or RGB. Uses LUT for speed.
    """
    if gamma <= 0:
        return img.copy()
    inv_gamma = 1.0 / gamma
    # Build LUT for 0..255
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    if img.ndim == 2:
        return cv2.LUT(img, lut)
    # apply per-channel on RGB
    chans = cv2.split(img)
    chans = [cv2.LUT(c, lut) for c in chans]
    return cv2.merge(chans)

# ------------------ Restoration ------------------
def denoise_gaussian(img: np.ndarray, ksize: int = 3, sigma: float = 0) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def denoise_median(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    if img.ndim == 2:
        return cv2.medianBlur(img, ksize)
    # apply per channel
    chans = cv2.split(img)
    chans = [cv2.medianBlur(c, ksize) for c in chans]
    return cv2.merge(chans)


def denoise_nlmeans_wrapper(img: np.ndarray, h: float = 0.8) -> np.ndarray:
    """Denoise with non-local means, supporting both old and new skimage APIs.

    - For scikit-image >= 0.19, use `channel_axis`.
    - For older versions, use `multichannel`.
    - If scikit-image unavailable, fall back to OpenCV fastNlMeans.
    """
    if denoise_nl_means is None or estimate_sigma is None:
        # fallback: use OpenCV fastNlMeansDenoising
        if img.ndim == 3:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out = cv2.fastNlMeansDenoisingColored(bgr, None, h * 10, h * 10, 7, 21)
            return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        else:
            return cv2.fastNlMeansDenoising(img, None, h * 10, 7, 21)

    # skimage expects float in [0,1]
    img_f = img.astype(np.float32) / 255.0
    is_color = img.ndim == 3

    # Estimate noise sigma with compatibility for API changes
    try:
        # New API (>=0.19): channel_axis
        sigma_est = estimate_sigma(img_f, channel_axis=2 if is_color else None)
    except TypeError:
        # Old API: multichannel
        sigma_est = estimate_sigma(img_f, multichannel=is_color)

    # Convert sigma to scalar for h scaling
    try:
        sigma_scalar = float(np.mean(sigma_est))
    except Exception:
        sigma_scalar = float(sigma_est)

    # Call NL-means with compatibility
    try:
        den = denoise_nl_means(
            img_f,
            h=h * sigma_scalar,
            fast_mode=True,
            patch_size=5,
            patch_distance=6,
            channel_axis=2 if is_color else None,
        )
    except TypeError:
        den = denoise_nl_means(
            img_f,
            h=h * sigma_scalar,
            fast_mode=True,
            patch_size=5,
            patch_distance=6,
            multichannel=is_color,
        )

    den = np.clip(den * 255, 0, 255).astype(np.uint8)
    return den


# ------------------ Geometric Transforms ------------------
def rotate(img: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None, scale: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)


def translate(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))


def scale(img: np.ndarray, sx: float, sy: Optional[float] = None) -> np.ndarray:
    if sy is None:
        sy = sx
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * sx), int(h * sy)), interpolation=cv2.INTER_LINEAR)


# ------------------ Segmentation ------------------
def otsu_threshold(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def canny_edges(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    return cv2.Canny(gray, low, high)


# ------------------ Morphology ------------------
def morphological_opening(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Perform opening (erode then dilate) to remove small specks. Works on grayscale or binary."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def morphological_erosion(img: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """Grayscale/binary erosion to thin strokes and suppress small bright noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.erode(img, kernel, iterations=iterations)


def binary_closing(bin_img: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """Binary closing (dilate then erode) to fill small gaps/holes in foreground.
    Ensures input is binary 0/255 before applying.
    """
    if bin_img.ndim == 3:
        bin_img = rgb_to_gray(bin_img)
    # Ensure binary
    _, b = cv2.threshold(bin_img, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=iterations)


# ------------------ Compression ------------------
def save_jpeg_compressed(path: str, img: np.ndarray, quality: int = 90) -> None:
    ensure_dir(os.path.dirname(path) or "")
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
    cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])


def save_png_compressed(path: str, img: np.ndarray, compression: int = 3) -> None:
    ensure_dir(os.path.dirname(path) or "")
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
    cv2.imwrite(path, bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), int(compression)])


# ------------------ Color Processing ------------------
def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def enhance_saturation(img: np.ndarray, factor: float = 1.2) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# ------------------ Feature Extraction ------------------
def sobel_edges(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        gray = rgb_to_gray(img)
    else:
        gray = img
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = np.uint8(np.clip((mag / mag.max()) * 255, 0, 255))
    return mag


def lbp_texture(img: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    if local_binary_pattern is None:
        raise RuntimeError("skimage.feature.local_binary_pattern not available; install scikit-image")
    if img.ndim == 3:
        gray = rgb_to_gray(img)
    else:
        gray = img
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    # normalize to 0..255
    lbp = lbp.astype(np.float32)
    lbp = 255 * (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-9)
    return lbp.astype(np.uint8)


# ------------------ Utilities for Ordered Pipeline ------------------
def to_grayscale(img: np.ndarray) -> np.ndarray:
    return rgb_to_gray(img) if img.ndim == 3 else img


def auto_deskew(img: np.ndarray, max_angle: float = 15.0) -> Tuple[np.ndarray, float]:
    """Estimate and correct small skew using Hough transform on edges.
    Returns (deskewed_image, estimated_angle_deg). Positive angle means image rotated counter-clockwise.
    """
    gray = to_grayscale(img)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=150)
    if lines is None:
        return img, 0.0
    angles = []
    for rho_theta in lines[:200]:
        rho, theta = rho_theta[0]
        angle = (theta * 180.0 / np.pi) - 90.0
        # Normalize angle to [-90, 90]
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180
        # We care about small skews near 0
        if -max_angle <= angle <= max_angle:
            angles.append(angle)
    if not angles:
        return img, 0.0
    med_angle = float(np.median(angles))
    # Rotate opposite to measured skew
    corrected = rotate(img, -med_angle)
    return corrected, med_angle


def feature_vector(gray_or_binary: np.ndarray) -> np.ndarray:
    """Extract a compact feature vector for a document region.
    Prefer LBP histogram; fallback to Sobel magnitude histogram if LBP unavailable.
    Returns a 1D float32 vector normalized to sum=1.
    """
    if gray_or_binary.ndim == 3:
        gray = rgb_to_gray(gray_or_binary)
    else:
        gray = gray_or_binary

    if local_binary_pattern is not None:
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        bins = np.arange(0, 8 + 2 + 1)  # P+2 bins for uniform LBP
        hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, 8 + 2))
    else:
        sob = sobel_edges(gray)
        hist, _ = np.histogram(sob.ravel(), bins=16, range=(0, 255))

    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def process_demo(input_path: str, out_dir: str = "outputs") -> None:
    """Run a sequence of processing steps on a sample document image and save outputs."""
    ensure_dir(out_dir)
    img = load_image(input_path, color=True)

    # Enhancement
    eq = histogram_equalization(img)
    save_image(os.path.join(out_dir, "enh_equalized.png"), eq)

    stretched = contrast_stretching(img)
    save_image(os.path.join(out_dir, "enh_contrast_stretched.png"), stretched)

    # Restoration / denoising
    den_gauss = denoise_gaussian(img, ksize=3)
    save_image(os.path.join(out_dir, "rest_gaussian.png"), den_gauss)

    den_med = denoise_median(img, ksize=3)
    save_image(os.path.join(out_dir, "rest_median.png"), den_med)

    den_nl = denoise_nlmeans_wrapper(img)
    save_image(os.path.join(out_dir, "rest_nlmeans.png"), den_nl)

    # Geometric
    rot = rotate(img, angle=2.0)
    save_image(os.path.join(out_dir, "geom_rotated.png"), rot)

    scaled = scale(img, 0.5)
    save_image(os.path.join(out_dir, "geom_scaled.png"), scaled)

    # Segmentation
    otsu = otsu_threshold(img)
    save_image(os.path.join(out_dir, "seg_otsu.png"), otsu)

    edges = canny_edges(img)
    save_image(os.path.join(out_dir, "seg_edges.png"), edges)

    # Compression
    save_jpeg_compressed(os.path.join(out_dir, "compressed.jpg"), img, quality=85)
    save_png_compressed(os.path.join(out_dir, "compressed.png"), img, compression=3)

    # Color processing
    hsv = rgb_to_hsv(img)
    # save HSV as RGB visualization
    save_image(os.path.join(out_dir, "color_hsv_as_rgb.png"), cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    saturated = enhance_saturation(img, factor=1.3)
    save_image(os.path.join(out_dir, "color_saturated.png"), saturated)

    # Features
    sob = sobel_edges(img)
    save_image(os.path.join(out_dir, "feat_sobel.png"), sob)
    try:
        lbp = lbp_texture(img)
        save_image(os.path.join(out_dir, "feat_lbp.png"), lbp)
    except RuntimeError:
        # skip if scikit-image not available
        pass


# ------------------ Sharpening & Pipeline ------------------
def unsharp_sharpen(img: np.ndarray, radius: int = 3, amount: float = 1.2, threshold: int = 0) -> np.ndarray:
    """Unsharp masking for image sharpening.
    radius: Gaussian blur kernel size (approximate).
    amount: scaling of the high-frequency components.
    threshold: optional threshold to suppress small differences (0=off).
    """
    if radius % 2 == 0:
        radius += 1  # kernel size must be odd
    blurred = cv2.GaussianBlur(img, (radius, radius), 0)
    # high-frequency component
    high_freq = cv2.subtract(img, blurred)
    if threshold > 0:
        mask = cv2.cvtColor(high_freq, cv2.COLOR_RGB2GRAY) if high_freq.ndim == 3 else high_freq
        _, mask_bin = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
        if high_freq.ndim == 3:
            mask_bin = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2RGB)
        high_freq = cv2.bitwise_and(high_freq, mask_bin)
    sharpened = cv2.addWeighted(img, 1.0, high_freq, amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def process_document_pipeline(input_path: str, out_dir: str = "outputs", save_intermediate: bool = True) -> dict:
    """Process a document image through a recommended enhancement pipeline.

    Sequence rationale:
    1. Load image
    2. Median filter: robust salt & pepper noise removal without blurring edges heavily.
    3. Histogram equalization: contrast normalization after noise suppression to avoid amplifying noise.
    4. Unsharp mask: sharpen structural edges once contrast is balanced; doing it earlier would amplify noise.
    5. Otsu threshold: binarize clean, contrast-boosted, sharpened image for text clarity.
    6. Canny edges: extract precise edges from sharpened image (after enhancement).
    7. Sobel + optional LBP: feature extraction on enhanced grayscale for downstream analysis.
    8. Compression: store final versions efficiently.
    Returns a dict of intermediate arrays.
    """
    ensure_dir(out_dir)
    img = load_image(input_path, color=True)
    results = {"original": img}

    # 1 Noise removal (median better for document salt/pepper artifacts)
    den_med = denoise_median(img, ksize=3)
    results["denoised_median"] = den_med
    if save_intermediate:
        save_image(os.path.join(out_dir, "pipeline_01_denoised_median.png"), den_med)

    # 2 Contrast normalization
    eq = histogram_equalization(den_med)
    results["equalized"] = eq
    if save_intermediate:
        save_image(os.path.join(out_dir, "pipeline_02_equalized.png"), eq)

    # 3 Sharpening
    sharp = unsharp_sharpen(eq, radius=3, amount=1.3, threshold=0)
    results["sharpened"] = sharp
    if save_intermediate:
        save_image(os.path.join(out_dir, "pipeline_03_sharpened.png"), sharp)

    # 4 Binarization
    otsu = otsu_threshold(sharp)
    results["otsu"] = otsu
    if save_intermediate:
        save_image(os.path.join(out_dir, "pipeline_04_otsu.png"), otsu)

    # 5 Edge detection
    edges = canny_edges(sharp)
    results["edges"] = edges
    if save_intermediate:
        save_image(os.path.join(out_dir, "pipeline_05_edges.png"), edges)

    # 6 Feature extraction
    sob = sobel_edges(sharp)
    results["sobel"] = sob
    if save_intermediate:
        save_image(os.path.join(out_dir, "pipeline_06_sobel.png"), sob)

    try:
        lbp = lbp_texture(sharp)
        results["lbp"] = lbp
        if save_intermediate:
            save_image(os.path.join(out_dir, "pipeline_07_lbp.png"), lbp)
    except RuntimeError:
        pass

    # 7 Compression of final sharpened & binarized versions
    save_jpeg_compressed(os.path.join(out_dir, "pipeline_final_sharpened.jpg"), sharp, quality=85)
    save_png_compressed(os.path.join(out_dir, "pipeline_final_otsu.png"), otsu, compression=3)

    return results


def process_document_pipeline_ordered(input_path: str, out_dir: str = "outputs", save_intermediate: bool = True) -> dict:
    """Run the 7-step pipeline as specified by the user.

    Steps:
    1. Color Processing: RGB -> Grayscale
    2. Restoration: Median denoise
    3. Enhancement: Histogram equalization
    4. Geometric Transform: Auto deskew (alignment)
    5. Segmentation: Otsu thresholding (binary)
    6. Feature Extraction: LBP (or Sobel) histogram vector
    7. Compression: Save compressed outputs
    """
    ensure_dir(out_dir)
    img_color = load_image(input_path, color=True)
    results: dict = {"original": img_color}

    # 1. Color Processing: to grayscale
    gray = to_grayscale(img_color)
    results["step1_gray"] = gray
    if save_intermediate:
        save_image(os.path.join(out_dir, "step_01_grayscale.png"), gray)

    # 2. Restoration: median filter
    den = denoise_median(gray, ksize=3)
    results["step2_denoised"] = den
    if save_intermediate:
        save_image(os.path.join(out_dir, "step_02_denoised_median.png"), den)

    # 3. Enhancement: histogram equalization
    eq = histogram_equalization(den)
    results["step3_equalized"] = eq
    if save_intermediate:
        save_image(os.path.join(out_dir, "step_03_equalized.png"), eq)

    # 4. Geometric: auto deskew
    # Work on an RGB version for rotation fidelity
    eq_rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB) if eq.ndim == 2 else eq
    deskewed_rgb, angle = auto_deskew(eq_rgb)
    results["step4_deskewed_rgb"] = deskewed_rgb
    results["deskew_angle_deg"] = angle
    if save_intermediate:
        save_image(os.path.join(out_dir, "step_04_deskewed.png"), deskewed_rgb)

    # 5. Segmentation: Otsu on deskewed grayscale
    deskewed_gray = to_grayscale(deskewed_rgb)
    binary = otsu_threshold(deskewed_gray)
    results["step5_binary"] = binary
    if save_intermediate:
        save_image(os.path.join(out_dir, "step_05_binary_otsu.png"), binary)

    # 6. Feature Extraction: vector
    vec = feature_vector(deskewed_gray)
    results["step6_feature_vector"] = vec
    if save_intermediate:
        np.savetxt(os.path.join(out_dir, "step_06_features_lbp_or_sobel.csv"), vec[None, :], delimiter=",", fmt="%.6f")

    # 7. Compression: save compressed outputs
    save_jpeg_compressed(os.path.join(out_dir, "step_07_deskewed.jpg"), deskewed_rgb, quality=85)
    save_png_compressed(os.path.join(out_dir, "step_07_binary.png"), binary, compression=3)

    return results


def process_document_pipeline_doc_clean(input_path: str, out_dir: str = "outputs", gamma: float = 1.4, ksize: int = 3, save_intermediate: bool = True) -> dict:
    """Run the 4-step document cleaning sequence:

    1. Median Filter (Restoration): removes noise without blurring text edges.
    2. Gamma Correction (Enhancement): brightens background and reduces bleed-through/shadows.
    3. Morphological Opening (Morphology): removes small artifacts/specks.
    4. Otsu's Method (Segmentation): final black-vs-white decision.
    """
    ensure_dir(out_dir)
    img_color = load_image(input_path, color=True)
    gray = rgb_to_gray(img_color)
    results = {"original": img_color, "gray": gray}

    # 1. Median Filter
    den = denoise_median(gray, ksize=ksize)
    results["step1_median"] = den
    if save_intermediate:
        save_image(os.path.join(out_dir, "docclean_01_median.png"), den)

    # 2. Gamma Correction (gamma>1 to brighten background and wash bleed-through)
    gc = gamma_correction(den, gamma=gamma)
    results["step2_gamma"] = gc
    if save_intermediate:
        save_image(os.path.join(out_dir, "docclean_02_gamma.png"), gc)

    # 3. Morphological Opening
    opened = morphological_opening(gc, ksize=ksize)
    results["step3_opening"] = opened
    if save_intermediate:
        save_image(os.path.join(out_dir, "docclean_03_opening.png"), opened)

    # 4. Otsu Threshold
    binary = otsu_threshold(opened)
    results["step4_otsu"] = binary
    if save_intermediate:
        save_image(os.path.join(out_dir, "docclean_04_otsu.png"), binary)

    # Save compressed results
    save_png_compressed(os.path.join(out_dir, "docclean_binary.png"), binary, compression=3)
    save_jpeg_compressed(os.path.join(out_dir, "docclean_gamma.jpg"), cv2.cvtColor(gc, cv2.COLOR_GRAY2RGB), quality=85)

    return results


def process_document_pipeline_morph_seq(
    input_path: str,
    out_dir: str = "outputs",
    ksize: int = 3,
    iterations: int = 1,
    save_intermediate: bool = True,
) -> dict:
    """Run the requested 4-step sequence:

    1) Grayscale (Color Processing)
    2) Grayscale Erosion (Morphological Processing)
    3) Otsu’s Binarization (Segmentation)
    4) Binary Closing (Morphological Processing)
    """
    ensure_dir(out_dir)
    color_img = load_image(input_path, color=True)
    results = {"original": color_img}

    # 1. Grayscale
    gray = to_grayscale(color_img)
    results["step1_gray"] = gray
    if save_intermediate:
        save_image(os.path.join(out_dir, "morphseq_01_gray.png"), gray)

    # 2. Grayscale Erosion
    eroded = morphological_erosion(gray, ksize=ksize, iterations=iterations)
    results["step2_eroded"] = eroded
    if save_intermediate:
        save_image(os.path.join(out_dir, "morphseq_02_eroded.png"), eroded)

    # 3. Otsu’s Binarization
    binary = otsu_threshold(eroded)
    results["step3_otsu"] = binary
    if save_intermediate:
        save_image(os.path.join(out_dir, "morphseq_03_otsu.png"), binary)

    # 4. Binary Closing
    closed = binary_closing(binary, ksize=ksize, iterations=iterations)
    results["step4_closed"] = closed
    if save_intermediate:
        save_image(os.path.join(out_dir, "morphseq_04_closed.png"), closed)

    # Compressed output
    save_png_compressed(os.path.join(out_dir, "morphseq_closed.png"), closed, compression=3)

    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Smart Image Processing demo for document scanner")
    p.add_argument("input", help="Input image path")
    p.add_argument("--out", default="outputs", help="Output directory")
    p.add_argument("--pipeline", action="store_true", help="Run legacy document enhancement pipeline")
    p.add_argument("--pipeline-ordered", action="store_true", help="Run the 7-step ordered pipeline (Color→Restoration→Enhancement→Geometric→Segmentation→Features→Compression)")
    p.add_argument("--doc-clean", action="store_true", help="Run 4-step doc cleaning pipeline (Median→Gamma→Opening→Otsu)")
    p.add_argument("--morph-seq", action="store_true", help="Run 4-step morphological sequence (Grayscale→Erosion→Otsu→Closing)")
    p.add_argument("--gamma", type=float, default=1.4, help="Gamma value used for doc-clean pipeline (default 1.4)")
    p.add_argument("--ksize", type=int, default=3, help="Kernel size for median/opening in doc-clean pipeline (default 3)")
    args = p.parse_args()
    if args.pipeline_ordered:
        process_document_pipeline_ordered(args.input, args.out, save_intermediate=True)
    elif args.pipeline:
        process_document_pipeline(args.input, args.out, save_intermediate=True)
    elif args.doc_clean:
        process_document_pipeline_doc_clean(args.input, args.out, gamma=args.gamma, ksize=args.ksize, save_intermediate=True)
    elif args.morph_seq:
        # Reuse ksize for structuring element, and gamma unused here
        process_document_pipeline_morph_seq(args.input, args.out, ksize=args.ksize, iterations=1, save_intermediate=True)
    else:
        process_demo(args.input, args.out)
