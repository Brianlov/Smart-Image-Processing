import os
import math
import argparse
from typing import Tuple, List, Optional

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
	if path:
		os.makedirs(path, exist_ok=True)


def load_image(path: str) -> np.ndarray:
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	if img is None:
		raise FileNotFoundError(f"Cannot load image: {path}")
	return img


def save_image(path: str, img: np.ndarray) -> None:
	ensure_dir(os.path.dirname(path))
	cv2.imwrite(path, img)


def resize_long_side(img: np.ndarray, scale_long: int) -> np.ndarray:
	h, w = img.shape[:2]
	if scale_long <= 0:
		return img
	long = max(h, w)
	sf = scale_long / float(long)
	new_w = int(round(w * sf))
	new_h = int(round(h * sf))
	interp = cv2.INTER_AREA if sf < 1.0 else cv2.INTER_CUBIC
	return cv2.resize(img, (new_w, new_h), interpolation=interp)


def preprocess(img: np.ndarray, bilateral_d: int = 9, bilateral_sigmaColor: float = 75, bilateral_sigmaSpace: float = 75,
			   gaussian_ksize: int = 0) -> np.ndarray:
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
	denoised = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace)
	if gaussian_ksize and gaussian_ksize > 1:
		denoised = cv2.GaussianBlur(denoised, (gaussian_ksize, gaussian_ksize), 0)
	return denoised


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
	# pts shape (4,2)
	s = pts.sum(axis=1)
	diff = np.diff(pts, axis=1).reshape(-1)
	ordered = np.zeros((4, 2), dtype=np.float32)
	ordered[0] = pts[np.argmin(s)]  # top-left
	ordered[2] = pts[np.argmax(s)]  # bottom-right
	ordered[1] = pts[np.argmin(diff)]  # top-right
	ordered[3] = pts[np.argmax(diff)]  # bottom-left
	return ordered


def _largest_quadrilateral(contours: List[np.ndarray]) -> Optional[np.ndarray]:
	max_area = 0.0
	best = None
	for c in contours:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		if len(approx) == 4:
			area = abs(cv2.contourArea(approx))
			if area > max_area:
				max_area = area
				best = approx
	if best is not None:
		return best.reshape(-1, 2).astype(np.float32)
	return None


def localize_document(img: np.ndarray, canny_low: int = 50, canny_high: int = 150,
					  min_area_ratio: float = 0.2, max_area_ratio: float = 0.98) -> Optional[np.ndarray]:
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, canny_low, canny_high)

	lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=10)
	line_img = np.zeros_like(edges)
	if lines is not None:
		for l in lines:
			x1, y1, x2, y2 = l[0]
			cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)

	edges_combined = cv2.bitwise_or(edges, line_img)
	contours, _ = cv2.findContours(edges_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Filter contours by area to avoid tiny or full-frame artifacts
	img_area = img.shape[0] * img.shape[1]
	filtered = []
	for c in contours:
		area = abs(cv2.contourArea(c))
		ratio = area / max(img_area, 1)
		if ratio >= min_area_ratio and ratio <= max_area_ratio:
			filtered.append(c)

	quad = _largest_quadrilateral(filtered if filtered else contours)
	if quad is None:
		if contours:
			c = max(contours, key=cv2.contourArea)
			rect = cv2.minAreaRect(c)
			box = cv2.boxPoints(rect).astype(np.float32)
			quad = box
		else:
			return None
	return _order_quad_points(quad)


def _a_series_ratio() -> float:
	# A-series (A4 etc.) aspect ratio: sqrt(2) (height/width in portrait)
	return math.sqrt(2.0)


def perspective_warp(img: np.ndarray, quad: np.ndarray, page: str = "A4", scale_long: int = 1600) -> np.ndarray:
	# Compute target size maintaining page ratio.
	# Determine orientation by quad's longer side.
	(tl, tr, br, bl) = quad
	w_top = np.linalg.norm(tr - tl)
	w_bottom = np.linalg.norm(br - bl)
	h_left = np.linalg.norm(bl - tl)
	h_right = np.linalg.norm(br - tr)
	width = max(int(w_top), int(w_bottom))
	height = max(int(h_left), int(h_right))

	portrait = height >= width
	if page.upper() in ("A4", "A3", "A5", "LETTER"):
		ratio = _a_series_ratio() if page.upper() != "LETTER" else (11.0 / 8.5)
	else:
		ratio = height / max(width, 1)

	if portrait:
		target_h = scale_long
		target_w = int(round(target_h / ratio))
	else:
		target_w = scale_long
		target_h = int(round(target_w * ratio))

	dst = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype=np.float32)
	M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
	warped = cv2.warpPerspective(img, M, (target_w, target_h), flags=cv2.INTER_LINEAR)
	return warped


def illumination_correction(gray: np.ndarray, method: str = "subtract", blur_frac: float = 0.02) -> np.ndarray:
	# Estimate background with large blur and normalize
	h, w = gray.shape[:2]
	base = max(15, int(round(min(h, w) * blur_frac)))
	if base % 2 == 0:
		base += 1
	bg = cv2.GaussianBlur(gray, (base, base), 0)
	if method.lower() == "divide":
		tmp = cv2.divide(gray, bg, scale=255)
		corrected = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX)
	else:
		tmp = cv2.subtract(gray, bg)
		corrected = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX)
	return corrected


def adaptive_binarize(gray: np.ndarray, block_size: int = 35, C: int = 10, method: str = "gaussian") -> np.ndarray:
	if block_size % 2 == 0:
		block_size += 1
	algo = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method.lower() == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
	th = cv2.adaptiveThreshold(gray, 255, algo, cv2.THRESH_BINARY, block_size, C)
	return th


def contrast_stretch(gray: np.ndarray) -> np.ndarray:
	return cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


def _compute_ink_mask(gray: np.ndarray,
					  mask_blur_ksize: int = 61,
					  blackhat_ksize: int = 9,
					  blackhat_vertical_ratio: float = 2.0,
					  dilate_iters: int = 1,
					  threshold_offset: int = 8) -> np.ndarray:
	# 1) Background blur subtract to emphasize dark ink
	if mask_blur_ksize % 2 == 0:
		mask_blur_ksize += 1
	bg = cv2.GaussianBlur(gray, (mask_blur_ksize, mask_blur_ksize), 0)
	ink_sub = cv2.subtract(bg, gray)
	ink_sub = cv2.normalize(ink_sub, None, 0, 255, cv2.NORM_MINMAX)
	t_sub, _ = cv2.threshold(ink_sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	t_sub = max(0, int(round(t_sub - threshold_offset)))
	_, mask_sub = cv2.threshold(ink_sub, t_sub, 255, cv2.THRESH_BINARY)

	# 2) Black-hat morphological to catch dark strokes on light background (vertically biased)
	if blackhat_ksize < 3:
		blackhat_ksize = 3
	if blackhat_ksize % 2 == 0:
		blackhat_ksize += 1
	bh_h = max(3, int(round(blackhat_ksize * blackhat_vertical_ratio)))
	if bh_h % 2 == 0:
		bh_h += 1
	k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (blackhat_ksize, bh_h))
	bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_vert)
	bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
	t_bh, _ = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	t_bh = max(0, int(round(t_bh - threshold_offset)))
	_, mask_bh = cv2.threshold(bh, t_bh, 255, cv2.THRESH_BINARY)

	# Combine masks (max) so any detected ink is preserved
	combined = cv2.max(mask_sub, mask_bh)

	# 3) Light dilation to regrow skeletal strokes
	if dilate_iters > 0:
		kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
		combined = cv2.dilate(combined, kernel_dilate, iterations=dilate_iters)

	return combined


def deskew(gray: np.ndarray, canny_low: int = 50, canny_high: int = 150, max_rotate: float = 10.0) -> np.ndarray:
	edges = cv2.Canny(gray, canny_low, canny_high)
	lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
	angle_deg = 0.0
	if lines is not None and len(lines) > 0:
		angles = []
		for rho, theta in lines[:, 0, :]:
			ang = (theta * 180.0 / np.pi)
			# Convert near-horizontal lines to small angles around 0/180
			ang = (ang + 90.0) % 180.0 - 90.0
			angles.append(ang)
		if angles:
			angle_deg = float(np.median(angles))
			if abs(angle_deg) > max_rotate:
				angle_deg = 0.0

	h, w = gray.shape[:2]
	M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
	rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
	return rotated


# def morph_cleanup(bin_img: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
# 	if ksize % 2 == 0:
# 		ksize += 1
# 	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
# 	cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
# 	cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
# 	return cleaned

def morph_cleanup(bin_img: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    if ksize <= 1:
        return bin_img # Skip if kernel is too small
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    
    # Only use Closing to join broken characters
    cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # Use a smaller kernel for Opening or skip it to keep thin text
    # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1) 
    
    return cleaned


def process_document(input_path: str, out_dir: str = "outputs", page: str = "A4", scale_long: int = 1600,
					 do_ocr: bool = False,
					 bilateral_d: int = 9, bilateral_sigmaColor: float = 75, bilateral_sigmaSpace: float = 75,
					 gaussian_ksize: int = 0,
					 canny_low: int = 50, canny_high: int = 150,
					 min_area_ratio: float = 0.2, max_area_ratio: float = 0.98,
					 illum_method: str = "subtract", illum_blur_frac: float = 0.02,
					 block_size: int = 35, C: int = 10, thresh_method: str = "gaussian",
					 mask_blur_ksize: int = 51, blackhat_ksize: int = 9,
					 blackhat_vertical_ratio: float = 2.0, ink_dilate_iters: int = 1,
					 mask_thresh_offset: int = 8,
					 morph_ksize: int = 3, morph_iters: int = 1,
					 max_rotate: float = 10.0,
					 fallback_use_whole: bool = True,
					 min_quad_area_ratio: float = 0.15) -> dict:
	ensure_dir(out_dir)
	color = load_image(input_path)

	pre = preprocess(color, bilateral_d=bilateral_d, bilateral_sigmaColor=bilateral_sigmaColor,
					 bilateral_sigmaSpace=bilateral_sigmaSpace, gaussian_ksize=gaussian_ksize)
	save_image(os.path.join(out_dir, "scan_01_pre.png"), pre)

	quad = localize_document(color, canny_low=canny_low, canny_high=canny_high,
							 min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio)
	use_whole = False
	if quad is None:
		use_whole = True
	else:
		img_area = color.shape[0] * color.shape[1]
		cnt = quad.astype(np.float32).reshape(-1, 1, 2)
		quad_area = float(cv2.contourArea(cnt))
		ratio = quad_area / max(img_area, 1)
		if ratio < min_quad_area_ratio:
			use_whole = True

	if use_whole and not fallback_use_whole:
		raise RuntimeError("Quad too small or missing, and fallback disabled.")

	overlay = color.copy()
	if not use_whole and quad is not None:
		pts = quad.astype(np.int32).reshape((-1, 1, 2))
		cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
	else:
		h, w = color.shape[:2]
		full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32).reshape((-1, 1, 2))
		cv2.polylines(overlay, [full], True, (0, 165, 255), 2)
	save_image(os.path.join(out_dir, "scan_02_quad.png"), overlay)

	if not use_whole and quad is not None:
		warped = perspective_warp(color, quad, page=page, scale_long=scale_long)
	else:
		warped = resize_long_side(color, scale_long)
	save_image(os.path.join(out_dir, "scan_03_warped.png"), warped)

	warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	illum = illumination_correction(warped_gray, method=illum_method, blur_frac=illum_blur_frac)
	save_image(os.path.join(out_dir, "scan_04_illum.png"), illum)

	stretched = contrast_stretch(illum)
	save_image(os.path.join(out_dir, "scan_05_stretch.png"), stretched)

	# Emphasize dark (ink) regions and suppress non-dark via an ink mask
	ink_mask = _compute_ink_mask(
		stretched,
		mask_blur_ksize=mask_blur_ksize,
		blackhat_ksize=blackhat_ksize,
		blackhat_vertical_ratio=blackhat_vertical_ratio,
		dilate_iters=ink_dilate_iters,
		threshold_offset=mask_thresh_offset,
	)
	save_image(os.path.join(out_dir, "scan_05a_inkmask.png"), ink_mask)

	base_bin = adaptive_binarize(stretched, block_size=block_size, C=C, method=thresh_method)
	save_image(os.path.join(out_dir, "scan_06_adapt.png"), base_bin)

	# Suppress non-ink regions: force them to white in the final binary
	bin_img = base_bin.copy()
	bin_img[ink_mask == 0] = 255
	save_image(os.path.join(out_dir, "scan_06b_weighted.png"), bin_img)

	desk = deskew(bin_img, canny_low=canny_low, canny_high=canny_high, max_rotate=max_rotate)
	save_image(os.path.join(out_dir, "scan_07_deskew.png"), desk)

	clean = morph_cleanup(desk, ksize=morph_ksize, iterations=morph_iters)
	save_image(os.path.join(out_dir, "scan_08_clean.png"), clean)

	result = {
		"quad": quad,
		"warped": warped,
		"binary": clean,
	}

	if do_ocr:
		try:
			import pytesseract
			ocr_text = pytesseract.image_to_string(clean, config="--psm 6")
			txt_path = os.path.join(out_dir, "scan_ocr.txt")
			with open(txt_path, "w", encoding="utf-8") as f:
				f.write(ocr_text)
			result["ocr_text"] = ocr_text
		except Exception as e:
			result["ocr_error"] = str(e)

	return result


def main() -> None:
	parser = argparse.ArgumentParser(description="Document scanner pipeline: preprocessing, localization, warp, enhance, OCR")
	parser.add_argument("input", help="Input image path")
	parser.add_argument("--out", dest="out", default="outputs", help="Output directory")
	parser.add_argument("--page", dest="page", default="A4", help="Target page type (A4|Letter|custom)")
	parser.add_argument("--scale-long", dest="scale_long", type=int, default=1600, help="Target long-side pixels")
	parser.add_argument("--ocr", dest="ocr", action="store_true", help="Run OCR and save text")
	# Tunables
	parser.add_argument("--bilateral-d", type=int, default=9)
	parser.add_argument("--bilateral-sigma-color", type=float, default=75)
	parser.add_argument("--bilateral-sigma-space", type=float, default=75)
	parser.add_argument("--gaussian-ksize", type=int, default=0)
	parser.add_argument("--canny-low", type=int, default=50)
	parser.add_argument("--canny-high", type=int, default=150)
	parser.add_argument("--min-area-ratio", type=float, default=0.2)
	parser.add_argument("--max-area-ratio", type=float, default=0.98)
	parser.add_argument("--illum-method", choices=["subtract", "divide"], default="subtract")
	parser.add_argument("--illum-blur-frac", type=float, default=0.02)
	parser.add_argument("--block-size", type=int, default=35)
	parser.add_argument("--C", type=int, default=10)
	parser.add_argument("--thresh-method", choices=["gaussian", "mean"], default="gaussian")
	parser.add_argument("--mask-blur-ksize", type=int, default=51)
	parser.add_argument("--blackhat-ksize", type=int, default=9)
	parser.add_argument("--blackhat-vertical-ratio", type=float, default=2.0)
	parser.add_argument("--ink-dilate-iters", type=int, default=1)
	parser.add_argument("--mask-thresh-offset", type=int, default=8)
	parser.add_argument("--morph-ksize", type=int, default=3)
	parser.add_argument("--morph-iters", type=int, default=1)
	parser.add_argument("--max-rotate", type=float, default=10.0)
	parser.add_argument("--fallback-use-whole", action="store_true", default=True,
						 help="Fallback to whole image if quad is missing or too small")
	parser.add_argument("--min-quad-area-ratio", type=float, default=0.15,
						 help="Minimum quad area ratio before fallback")
	args = parser.parse_args()

	res = process_document(
		args.input,
		out_dir=args.out,
		page=args.page,
		scale_long=args.scale_long,
		do_ocr=args.ocr,
		bilateral_d=args.bilateral_d,
		bilateral_sigmaColor=args.bilateral_sigma_color,
		bilateral_sigmaSpace=args.bilateral_sigma_space,
		gaussian_ksize=args.gaussian_ksize,
		canny_low=args.canny_low,
		canny_high=args.canny_high,
		min_area_ratio=args.min_area_ratio,
		max_area_ratio=args.max_area_ratio,
		illum_method=args.illum_method,
		illum_blur_frac=args.illum_blur_frac,
		block_size=args.block_size,
		C=args.C,
		thresh_method=args.thresh_method,
		mask_blur_ksize=args.mask_blur_ksize,
		blackhat_ksize=args.blackhat_ksize,
		blackhat_vertical_ratio=args.blackhat_vertical_ratio,
		ink_dilate_iters=args.ink_dilate_iters,
		mask_thresh_offset=args.mask_thresh_offset,
		morph_ksize=args.morph_ksize,
		morph_iters=args.morph_iters,
		max_rotate=args.max_rotate,
		fallback_use_whole=args.fallback_use_whole,
		min_quad_area_ratio=args.min_quad_area_ratio,
	)
	print(f"Done. Outputs in {args.out}")


if __name__ == "__main__":
	main()

