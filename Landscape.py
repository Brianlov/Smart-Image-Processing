# ==========================================
# IMPORTS AND CONFIGURATION
# ==========================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
import glob

# ==========================================
# INPUT CONFIGURATION
# ==========================================

# Input can be a single file or a folder
INPUT_PATH = "C:\\Users\\User\\Downloads\\All Study Shits\\Each Sem Material\\Y4S1\\BERR4723 Digital Image Processing\\Assignment\\LandscapeImages"
OUTPUT_FOLDER = "enhanced_results"

# ==========================================
# GENERAL ENHANCEMENT PRESET
# ==========================================

ENHANCEMENT_PRESET = {
    # Image Restoration - Denoising
    "denoising": {
        "enabled": True,
        "method": "bilateral",        # Changed to bilateral (better edge preservation)
        "kernel_size": 5              # For median filter (3, 5, 7)
    },
    
    # Contrast Enhancement - CLAHE
    "clahe": {
        "enabled": True,
        "clip_limit": 2.2,            # Balanced enhancement
        "tile_grid_size": (8, 8),
        "sky_protection_power": 2.0,  # Less aggressive protection (avoid darkening)
        "blend_strength": 0.55        # Moderate blending
    },
    
    # Sharpening
    "sharpening": {
        "enabled": True,
        "amount": 0.8,                # Gentler sharpening (preserve naturalness)
        "radius": 1.0
    },
    
    # Degradation settings (for demonstration)
    "degradation": {
        "contrast_reduction": 0.7,    # Less aggressive (was 0.6)
        "underexposure": 0.85,        # Less aggressive (was 0.8)
        "noise_amount": 10,           # Reduced from 15 (less noise = easier recovery)
        "saturation_reduction": 0.85  # Less aggressive (was 0.8)
    }
}

print("="*60)
print("LANDSCAPE ENHANCEMENT PIPELINE")
print("="*60)
print(f"Input: {INPUT_PATH}")
print(f"Output Folder: {OUTPUT_FOLDER}")
print(f"\nEnhancement Configuration:")
print(f"  • Denoising: {ENHANCEMENT_PRESET['denoising']['method'].upper()}")
print(f"  • CLAHE Clip Limit: {ENHANCEMENT_PRESET['clahe']['clip_limit']}")
print(f"  • Sky Protection Power: {ENHANCEMENT_PRESET['clahe']['sky_protection_power']}")
print(f"  • Blend Strength: {ENHANCEMENT_PRESET['clahe']['blend_strength']}")
print(f"  • Sharpening Amount: {ENHANCEMENT_PRESET['sharpening']['amount']}")
print("="*60)

# ==========================================
# ENHANCEMENT PIPELINE FUNCTIONS
# ==========================================

def degrade_image(img, config):
    """Simulate poor image quality for demonstration"""
    img_float = img.astype(np.float32) / 255.0
    
    # Reduce contrast
    contrast = config.get("contrast_reduction", 0.6)
    img_float = img_float * contrast + 0.5 * (1 - contrast)
    
    # Darken (underexposure)
    underexp = config.get("underexposure", 0.8)
    img_float = np.power(img_float, 1.0 / underexp)
    
    # Desaturate
    sat = config.get("saturation_reduction", 0.8)
    hsv = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= sat
    img_float = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    
    # Add noise
    noise_level = config.get("noise_amount", 15)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level / 255.0, img_float.shape)
        img_float += noise
    
    return np.clip(img_float * 255, 0, 255).astype(np.uint8)


def denoise_image(img, method="median", kernel_size=5, is_noisy=False):
    """
    Apply denoising based on method.
    If is_noisy=True, apply stronger denoising for degraded images.
    """
    if method == "median":
        # Use larger kernel for noisy images
        k = kernel_size + 2 if is_noisy else kernel_size
        return cv2.medianBlur(img, k)
    elif method == "bilateral":
        # Stronger bilateral for noisy images
        d = 11 if is_noisy else 9
        sigma = 100 if is_noisy else 75
        return cv2.bilateralFilter(img, d, sigma, sigma)
    elif method == "nlmeans":
        # Stronger NLMeans for noisy images
        h = 15 if is_noisy else 10
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    return img


def enhance_contrast_clahe(img, clip_limit=2.5, tile_grid=(8, 8), sky_power=3, blend=0.6):
    """Apply CLAHE with sky protection"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_orig, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_clahe = clahe.apply(l_orig)
    
    # Sky protection mask
    l_norm = l_orig.astype(np.float32) / 255.0
    protection_mask = np.power(l_norm, sky_power)
    enhance_weight = (1.0 - protection_mask) * blend
    
    l_final = (l_clahe.astype(np.float32) * enhance_weight + 
               l_orig.astype(np.float32) * (1.0 - enhance_weight)).astype(np.uint8)
    
    lab_enhanced = cv2.merge((l_final, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def sharpen_image(img, amount=1.5, radius=1.0):
    """Apply unsharp masking"""
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def calculate_metrics(img_before, img_after):
    """Calculate PSNR and SSIM"""
    psnr = cv2.PSNR(img_before, img_after)
    gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
    ssim_val, _ = ssim(gray_before, gray_after, full=True)
    return psnr, ssim_val


def enhance_image(img, preset, is_noisy=False):
    """
    Complete enhancement pipeline.
    is_noisy: If True, applies stronger denoising (for degraded images)
    """
    current = img.copy()
    
    # Step 1: Denoising (stronger for noisy/degraded images)
    if preset["denoising"]["enabled"]:
        current = denoise_image(
            current, 
            method=preset["denoising"]["method"],
            kernel_size=preset["denoising"]["kernel_size"],
            is_noisy=is_noisy
        )
    
    # Step 2: Contrast Enhancement
    if preset["clahe"]["enabled"]:
        current = enhance_contrast_clahe(
            current,
            clip_limit=preset["clahe"]["clip_limit"],
            tile_grid=preset["clahe"]["tile_grid_size"],
            sky_power=preset["clahe"]["sky_protection_power"],
            blend=preset["clahe"]["blend_strength"]
        )
    
    # Step 3: Sharpening (gentler for noisy images to avoid enhancing noise)
    if preset["sharpening"]["enabled"]:
        amount = preset["sharpening"]["amount"] * (0.7 if is_noisy else 1.0)
        current = sharpen_image(
            current,
            amount=amount,
            radius=preset["sharpening"]["radius"]
        )
    
    return current


def get_image_stats(img):
    """Get basic image statistics"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": np.mean(gray),
        "contrast": np.std(gray)
    }

print("✓ Enhancement pipeline functions loaded")

# ==========================================
# IMAGE PROCESSING - One image per output block
# ==========================================

# Collect input files
if os.path.isfile(INPUT_PATH):
    image_files = [INPUT_PATH]
elif os.path.isdir(INPUT_PATH):
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(INPUT_PATH, ext)))
else:
    raise ValueError(f"Invalid path: {INPUT_PATH}")

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"\n{'='*60}")
print(f"PROCESSING {len(image_files)} IMAGE(S)")
print(f"{'='*60}")

# Process each image
for idx, img_path in enumerate(image_files, 1):
    filename = os.path.basename(img_path)
    
    print(f"\n{'='*60}")
    print(f"[{idx}/{len(image_files)}] {filename}")
    print(f"{'='*60}")
    
    # Load original image
    img_original = cv2.imread(img_path)
    if img_original is None:
        print(f"✗ Error reading file. Skipping.")
        continue
    
    # ========================================
    # PATH 1: Enhance ORIGINAL image
    # ========================================
    print("\n--- PATH 1: Original Image Enhancement ---")
    img_enhanced_orig = enhance_image(img_original, ENHANCEMENT_PRESET, is_noisy=False)
    psnr_orig, ssim_orig = calculate_metrics(img_original, img_enhanced_orig)
    stats_orig_before = get_image_stats(img_original)
    stats_orig_after = get_image_stats(img_enhanced_orig)
    
    print(f"  Before: Brightness={stats_orig_before['brightness']:.1f}, Contrast={stats_orig_before['contrast']:.1f}")
    print(f"  After:  Brightness={stats_orig_after['brightness']:.1f}, Contrast={stats_orig_after['contrast']:.1f}")
    print(f"  PSNR: {psnr_orig:.2f} dB | SSIM: {ssim_orig:.4f}")
    
    # ========================================
    # PATH 2: Degrade then Enhance
    # ========================================
    print("\n--- PATH 2: Degraded Image Enhancement ---")
    img_degraded = degrade_image(img_original, ENHANCEMENT_PRESET["degradation"])
    img_enhanced_degraded = enhance_image(img_degraded, ENHANCEMENT_PRESET, is_noisy=True)
    psnr_deg, ssim_deg = calculate_metrics(img_degraded, img_enhanced_degraded)
    stats_deg_before = get_image_stats(img_degraded)
    stats_deg_after = get_image_stats(img_enhanced_degraded)
    
    print(f"  Before: Brightness={stats_deg_before['brightness']:.1f}, Contrast={stats_deg_before['contrast']:.1f}")
    print(f"  After:  Brightness={stats_deg_after['brightness']:.1f}, Contrast={stats_deg_after['contrast']:.1f}")
    print(f"  PSNR: {psnr_deg:.2f} dB | SSIM: {ssim_deg:.4f}")
    
    # ========================================
    # VISUALIZATION
    # ========================================
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{filename}", fontsize=14, fontweight='bold')
    
    # Row 1: Original Path
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.title("Original", fontsize=11)
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(cv2.cvtColor(img_enhanced_orig, cv2.COLOR_BGR2RGB))
    plt.title(f"Enhanced (Original)\nPSNR:{psnr_orig:.1f}dB SSIM:{ssim_orig:.3f}", fontsize=10, color='green')
    plt.axis('off')
    
    # Row 1: Histogram for Original Path
    plt.subplot(2, 4, 3)
    gray_orig = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    gray_enh_orig = cv2.cvtColor(img_enhanced_orig, cv2.COLOR_BGR2GRAY)
    plt.hist(gray_orig.flatten(), bins=256, range=[0, 256], color='gray', alpha=0.6, label='Before')
    plt.hist(gray_enh_orig.flatten(), bins=256, range=[0, 256], color='green', alpha=0.6, label='After')
    plt.title("Histogram (Original Path)", fontsize=10)
    plt.legend(fontsize=8)
    plt.xlabel("Intensity")
    
    # Row 1: Metrics Text
    plt.subplot(2, 4, 4)
    plt.axis('off')
    metrics_text1 = f"""ORIGINAL PATH METRICS
{'─'*25}
Brightness:
  Before: {stats_orig_before['brightness']:.1f}
  After:  {stats_orig_after['brightness']:.1f}
  Change: {stats_orig_after['brightness']-stats_orig_before['brightness']:+.1f}

Contrast (Std):
  Before: {stats_orig_before['contrast']:.1f}
  After:  {stats_orig_after['contrast']:.1f}
  Change: {stats_orig_after['contrast']-stats_orig_before['contrast']:+.1f}

Quality:
  PSNR: {psnr_orig:.2f} dB
  SSIM: {ssim_orig:.4f}"""
    plt.text(0.1, 0.5, metrics_text1, fontsize=9, family='monospace', va='center')
    
    # Row 2: Degraded Path
    plt.subplot(2, 4, 5)
    plt.imshow(cv2.cvtColor(img_degraded, cv2.COLOR_BGR2RGB))
    plt.title("Degraded (Noisy)", fontsize=11)
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.imshow(cv2.cvtColor(img_enhanced_degraded, cv2.COLOR_BGR2RGB))
    plt.title(f"Enhanced (Degraded)\nPSNR:{psnr_deg:.1f}dB SSIM:{ssim_deg:.3f}", fontsize=10, color='blue')
    plt.axis('off')
    
    # Row 2: Histogram for Degraded Path
    plt.subplot(2, 4, 7)
    gray_deg = cv2.cvtColor(img_degraded, cv2.COLOR_BGR2GRAY)
    gray_enh_deg = cv2.cvtColor(img_enhanced_degraded, cv2.COLOR_BGR2GRAY)
    plt.hist(gray_deg.flatten(), bins=256, range=[0, 256], color='red', alpha=0.6, label='Degraded')
    plt.hist(gray_enh_deg.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.6, label='Enhanced')
    plt.title("Histogram (Degraded Path)", fontsize=10)
    plt.legend(fontsize=8)
    plt.xlabel("Intensity")
    
    # Row 2: Metrics Text
    plt.subplot(2, 4, 8)
    plt.axis('off')
    metrics_text2 = f"""DEGRADED PATH METRICS
{'─'*25}
Brightness:
  Before: {stats_deg_before['brightness']:.1f}
  After:  {stats_deg_after['brightness']:.1f}
  Change: {stats_deg_after['brightness']-stats_deg_before['brightness']:+.1f}

Contrast (Std):
  Before: {stats_deg_before['contrast']:.1f}
  After:  {stats_deg_after['contrast']:.1f}
  Change: {stats_deg_after['contrast']-stats_deg_before['contrast']:+.1f}

Quality:
  PSNR: {psnr_deg:.2f} dB
  SSIM: {ssim_deg:.4f}"""
    plt.text(0.1, 0.5, metrics_text2, fontsize=9, family='monospace', va='center')
    
    plt.tight_layout()
    plt.show()
    
    # ========================================
    # SAVE OUTPUTS
    # ========================================
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"enhanced_original_{filename}"), img_enhanced_orig)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"degraded_{filename}"), img_degraded)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"enhanced_degraded_{filename}"), img_enhanced_degraded)
    print(f"\n✓ Saved outputs to {OUTPUT_FOLDER}/")

print(f"\n{'='*60}")
print(f"ALL PROCESSING COMPLETE")
print(f"{'='*60}")