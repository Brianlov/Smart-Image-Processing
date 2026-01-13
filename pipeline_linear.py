import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from FaceEnhancement import (
    load_and_prep, 
    get_refined_skin_mask, 
    apply_glamour_skin, 
    enhance_details, 
    pixel_pop_eyes, 
    adjust_saturation, 
    apply_contrast_stretching, 
    apply_masked_sharpening,
    classify_noise_type,
    apply_smart_denoise,
    apply_warmth,
    SHARPEN_AMOUNT,
    COLOR_SATURATION
)

# Configuration
INPUT_IMAGE = "public/facewithnoise.jpg"
OUTPUT_DIR = "public/pipeline_stages_linear"

def save_stage(image, stage_name, index):
    filename = f"{index:02d}_{stage_name}.jpg"
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, image)
    print(f"Saved Stage {index}: {stage_name}")
    return image

def main():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    try:
        print(f"Processing {INPUT_IMAGE} (Linear Contrast Version)...")
        
        # Stage 1: Load (Upscale was disabled in load_and_prep)
        base_origin = load_and_prep(INPUT_IMAGE)
        # save_stage(base_origin, "0_Original_Loaded", 0)

        # Stage 1b: Noise Classification & Dual Strategy Denoising
        # --------------------------------------------------------
        noise_type = classify_noise_type(base_origin)
        print(f"Detected Noise Type: {noise_type.upper()}")

        if noise_type == "gaussian":
            # Gaussian Strategy
            print(" Using Gaussian Strategy (5x5 Face / 9x9 BG)")
            denoised_light = cv2.GaussianBlur(base_origin, (5, 5), 0)
            denoised_strong = cv2.GaussianBlur(base_origin, (9, 9), 0)
        
        elif noise_type == "impulse":
            # Impulse Strategy
            print(" Using Impulse Strategy (Median 3 Face / Median 5 BG)")
            denoised_light = cv2.medianBlur(base_origin, 3)
            denoised_strong = cv2.medianBlur(base_origin, 5)
        
        else:
            # Legacy/Fallback
            print(" Using Legacy Smart Denoise Strategy")
            denoised_light, _ = apply_smart_denoise(base_origin, override_h=10)
            denoised_strong, _ = apply_smart_denoise(base_origin, override_h=30)

        # Stage 2: Skin Masking (Use Light Denoise for best edge detection)
        skin_mask = get_refined_skin_mask(denoised_light)
        save_stage(skin_mask, "2_Mask_Used_For_Sharpening", 2)
        
        # Stage 1c: Blend Denoised Versions
        mask_norm = skin_mask.astype(float) / 255.0
        mask_norm = cv2.merge([mask_norm, mask_norm, mask_norm])
        
        denoised_combined = (denoised_light * mask_norm) + (denoised_strong * (1 - mask_norm))
        base_image = denoised_combined.astype(np.uint8)
        save_stage(base_image, "1_Denoised_Dual_Strategy", 1)
        
        # Stage 3: Skin Smoothing
        skin_enhanced = apply_glamour_skin(base_image, skin_mask)
        save_stage(skin_enhanced, "3_Skin_Smoothed", 3)
        
        # Stage 5: Eye Enhancement (Skipped Stage 4 Sharpening)
        features_popped = pixel_pop_eyes(skin_enhanced)
        save_stage(features_popped, "5_Eyes_Popped", 5)
        
        # Stage 6: Color Correction
        color_corrected = adjust_saturation(features_popped, saturation=COLOR_SATURATION)
        # save_stage(color_corrected, "6_Color_Corrected", 6)

        # Stage 6b: Warmth (Red Boost) - Added to match FaceEnhancement.py
        warmed = apply_warmth(color_corrected, amount=15)
        save_stage(warmed, "6_Color_Warmth_Corrected", 6)
        
        # Stage 7: Contrast Stretching (Re-added)
        stretched = apply_contrast_stretching(warmed)
        save_stage(stretched, "7_Contrast_Stretched", 7)
        
        # Stage 8: CLAHE (REMOVED) - We skip histogram equalization
        
        # Stage 8b: Post-Contrast Polish (Renamed from Post-CLAHE)
        # We still apply the polish to smooth potential noise from stretching
        polished = cv2.bilateralFilter(stretched, d=5, sigmaColor=20, sigmaSpace=20)
        save_stage(polished, "8b_Polished_Linear", 8)

        # Stage 9: Final Masked Sharpening (Face Only)
        final_output = apply_masked_sharpening(polished, skin_mask, amount=SHARPEN_AMOUNT)
        save_stage(final_output, "9_Final_Masked_Sharpen", 9)
        
        print("\nAll stages saved to:", OUTPUT_DIR)
        
        # Combined Strip
        h = 300
        w = int(base_image.shape[1] * (h / base_image.shape[0]))
        
        images = [base_image, skin_enhanced, features_popped, stretched, polished, final_output]
        descriptions = ["Denoised", "Smoothed", "Eyes", "Stretched", "Polished", "Final"]
        
        resized_images = [cv2.resize(img, (w, h)) for img in images]
        combined = np.hstack(resized_images)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "combined_strip_linear.jpg"), combined)
        print("Saved combined comparison strip.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
