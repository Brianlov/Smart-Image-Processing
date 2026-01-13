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
    apply_histogram_equalization,
    apply_masked_sharpening,
    apply_smart_denoise,
    apply_warmth,
    SHARPEN_AMOUNT,
    COLOR_SATURATION
)

# Configuration for GAUSSIAN Pipeline
INPUT_IMAGE = "public/gaussiannoiseNew.jpg"
OUTPUT_DIR = "public/pipeline_stages_gaussian"

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
        print(f"Processing {INPUT_IMAGE}...")
        
        # Stage 0: Load & Upscale (Raw NOISY input)
        base_image_noisy = load_and_prep(INPUT_IMAGE)
        save_stage(base_image_noisy, "0_Raw_Upscaled", 0)
        
        # Stage 1: Denoise (Dual Strategy - Using GAUSSIAN FILTER)
        print(">>> Generating Light Denoise (Gaussian Filter)...")
        # Light Blur for Face (Increased to 5x5 as requested)
        denoised_light = cv2.GaussianBlur(base_image_noisy, (5, 5), 0)
        noise_type = "gaussian" # Hardcoded since we know the input
        
        print(">>> Generating Strong Denoise (Gaussian Filter)...")
        # Strong Blur for Background (9x9)
        denoised_strong = cv2.GaussianBlur(base_image_noisy, (9, 9), 0)
        
        # Need mask to blend
        temp_mask = get_refined_skin_mask(denoised_light)
        mask_norm = temp_mask.astype(float) / 255.0
        mask_norm = cv2.merge([mask_norm, mask_norm, mask_norm])
        
        # Blend
        base_image = (denoised_light * mask_norm) + (denoised_strong * (1 - mask_norm))
        base_image = base_image.astype(np.uint8)
        
        save_stage(base_image, "1_Denoised_Dual_GaussianBlur", 1)
        
        # Stage 2: Skin Masking (Visualization only)
        skin_mask = get_refined_skin_mask(base_image)
        save_stage(skin_mask, "2_Mask_Used_For_Sharpening", 2)
        
        # Stage 3: Skin Smoothing
        skin_enhanced = apply_glamour_skin(base_image, skin_mask)
        save_stage(skin_enhanced, "3_Skin_Smoothed", 3)
        
        # Stage 5: Eye Enhancement (Skipped Stage 4 Sharpening)
        features_popped = pixel_pop_eyes(skin_enhanced)
        save_stage(features_popped, "5_Eyes_Popped", 5)
        
        # Stage 6: Color Correction
        color_corrected = adjust_saturation(features_popped, saturation=COLOR_SATURATION)
        save_stage(color_corrected, "6_Color_Corrected", 6)
        
        # Stage 6b: Warmth/Red Boost
        warmed = apply_warmth(color_corrected, amount=15)
        save_stage(warmed, "6b_Warmed", 6)
        
        # Stage 7: Contrast Stretching - SKIPPED FOR GAUSSIAN
        # stretched = apply_contrast_stretching(warmed)
        # save_stage(stretched, "7_Contrast_Stretched", 7)
        
        # Stage 8: Histogram Equalization (CLAHE) - RESTORED
        clahe_result = apply_histogram_equalization(warmed)
        save_stage(clahe_result, "8_CLAHE", 8)

        # Stage 8b: Post-CLAHE Polish - RESTORED
        polished = cv2.bilateralFilter(clahe_result, d=5, sigmaColor=20, sigmaSpace=20)
        save_stage(polished, "8b_Polished_CLAHE", 10) 

        # Stage 9: Final Masked Sharpening (Face Only)
        final_sharpen_amount = SHARPEN_AMOUNT
        final_output = apply_masked_sharpening(polished, skin_mask, amount=final_sharpen_amount)
        save_stage(final_output, "9_Final_Masked_Sharpen", 9)
        
        print("\nAll stages saved to:", OUTPUT_DIR)
        
        # Optional: Create a combined strip
        # Resize for display strip
        h = 300
        w = int(base_image.shape[1] * (h / base_image.shape[0]))
        
        # Updated list including Polish and Final
        images = [base_image_noisy, base_image, skin_enhanced, features_popped, warmed, clahe_result, polished, final_output]
        descriptions = ["Noisy", "Denoise", "Smooth", "Eyes", "Warm", "CLAHE", "Polish", "Final"]
        
        resized_images = [cv2.resize(img, (w, h)) for img in images]
        combined = np.hstack(resized_images)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "combined_strip.jpg"), combined)
        print("Saved combined comparison strip.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
