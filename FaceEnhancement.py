import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# CONFIGURATION
# ===============================================================
UPSCALE_FACTOR = 1.0          # 2x is usually sufficient; 4x can introduce false blur
BILATERAL_SIGMA_COLOR = 30  # Intensity difference to be considered an edge (Lower = More edge preservation)
BILATERAL_SIGMA_SPACE = 10  # Spatial distance to smooth (Lower = Less blurring)
SHARPEN_AMOUNT = 2.0        # Amount of detail to add back
SKIN_MASK_THRESHOLD = (0, 133, 77, 255, 173, 127) # YCrCb lower/upper
COLOR_SATURATION = 1.20  # Increased to 1.20 for more vibrant/red color

# ===============================================================
# 1. LOAD & PREPROCESS
# ===============================================================
def load_and_prep(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image not found")
    
    
    # 1. Gentle Upscale (Bicubic) - Creates a better canvas for processing
    img = cv2.resize(img, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)

    return img

def apply_smart_denoise(img, override_h=None):
    """
    Analyzes noise type and applies the appropriate filter.
    Returns: (denoised_image, noise_type_string)
    """
    # 2. Smart Noise Detection & Removal
    # ----------------------------------
    noise_type = classify_noise_type(img)
    
    if noise_type == "impulse":
        print("Detected IMPULSE Noise (Salt & Pepper). Using Median Filter.")
        # Median filter is the best for removing salt-and-pepper noise
        # Reverted to 3 for better edge preservation (5 is too blunt)
        img = cv2.medianBlur(img, 3) 
    else:
        print("Detected GAUSSIAN/NORMAL Noise. Using Non-Local Means.")
        
        # Default strength
        h_val = 10 
        
        # If override provided, use it
        if override_h is not None:
             h_val = override_h
        else:
             # Default Logic (Legacy fallback)
             pass
             
        print(f"Applying NLM Denoise with h={h_val}")
        img = cv2.fastNlMeansDenoisingColored(img, None, h=h_val, hColor=h_val, templateWindowSize=7, searchWindowSize=21)
    
    return img, noise_type

def classify_noise_type(image):
    """
    Determines if noise is 'Impulse' (Salt & Pepper) or 'Gaussian' (Normal Distribution).
    Logic: Uses Kurtosis of the noise residuals.
    - Gaussian noise follows a Normal distribution (Kurtosis ~ 0 in Fisher def, or 3 in Pearson).
    - Impulse noise has a sharp peak at 0 (unaffected pixels) and heavy tails (outliers).
      This results in a very high likelihood of High Kurtosis (Leptokurtic).
    """
    # 1. Convert to Gray
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # 2. Estimate Noise Surface (Residuals)
    # We assume median blur gives a "clean" reference.
    clean_est = cv2.medianBlur(gray, 3)
    
    # Cast to float to avoid overflow issues during subtraction
    noise_residuals = gray.astype(np.float64) - clean_est.astype(np.float64)
    noise_flat = noise_residuals.flatten()
    
    # 3. Calculate Kurtosis
    # Formula: Mean((x - mean)^4) / Std^4
    mean = np.mean(noise_flat)
    std = np.std(noise_flat)
    
    if std == 0:
        return "gaussian" # Flat image, no noise
        
    fourth_moment = np.mean((noise_flat - mean) ** 4)
    kurtosis = fourth_moment / (std ** 4)
    
    print(f"Noise Analysis - Kurtosis: {kurtosis:.2f}")
    
    # 4. Threshold
    # Normal Dist (Gaussian) has pixel-kurtosis ~ 3.0 (Pearson).
    # Impulse noise (S&P) makes the distribution extremely peaked (Leptokurtic) -> High Kurtosis.
    if kurtosis > 5.0: # Threshold of 5 gives a safe margin
        return "impulse"
    else:
        return "gaussian"

# ===============================================================
# 2. ADVANCED SKIN MASKING (YCbCr + Ellipse Prior)
# ===============================================================
def get_refined_skin_mask(image):
    """
    Generates a soft, feathered mask for skin areas.
    """
    # Convert to YCrCb for skin tone detection
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Classical skin segmentation thresholds
    lower = np.array([SKIN_MASK_THRESHOLD[0], SKIN_MASK_THRESHOLD[1], SKIN_MASK_THRESHOLD[2]], dtype=np.uint8)
    upper = np.array([SKIN_MASK_THRESHOLD[3], SKIN_MASK_THRESHOLD[4], SKIN_MASK_THRESHOLD[5]], dtype=np.uint8)
    
    mask = cv2.inRange(ycbcr, lower, upper)
    
    # Clean up valid skin regions (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Soften the mask for natural blending (Feathering)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    return mask

# ===============================================================
# 3. GLAMOUR BLUR (Bilateral Filter)
# ===============================================================
def apply_glamour_skin(image, mask):
    """
    Applies Bilateral Filter only to skin regions.
    Bilateral filter smooths flat regions (skin) but keeps edges (chins, noses) sharp.
    """
    # Bilateral Filter: The 'Beauty' filter standard
    # d=-1 uses sigmaSpace to determine diameter
    skin_smooth = cv2.bilateralFilter(image, d=-1, 
                                      sigmaColor=BILATERAL_SIGMA_COLOR, 
                                      sigmaSpace=BILATERAL_SIGMA_SPACE)
    
    # Blend: (Smooth * Mask) + (Original * (1-Mask))
    mask_norm = mask.astype(float) / 255.0
    mask_norm = cv2.merge([mask_norm, mask_norm, mask_norm]) # Make 3 channel
    
    # Linear blend
    result = (skin_smooth * mask_norm) + (image * (1 - mask_norm))
    return result.astype(np.uint8)

# ===============================================================
# 4. DETAIL ENHANCEMENT (Unsharp Mask on Luminance)
# ===============================================================
def enhance_details(image, amount=1.0, threshold=0):
    """
    Sharpen the image using Unsharp Masking, applied primarily to the Luminance channel
    to avoid color artifacts.
    """
    # Split into LAB to sharpen only the 'L' (Lightness) channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Gaussian Blur determines the "unsharp" part
    blurred_l = cv2.GaussianBlur(l, (0, 0), 3)
    
    # Sharpen: Original + (Original - Blurred) * Amount
    sharpened_l = cv2.addWeighted(l, 1.0 + amount, blurred_l, -amount, 0)
    
    # Merge back
    enhanced_lab = cv2.merge([sharpened_l, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr

# ===============================================================
# 5. FEATURE POP (Eye Enhancement)
# ===============================================================
def pixel_pop_eyes(image):
    """
    Detects eyes and adds local contrast and brightness to make them 'pop'.
    """
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if eye_cascade.empty():
        return image
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    output = image.copy()
    
    for (x, y, w, h) in eyes:
        roi = output[y:y+h, x:x+w]
        
        # 0. Pre-smooth the eye area to reduce grain BEFORE enhancing contrast
        roi = cv2.medianBlur(roi, 3) # Removes salt-and-pepper noise inside the eye

        # increase local contrast (CLAHE on ROI)
        # Convert ROI to LAB
        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(roi_lab)
        
        # Apply CLAHE to L channel of ROI (Drastically reduced power)
        clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(4,4))
        l = clahe.apply(l)
        
        # Merge and put back
        roi_enhanced = cv2.merge([l, a, b])
        roi_enhanced = cv2.cvtColor(roi_enhanced, cv2.COLOR_LAB2BGR)
        
        # Slight sharpen on eyes specifically
        roi_enhanced = enhance_details(roi_enhanced, amount=0.5)
        
        # --- BLENDING TO REMOVE SQUARE ARTIFACT ---
        # Create an elliptical mask for the eye
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (w // 2, h // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Soften the mask edges (Feathering)
        mask_soft = cv2.GaussianBlur(mask, (31, 31), 0)
        
        # Normalize mask to 0-1 and reduce opacity
        alpha = mask_soft.astype(float) / 255.0
        alpha = alpha * 0.1  # Reduce intensity of enhancement to 60%
        alpha = cv2.merge([alpha, alpha, alpha])
        
        # Alpha Blend: Enhanced * alpha + Original * (1 - alpha)
        roi_original = roi.astype(float)
        roi_enhanced_float = roi_enhanced.astype(float)
        
        blended = (roi_enhanced_float * alpha + roi_original * (1.0 - alpha))
        output[y:y+h, x:x+w] = blended.astype(np.uint8)
        
    return output

# ===============================================================
# 6. GLOBAL TONE & COLOR
# ===============================================================
def adjust_saturation(image, saturation=1.0):
    """
    Adjusts the color saturation.
    saturation < 1.0: Desaturate (Cooler/Blander)
    saturation > 1.0: Saturate (Warmer/Vibrant)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Multiply saturation channel
    s = s.astype(float) * saturation
    s = np.clip(s, 0, 255).astype(np.uint8)
    
    hsv_new = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

def apply_warmth(image, amount=10):
    """
    Increases the Red channel and decreases Blue channel to warm up the image.
    amount: strength of shift (0-255)
    """
    image = image.astype(float)
    b, g, r = cv2.split(image)
    
    # Warmer = More Red, Less Blue
    r += amount
    b -= amount *.05 # Slight blue reduction
    
    merged = cv2.merge([b, g, r])
    return np.clip(merged, 0, 255).astype(np.uint8)

def apply_contrast_stretching(image):
    """
    Linearly stretches the contrast to cover the full 0-255 range.
    (Min-Max Normalization on Luminance)
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Normalize L channel (Stretches darkest to 0 and brightest to 255)
    l_stretched = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
    
    merged = cv2.merge([l_stretched, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_histogram_equalization(image):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
    This is the modern/better version of Histogram Equalization for faces.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE adds local contrast (Brighter brights, darker darks locally)
    # Reduced clipLimit to 0.5 (from 1.5) to stop it from amplifying hidden noise
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
    l_eq = clahe.apply(l)
    
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def apply_masked_sharpening(image, mask, amount=1.0):
    """
    Applies DUAL sharpening:
    - Strong sharpening on the masked area (Face/Skin).
    - Moderate sharpening (50%) on the background to recover edges/texture.
    """
    # 1. Generate Sharpened versions
    sharpened_face = enhance_details(image, amount=amount)        # Strong for face
    sharpened_bg   = enhance_details(image, amount=amount * 0.5)  # Moderate for hair/bg
    
    # 2. Blend: (SharpenedFace * Mask) + (SharpenedBG * (1-Mask))
    mask_norm = mask.astype(float) / 255.0
    mask_norm = cv2.merge([mask_norm, mask_norm, mask_norm])
    
    result = (sharpened_face * mask_norm) + (sharpened_bg * (1 - mask_norm))
    return result.astype(np.uint8)

# ===============================================================
# MAIN PIPELINE
# ===============================================================
if __name__ == "__main__":
    try:
        # 1. Load & Prop
        original_noisy = load_and_prep("public/facewithgaussiannoise.jpg")
        print("Image loaded and upscaled (Contains Noise).")
        
        # 1. Denoising (Dual Strategy)
        # ----------------------------
        # First, classify noise to decide strategy
        noise_type_detected = classify_noise_type(original_noisy)
        
        if noise_type_detected == "gaussian":
            print(f"Detected {noise_type_detected.upper()} Noise -> Using Gaussian Blur Strategy.")
            # validated strategy from 'pipeline_gaussian.py'
            
            # A. Light Denoise (Face) - Kernel 5x5
            print(">>> Generating Light Denoise (Gaussian 5x5)...")
            denoised_light = cv2.GaussianBlur(original_noisy, (5, 5), 0)
            
            # B. Strong Denoise (Background) - Kernel 9x9
            print(">>> Generating Strong Denoise (Gaussian 9x9)...")
            denoised_strong = cv2.GaussianBlur(original_noisy, (9, 9), 0)
            
        elif noise_type_detected == "impulse":
            print(f"Detected {noise_type_detected.upper()} Noise -> Using Median Filter Strategy.")
            
            # A. Light Denoise (Face) - Kernel 3 (Standard)
            print(">>> Generating Light Denoise (Median 3)...")
            denoised_light = cv2.medianBlur(original_noisy, 3)
            
            # B. Strong Denoise (Background) - Kernel 5 (Aggressive)
            print(">>> Generating Strong Denoise (Median 5)...")
            denoised_strong = cv2.medianBlur(original_noisy, 5)

        else:
            print(f"Detected {noise_type_detected.upper()} Noise -> Using Legacy Smart Denoise.")
            # Fallback for unknown noise types
            # A. Light Denoise (For Face/Edges) - h=10
            print(">>> Generating Light Denoise (Target: Face)...")
            denoised_light, _ = apply_smart_denoise(original_noisy, override_h=10)
            
            # B. Strong Denoise (For Background/Hair) - h=30
            print(">>> Generating Strong Denoise (Target: Background)...")
            denoised_strong, _ = apply_smart_denoise(original_noisy, override_h=30)

        # 2. Skin Masking (Use Light Denoise for best edge detection)
        skin_mask = get_refined_skin_mask(denoised_light)
        
        # 3. Blend Denoised Versions
        # (Light * Mask) + (Strong * (1-Mask))
        mask_norm = skin_mask.astype(float) / 255.0
        mask_norm = cv2.merge([mask_norm, mask_norm, mask_norm])
        
        denoised_combined = (denoised_light * mask_norm) + (denoised_strong * (1 - mask_norm))
        denoised_combined = denoised_combined.astype(np.uint8)
        print("Merged Denoised Images (Face=Light, BG=Strong).")
        
        # 4. Glamour Skin (Bilateral)
        # Apply smoothing to the Combined result
        skin_enhanced = apply_glamour_skin(denoised_combined, skin_mask)
        print("Skin smoothed (Bilateral Filter).")
        
        # 4. Detail Enhancement (Sharpening) - SKIPPED AS REQUESTED
        # sharpened = enhance_details(skin_enhanced, amount=SHARPEN_AMOUNT)
        # print("Details sharpened.")
        
        # 5. Feature Pop (Eyes)
        features_popped = pixel_pop_eyes(skin_enhanced)
        print("Eyes enhanced.")
        
        # 6. Global Tone & Contrast
        # Reduce warmth first
        color_corrected = adjust_saturation(features_popped, saturation=COLOR_SATURATION)
        
        # Apply Warmth (Red Boost)
        warmed = apply_warmth(color_corrected, amount=15) # Strong warmth boost
        print("Warmth/Redness boosted.")

        # Apply Contrast Stretching (Linear stretching)
        # Apply Contrast Stretching (Linear stretching) - SKIP IF GAUSSIAN
        if noise_type_detected == "gaussian":
             # For Gaussian pipeline, we skip contrast stretching to avoid noise amp
             print("Skipping Contrast Stretching (Gaussian Mode).")
             stretched = warmed
        else:
             stretched = apply_contrast_stretching(warmed)
        
        # Apply Histogram Equalization (CLAHE) for extra "Pop"
        clahe_result = apply_histogram_equalization(stretched)
        
        # 6b. Post-CLAHE Polish (Edge-Preserving Smooth)
        # SKIP for Gaussian (as per request "Remove ... polishing")
        if noise_type_detected == "gaussian":
             polished = clahe_result
        else:
             # CLAHE can re-introduce micro-noise. We smooth it out gently.
             polished = cv2.bilateralFilter(clahe_result, d=5, sigmaColor=20, sigmaSpace=20)
        print("Global tone (Contrast Stretching + CLAHE) applied & Polished.")

        # 7. Final Masked Sharpening (Face Only)
        # 7. Final Masked Sharpening - SKIP IF GAUSSIAN
        if noise_type_detected == "gaussian":
            print("Skipping Final Sharpening (Gaussian Mode).")
            final_output = polished
        else:
            final_sharpen_amount = SHARPEN_AMOUNT
            # If impulse/other, we might want extra sharpening
            if noise_type_detected != "gaussian": 
                 # Legacy logic
                 pass
            
            print(f"Applying Final Masked Sharpening (Amount={final_sharpen_amount})...")
            final_output = apply_masked_sharpening(polished, skin_mask, amount=final_sharpen_amount)
        
        # Save
        cv2.imwrite("public/enhanced_face.jpg", final_output)
        print("Saved 'public/enhanced_face.jpg'")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original (Upscaled + Noisy)")
        plt.imshow(cv2.cvtColor(original_noisy, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Final Enhanced (Beauty Pipeline)")
        plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
