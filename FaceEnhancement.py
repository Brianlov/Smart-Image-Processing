import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# CONFIGURATION
# ===============================================================
UPSCALE_FACTOR = 2          # 2x is usually sufficient; 4x can introduce false blur
BILATERAL_SIGMA_COLOR = 50  # Intensity difference to be considered an edge (Skin smoothing strength)
BILATERAL_SIGMA_SPACE = 20  # Spatial distance to smooth
SHARPEN_AMOUNT = 2.0        # Amount of detail to add back
SKIN_MASK_THRESHOLD = (0, 133, 77, 255, 173, 127) # YCrCb lower/upper
COLOR_SATURATION = 1.00  # Reduce saturation to cool down the image (< 1.0 = Cooler/Less Warm)

# ===============================================================
# 1. LOAD & PREPROCESS
# ===============================================================
def load_and_prep(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image not found")
    
    # 1. Gentle Upscale (Bicubic) - Creates a better canvas for processing
    img = cv2.resize(img, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
    
    # 2. Global Denoise (Gaussian Filter) - As requested
    img = cv2.GaussianBlur(img, (7,7), 0)
    
    return img

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
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l_eq = clahe.apply(l)
    
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# ===============================================================
# MAIN PIPELINE
# ===============================================================
if __name__ == "__main__":
    try:
        # 1. Load & Prop
        original = load_and_prep("public/girlsWithSomeNoise.png")
        print("Image loaded and upscaled.")
        
        # 2. Skin Masking
        skin_mask = get_refined_skin_mask(original)
        
        # 3. Glamour Skin (Bilateral)
        # Apply smoothing only where the skin mask decides
        skin_enhanced = apply_glamour_skin(original, skin_mask)
        print("Skin smoothed (Bilateral Filter).")
        
        # 4. Detail Enhancement (Sharpening)
        # Applied globally but works well after smoothing skin cleanly
        sharpened = enhance_details(skin_enhanced, amount=SHARPEN_AMOUNT)
        print("Details sharpened.")
        
        # 5. Feature Pop (Eyes)
        features_popped = pixel_pop_eyes(sharpened)
        print("Eyes enhanced.")
        
        # 6. Global Tone & Contrast
        # Reduce warmth first
        color_corrected = adjust_saturation(features_popped, saturation=COLOR_SATURATION)
        
        # Apply Contrast Stretching (Linear stretching)
        stretched = apply_contrast_stretching(color_corrected)
        
        # Apply Histogram Equalization (CLAHE) for extra "Pop"
        final_output = apply_histogram_equalization(stretched)
        
        print("Global tone (Contrast Stretching + CLAHE) applied.")
        
        # Save
        cv2.imwrite("public/enhanced_face.jpg", final_output)
        print("Saved 'public/enhanced_face.jpg'")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original (Upscaled)")
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Final Enhanced (Beauty Pipeline)")
        plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
