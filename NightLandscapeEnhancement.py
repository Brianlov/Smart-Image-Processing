# Digital Image Processing Assignment: Night Landscape Enhancement
# Student: Kok Yu Yuan (B022210136)
# Task: Implement filtering and enhancement for a low-light/night scene
#
# Pipeline:
# 1. Noise Reduction – Median Filter (3x3)
# 2. Contrast Enhancement – CLAHE
#
# Assumption: 'nightview.jpg' is in the same directory

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Image Acquisition & Pre-processing
# -------------------------------

# 1.1 Image file name
img_name = 'nightview.png'

# 1.2 Read image (grayscale)
I_original = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

if I_original is None:
    raise FileNotFoundError("Image not found. Check the file name or directory.")

# 1.3 Convert to double precision (0–1 range)
I_double = I_original.astype(np.float64) / 255.0

# Plotting logic moved to the end

# -------------------------------
# 2. Noise Reduction using Median Filter
# -------------------------------

# 2.1 Median filter (3x3)
I_filtered = cv2.medianBlur((I_double * 255).astype(np.uint8), 3)
I_filtered = I_filtered.astype(np.float64) / 255.0

# Plotting logic moved to the end

print("Noise Reduction Complete: Median filtering applied.")

# -------------------------------
# 3. Contrast Enhancement using CLAHE
# -------------------------------

# 3.1 CLAHE parameters (equivalent to MATLAB adapthisteq)
clahe = cv2.createCLAHE(
    clipLimit=2.0,        # Similar effect to MATLAB ClipLimit = 0.01
    tileGridSize=(8, 8)
)

I_enhanced = clahe.apply((I_filtered * 255).astype(np.uint8))

# -------------------------------
# 4. Final Comparison & Visualization
# -------------------------------

plt.figure(figsize=(12, 10))

# 1. Original
plt.subplot(2, 2, 1)
plt.imshow(I_double, cmap='gray')
plt.title('1. Original Night Scene')
plt.axis('off')

# 2. Histogram
plt.subplot(2, 2, 2)
plt.hist(I_double.ravel(), bins=256, range=(0, 1))
plt.title('Original Histogram')

# 3. Median Filtered
plt.subplot(2, 2, 3)
plt.imshow(I_filtered, cmap='gray')
plt.title('2. Median Filtered (Noise Reduced)')
plt.axis('off')

# 4. Enhanced
plt.subplot(2, 2, 4)
plt.imshow(I_enhanced, cmap='gray')
plt.title('3. Enhanced (CLAHE)')
plt.axis('off')

plt.tight_layout(pad=3.0)
plt.show()

print("Contrast Enhancement Complete: Final image ready.")

# -------------------------------
# Optional: Save result
# -------------------------------
# cv2.imwrite('enhanced_night_landscape.jpg', I_enhanced)