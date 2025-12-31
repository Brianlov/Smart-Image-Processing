# Digital Image Processing Assignment: Night Landscape Enhancement (RGB Version)
# Student: Kok Yu Yuan (B022210136)
# Task: Implement filtering and enhancement for a low-light/night scene (RGB Color)
#
# Pipeline:
# 1. Noise Reduction – Median Filter (3x3)
# 2. Contrast Enhancement – CLAHE on LAB color space
#
# Assumption: 'nightview_rgb.jpg' is in the same directory

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Image Acquisition & Pre-processing
# -------------------------------

# 1.1 Image file name
img_name = 'nightview.jpg'

# 1.2 Read image (RGB color)
I_original_bgr = cv2.imread(img_name, cv2.IMREAD_COLOR)

if I_original_bgr is None:
    raise FileNotFoundError("Image not found. Check the file name or directory.")

# 1.3 Convert BGR to RGB for display
I_original = cv2.cvtColor(I_original_bgr, cv2.COLOR_BGR2RGB)

# 1.4 Display original image and histograms
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(I_original)
plt.title('A. Original Night Scene (Low-Light)')
plt.axis('off')

plt.subplot(1, 3, 2)
colors = ('r', 'g', 'b')
for i, color in enumerate(colors):
    plt.hist(I_original[:, :, i].ravel(), bins=256, range=(0, 256), 
             color=color, alpha=0.5, label=color.upper())
plt.title('RGB Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(cv2.cvtColor(I_original, cv2.COLOR_RGB2GRAY).ravel(), 
         bins=256, range=(0, 256), color='gray')
plt.title('Grayscale Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

plt.tight_layout()

# -------------------------------
# 2. Noise Reduction using Median Filter
# -------------------------------

# 2.1 Median filter (3x3) applied to each channel
I_filtered_bgr = cv2.medianBlur(I_original_bgr, 3)
I_filtered = cv2.cvtColor(I_filtered_bgr, cv2.COLOR_BGR2RGB)

# 2.2 Display filtering result
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(I_original)
plt.title('Original (Reference Noise Level)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(I_filtered)
plt.title('B. After Median Filtering (Noise Reduced)')
plt.axis('off')

plt.tight_layout()

print("Noise Reduction Complete: Median filtering applied.")

# -------------------------------
# 3. Contrast Enhancement using CLAHE
# -------------------------------

# 3.1 Convert to LAB color space (apply CLAHE on L channel only)
I_lab = cv2.cvtColor(I_filtered_bgr, cv2.COLOR_BGR2LAB)
l_channel, a_channel, b_channel = cv2.split(I_lab)

# 3.2 Apply CLAHE to L channel
clahe = cv2.createCLAHE(
    clipLimit=2.0,        # Similar effect to MATLAB ClipLimit = 0.01
    tileGridSize=(8, 8)
)
l_channel_enhanced = clahe.apply(l_channel)

# 3.3 Merge channels back
I_lab_enhanced = cv2.merge([l_channel_enhanced, a_channel, b_channel])

# 3.4 Convert back to RGB
I_enhanced_bgr = cv2.cvtColor(I_lab_enhanced, cv2.COLOR_LAB2BGR)
I_enhanced = cv2.cvtColor(I_enhanced_bgr, cv2.COLOR_BGR2RGB)

# -------------------------------
# 4. Final Comparison & Visualization
# -------------------------------

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.imshow(I_original)
plt.title('1. Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(I_filtered)
plt.title('2. Median Filtered')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(I_enhanced)
plt.title('3. Enhanced (CLAHE)')
plt.axis('off')

plt.tight_layout()

print("Contrast Enhancement Complete: Final image ready.")

# -------------------------------
# 5. Histogram Comparison
# -------------------------------

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
for i, color in enumerate(colors):
    plt.hist(I_original[:, :, i].ravel(), bins=256, range=(0, 256), 
             color=color, alpha=0.5, label=color.upper())
plt.title('Original RGB Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 3, 2)
for i, color in enumerate(colors):
    plt.hist(I_filtered[:, :, i].ravel(), bins=256, range=(0, 256), 
             color=color, alpha=0.5, label=color.upper())
plt.title('Filtered RGB Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 3, 3)
for i, color in enumerate(colors):
    plt.hist(I_enhanced[:, :, i].ravel(), bins=256, range=(0, 256), 
             color=color, alpha=0.5, label=color.upper())
plt.title('Enhanced RGB Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()

# Display all plots together
plt.show()

# -------------------------------
# Optional: Save result
# -------------------------------
# cv2.imwrite('enhanced_night_landscape_rgb.jpg', I_enhanced_bgr)
