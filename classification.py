"""GUI app to classify images into: nightscape, landscape, document, face.

Features:
- Upload button: choose image from local PC
- Auto-detect image type using lightweight heuristics
- Show detected type with Confirm button
- Dropdown to override/choose another type

Run:
  python .\\classification.py
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk  # for image preview in Tkinter
except Exception:
    Image = None
    ImageTk = None

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from tkinter import ttk
except Exception as e:
    print("Tkinter is required for this GUI.")
    raise


LABELS = ["nightscape", "landscape", "document", "face"]


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def detect_faces(img_bgr: np.ndarray) -> int:
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            return 0
        gray = to_gray(img_bgr)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return len(faces)
    except Exception:
        return 0


def document_score(img_bgr: np.ndarray) -> Tuple[float, int, bool]:
    """Return (white_ratio, lines_count, large_rect_present)."""
    gray = to_gray(img_bgr)
    # Otsu to estimate white background ratio
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = float(np.mean(binary == 255))

    # Edge and Hough lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    lines_count = 0 if lines is None else len(lines)

    # Large rectangular contour covering significant area
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    img_area = float(h * w)
    large_rect_present = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.3 * img_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            large_rect_present = True
            break

    return white_ratio, lines_count, large_rect_present


def brightness_mean(gray: np.ndarray) -> float:
    return float(gray.mean())


def classify_image(path: str) -> str:
    img_bgr = load_image_bgr(path)
    gray = to_gray(img_bgr)
    bmean = brightness_mean(gray)

    # Priority 1: Faces
    face_count = detect_faces(img_bgr)
    if face_count > 0:
        return "face"

    # Priority 2: Document
    white_ratio, lines_count, large_rect = document_score(img_bgr)
    if white_ratio >= 0.5 and (lines_count >= 50 or large_rect):
        return "document"

    # Priority 3/4: Nightscape vs Landscape by brightness
    if bmean < 80.0:
        return "nightscape"
    else:
        return "landscape"


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Smart Image Type Classifier")
        self.root.geometry("700x500")

        self.image_path: Optional[str] = None
        self.preview_label = tk.Label(self.root, text="No image uploaded yet", anchor="center")
        self.preview_label.pack(fill="both", expand=True, padx=10, pady=10)

        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=10, pady=10)

        self.upload_btn = tk.Button(controls, text="Upload Image", command=self.on_upload)
        self.upload_btn.pack(side="left")

        self.detect_label_var = tk.StringVar(value="Detected: -")
        self.detect_label = tk.Label(controls, textvariable=self.detect_label_var)
        self.detect_label.pack(side="left", padx=10)

        # Dropdown to choose category
        self.choice_var = tk.StringVar(value=LABELS[0])
        self.dropdown = ttk.Combobox(controls, textvariable=self.choice_var, values=LABELS, state="readonly")
        self.dropdown.pack(side="left", padx=10)

        self.confirm_btn = tk.Button(controls, text="Confirm", command=self.on_confirm)
        self.confirm_btn.pack(side="left", padx=10)

        self.photo_image: Optional[tk.PhotoImage] = None

    def on_upload(self) -> None:
        filetypes = [("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select an image", filetypes=filetypes)
        if not path:
            return
        self.image_path = path
        try:
            pred = classify_image(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to classify image:\n{e}")
            return
        self.detect_label_var.set(f"Detected: {pred}")
        self.choice_var.set(pred)
        self.show_preview(path)

    def show_preview(self, path: str) -> None:
        # Prefer PIL for display; fallback to text only
        if Image is None or ImageTk is None:
            self.preview_label.config(text=os.path.basename(path))
            return
        try:
            img = Image.open(path)
            # Resize to fit
            max_w, max_h = 640, 360
            img.thumbnail((max_w, max_h))
            self.photo_image = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self.photo_image)
            self.preview_label.image = self.photo_image
            self.preview_label.config(text="")
        except Exception:
            self.preview_label.config(text=os.path.basename(path))

    def on_confirm(self) -> None:
        selection = self.choice_var.get()
        if not self.image_path:
            messagebox.showinfo("Info", "Please upload an image first.")
            return
        messagebox.showinfo("Confirmed", f"Image: {os.path.basename(self.image_path)}\nType: {selection}")
        print(f"CONFIRMED: path={self.image_path} type={selection}")


def main() -> None:
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
