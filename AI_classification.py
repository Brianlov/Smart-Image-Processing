"""Lightweight AI GUI classifier for 4 classes: nightscape, landscape, document, face.

Approach:
- Primary: Zero-shot CLIP (open-clip-torch, ViT-B/32) over 4 textual prompts.
- Fallback: Simple OpenCV heuristics (face cascade, document cues, brightness).

Run:
  python .\AI_classification.py

Optional installs:
  pip install open-clip-torch pillow opencv-python
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from tkinter import ttk
except Exception:
    raise RuntimeError("Tkinter is required to run this GUI.")

# Optional deps
try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

try:
    import cv2
except Exception:
    cv2 = None

# AI model (optional)
_clip_available = False
try:
    import torch
    import open_clip
    _clip_available = True
except Exception:
    _clip_available = False


LABELS = ["nightscape", "landscape", "document", "face"]
PROMPTS = {
    "nightscape": "a night cityscape photograph with bright lights and dark sky",
    "landscape": "a daytime natural landscape photograph with sky or mountains or trees",
    "document": "a scanned paper document page with text on a white background",
    "face": "a human face portrait photograph"
}


def _cv2_to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _load_cv_image(path: str) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV not installed. Install opencv-python for fallback heuristics.")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def _face_count(img_bgr: np.ndarray) -> int:
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            return 0
        gray = _cv2_to_gray(img_bgr)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return len(faces)
    except Exception:
        return 0


def _document_cues(img_bgr: np.ndarray) -> Tuple[float, int, bool]:
    gray = _cv2_to_gray(img_bgr)
    # Otsu white ratio
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = float(np.mean(binary == 255))
    # Lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    lines_count = 0 if lines is None else len(lines)
    # Large rectangular contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    area_total = float(h*w)
    large_rect = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.3 * area_total:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4:
            large_rect = True
            break
    return white_ratio, lines_count, large_rect


def _brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def classify_heuristic(path: str) -> Tuple[str, Dict[str, float]]:
    """Heuristic fallback if AI model unavailable."""
    if cv2 is None:
        raise RuntimeError("OpenCV required for heuristic classifier.")
    img_bgr = _load_cv_image(path)
    gray = _cv2_to_gray(img_bgr)

    scores: Dict[str, float] = {k: 0.0 for k in LABELS}

    # Face
    fc = _face_count(img_bgr)
    if fc > 0:
        scores["face"] += 1.0 + 0.5 * min(fc, 3)

    # Document
    white_ratio, lines_count, large_rect = _document_cues(img_bgr)
    scores["document"] += (white_ratio - 0.5) * 2.0  # >0 when white_ratio>0.5
    if lines_count >= 50:
        scores["document"] += 0.5
    if large_rect:
        scores["document"] += 0.5

    # Night vs Landscape
    bmean = _brightness(gray)
    if bmean < 80:
        scores["nightscape"] += (80 - bmean) / 80.0
    else:
        scores["landscape"] += (bmean - 80) / 80.0

    # pick best
    label = max(scores.items(), key=lambda kv: kv[1])[0]
    # normalize for display
    total = sum(v for v in scores.values() if v > 0) or 1.0
    probs = {k: max(v, 0.0)/total for k, v in scores.items()}
    return label, probs


class ClipZeroShot:
    def __init__(self) -> None:
        if not _clip_available:
            raise RuntimeError("open-clip-torch and torch are required for AI classification.")
        # Model choice: ViT-B-32 with laion2b weights (small, fast)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        # Cache tokenized prompts
        texts = [PROMPTS[k] for k in LABELS]
        with torch.no_grad():
            self.text_tokens = self.tokenizer(texts)

    def predict(self, image_path: str) -> Tuple[str, Dict[str, float]]:
        # Load with PIL for preprocess
        if Image is None:
            raise RuntimeError("Pillow is required for CLIP preprocessing.")
        img = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(img).unsqueeze(0)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(self.text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

        probs = logits.cpu().numpy()
        result: Dict[str, float] = {}
        for i, k in enumerate(LABELS):
            result[k] = float(probs[i])
        label = LABELS[int(np.argmax(probs))]
        return label, result


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Image Type Classifier")
        self.root.geometry("760x560")

        self.image_path: Optional[str] = None
        self.ai: Optional[ClipZeroShot] = None

        # UI: top controls
        top = tk.Frame(root)
        top.pack(fill="x", padx=10, pady=8)

        self.upload_btn = tk.Button(top, text="Upload Image", command=self.on_upload)
        self.upload_btn.pack(side="left")

        self.detect_var = tk.StringVar(value="Detected: -")
        tk.Label(top, textvariable=self.detect_var).pack(side="left", padx=10)

        self.choice_var = tk.StringVar(value=LABELS[0])
        self.dropdown = ttk.Combobox(top, textvariable=self.choice_var, values=LABELS, state="readonly")
        self.dropdown.pack(side="left", padx=10)

        self.confirm_btn = tk.Button(top, text="Confirm", command=self.on_confirm)
        self.confirm_btn.pack(side="left", padx=10)

        # model status
        self.status_var = tk.StringVar(value="Model: loading when needed (CLIP zero-shot)")
        tk.Label(root, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10)

        # preview area
        self.preview = tk.Label(root, text="No image uploaded yet", anchor="center")
        self.preview.pack(fill="both", expand=True, padx=10, pady=10)
        self.photo_image: Optional[tk.PhotoImage] = None

    def ensure_model(self) -> None:
        if self.ai is not None:
            return
        if not _clip_available:
            self.status_var.set("Model: CLIP not installed; using heuristic fallback")
            return
        try:
            self.status_var.set("Model: loading CLIP (first time may download weights)…")
            self.root.update_idletasks()
            self.ai = ClipZeroShot()
            self.status_var.set("Model: CLIP ready")
        except Exception as e:
            self.ai = None
            self.status_var.set(f"Model: CLIP unavailable ({e}); using fallback")

    def on_upload(self) -> None:
        filetypes = [("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select an image", filetypes=filetypes)
        if not path:
            return
        self.image_path = path
        self.show_preview(path)

        # Load or fallback
        self.ensure_model()

        try:
            if self.ai is not None:
                label, probs = self.ai.predict(path)
            else:
                label, probs = classify_heuristic(path)
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed:\n{e}")
            self.detect_var.set("Detected: -")
            return

        # show results
        top1_prob = probs.get(label, 0.0)
        self.detect_var.set(f"Detected: {label} ({top1_prob:.2f})")
        self.choice_var.set(label)

    def show_preview(self, path: str) -> None:
        if Image is None or ImageTk is None:
            self.preview.config(text=os.path.basename(path))
            return
        try:
            img = Image.open(path)
            img.thumbnail((720, 420))
            self.photo_image = ImageTk.PhotoImage(img)
            self.preview.config(image=self.photo_image)
            self.preview.image = self.photo_image
            self.preview.config(text="")
        except Exception:
            self.preview.config(text=os.path.basename(path))

    def on_confirm(self) -> None:
        sel = self.choice_var.get()
        if not self.image_path:
            messagebox.showinfo("Info", "Please upload an image first.")
            return
        messagebox.showinfo("Confirmed", f"Image: {os.path.basename(self.image_path)}\nType: {sel}")
        print(f"CONFIRMED: path={self.image_path} type={sel}")


def main() -> None:
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
