r"""Lightweight AI GUI classifier for 4 classes: nightscape, landscape, document, face.

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
import threading
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

        # Progress bar (initially hidden, beside buttons)
        self.progress = ttk.Progressbar(top, mode="indeterminate", length=140)
        # self.progress.pack(side="left", padx=10) # Pack when needed

        # model status
        self.status_var = tk.StringVar(value="Model: loading when needed (CLIP zero-shot)")
        tk.Label(root, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10)

        # preview area
        self.preview = tk.Label(root, text="No image uploaded yet", anchor="center")
        self.preview.pack(fill="both", expand=True, padx=10, pady=10)
        self.photo_image: Optional[tk.PhotoImage] = None

    # ------------------ Compression Presets ------------------
    COMPRESSION_PRESETS = {
        "FAST": {"jpg_quality": 95, "png_compression": 1, "optimize": False},
        "BALANCED": {"jpg_quality": 90, "png_compression": 6, "optimize": True},
        "HIGH": {"jpg_quality": 85, "png_compression": 9, "optimize": True},
        "MAXIMUM": {"jpg_quality": 82, "png_compression": 9, "optimize": True}
    }

    def _compress_and_save(self, img_rgb: np.ndarray, output_path: str, preset_name: str) -> Tuple[float, float]:
        """Compress and save RGB image. Returns (original_size_mb, compressed_size_mb)"""
        if Image is None:
            raise RuntimeError("Pillow required for compression")
        
        preset = self.COMPRESSION_PRESETS[preset_name]
        file_ext = os.path.splitext(output_path)[1].lower()
        
        # Convert RGB to PIL Image
        pil_img = Image.fromarray(img_rgb)
        
        # Save to temp file first to get original size
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Save without compression for size comparison
            if file_ext in ['.jpg', '.jpeg']:
                pil_img.save(temp_path, 'JPEG', quality=100)
            else:
                pil_img.save(temp_path, 'PNG', compress_level=0)
            
            original_size = os.path.getsize(temp_path) / (1024 * 1024)
            
            # Save with compression
            if file_ext in ['.jpg', '.jpeg']:
                pil_img.save(output_path, 'JPEG',
                           quality=preset["jpg_quality"],
                           optimize=preset["optimize"],
                           progressive=True)
            elif file_ext == '.png':
                pil_img.save(output_path, 'PNG',
                           compress_level=preset["png_compression"],
                           optimize=preset["optimize"])
            else:
                # Default save for other formats
                pil_img.save(output_path)
            
            compressed_size = os.path.getsize(output_path) / (1024 * 1024)
            return original_size, compressed_size
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ------------------ Enhancement runners ------------------
    def _load_image_rgb(self, path: str) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV is required to run pipelines. Install opencv-python.")
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _rgb_to_photoimage(self, img_rgb: np.ndarray, max_size=(720, 420)) -> tk.PhotoImage:
        if Image is None or ImageTk is None:
            raise RuntimeError("Pillow is required to preview results. Install pillow.")
        im = Image.fromarray(img_rgb)
        im.thumbnail(max_size)
        return ImageTk.PhotoImage(im)

    def _show_pair_window(self, original_rgb: np.ndarray, enhanced_rgb: np.ndarray, title: str) -> None:
        win = tk.Toplevel(self.root)
        win.title(title)
        try:
            # Maximize window (Windows)
            win.state('zoomed')
        except Exception:
            pass

        # Main container
        main = tk.Frame(win)
        main.pack(fill="both", expand=True)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1, uniform="fig")
        main.columnconfigure(1, weight=0)  # Middle column for sync button
        main.columnconfigure(2, weight=1, uniform="fig")

        # Sync state
        sync_enabled = [False]  # Shared state for syncing

        # Left panel (Figure 1: Original)
        left_panel = tk.Frame(main, bd=2, relief="groove")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left_panel.rowconfigure(0, weight=0)
        left_panel.rowconfigure(1, weight=1)
        left_panel.rowconfigure(2, weight=0)
        left_panel.columnconfigure(0, weight=1)
        tk.Label(left_panel, text="Figure 1: Original", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, pady=(0, 6))

        # Canvas with scrollbars for left image
        left_canvas = tk.Canvas(left_panel, bg="#202020", highlightthickness=0)
        left_canvas.grid(row=1, column=0, sticky="nsew")
        left_img_id = [None]
        left_cached_photo = [None]  # Cache the PhotoImage to avoid recreating
        left_zoom_after_id = [None]  # For debouncing zoom

        left_controls = tk.Frame(left_panel)
        left_controls.grid(row=2, column=0, sticky="ew", pady=6)
        left_zoom_var = tk.DoubleVar(value=1.0)
        left_zoom_label = tk.Label(left_controls, text="Zoom: 1.00x")
        left_zoom_label.pack(side="left", padx=8)

        def apply_left_zoom(force=False) -> None:
            if Image is None or ImageTk is None:
                return
            try:
                z = max(0.1, min(4.0, float(left_zoom_var.get())))
            except Exception:
                z = 1.0
            
            # Update label immediately
            left_zoom_label.config(text=f"Zoom: {z:.2f}x")
            
            # Debounce: only recreate image after user stops adjusting
            if not force and left_zoom_after_id[0] is not None:
                left_canvas.after_cancel(left_zoom_after_id[0])
            
            def do_zoom():
                pil = Image.fromarray(original_rgb)
                w, h = pil.size
                new_w, new_h = max(1, int(w * z)), max(1, int(h * z))
                # Use BILINEAR for faster resizing during zoom, LANCZOS for final
                resample = Image.BILINEAR if new_w * new_h > 4000000 else Image.LANCZOS
                pil_z = pil.resize((new_w, new_h), resample)
                photo = ImageTk.PhotoImage(pil_z)
                left_cached_photo[0] = photo
                
                # Update canvas
                left_canvas.delete("all")
                left_img_id[0] = left_canvas.create_image(0, 0, anchor="nw", image=photo)
                left_canvas.config(scrollregion=(0, 0, new_w, new_h))
                left_zoom_after_id[0] = None
            
            if force:
                do_zoom()
            else:
                left_zoom_after_id[0] = left_canvas.after(150, do_zoom)

        tk.Button(left_controls, text="-", width=3, command=lambda: (left_zoom_var.set(max(0.1, left_zoom_var.get() - 0.1)), apply_left_zoom())).pack(side="left")
        tk.Button(left_controls, text="+", width=3, command=lambda: (left_zoom_var.set(min(4.0, left_zoom_var.get() + 0.1)), apply_left_zoom())).pack(side="left", padx=4)
        ttk.Scale(left_controls, from_=0.1, to=4.0, orient="horizontal", variable=left_zoom_var, command=lambda e: apply_left_zoom()).pack(side="left", fill="x", expand=True, padx=8)

        def on_left_wheel(event) -> None:
            delta = event.delta or 0
            step = 0.1 if delta > 0 else -0.1
            left_zoom_var.set(max(0.1, min(4.0, left_zoom_var.get() + step)))
            apply_left_zoom()
            if sync_enabled[0]:
                right_zoom_var.set(left_zoom_var.get())
                apply_right_zoom()
        
        # Pan functionality - proper implementation
        def on_left_press(event) -> None:
            left_canvas.config(cursor="fleur")
            left_canvas.scan_mark(event.x, event.y)
            if sync_enabled[0]:
                right_canvas.config(cursor="fleur")
                right_canvas.scan_mark(event.x, event.y)
        
        def on_left_drag(event) -> None:
            left_canvas.scan_dragto(event.x, event.y, gain=1)
            if sync_enabled[0]:
                right_canvas.scan_dragto(event.x, event.y, gain=1)
        
        def on_left_release(event) -> None:
            left_canvas.config(cursor="")
            if sync_enabled[0]:
                right_canvas.config(cursor="")

        left_canvas.bind("<MouseWheel>", on_left_wheel)
        left_canvas.bind("<ButtonPress-1>", on_left_press)
        left_canvas.bind("<B1-Motion>", on_left_drag)
        left_canvas.bind("<ButtonRelease-1>", on_left_release)
        left_canvas.bind("<Configure>", lambda e: apply_left_zoom(force=True))

        # Middle column - Sync button
        middle_panel = tk.Frame(main)
        middle_panel.grid(row=0, column=1, sticky="ns", padx=5, pady=10)
        
        sync_button = tk.Button(middle_panel, text="⛓\n\nLink", font=("Arial", 10, "bold"),
                               width=5, height=6, relief="raised", bg="#e0e0e0")
        sync_button.pack(expand=True)
        
        def toggle_sync():
            sync_enabled[0] = not sync_enabled[0]
            if sync_enabled[0]:
                sync_button.config(relief="sunken", bg="#4CAF50", fg="white", text="🔗\n\nLinked")
                # Sync zoom levels
                right_zoom_var.set(left_zoom_var.get())
                apply_right_zoom()
            else:
                sync_button.config(relief="raised", bg="#e0e0e0", fg="black", text="⛓\n\nLink")
        
        sync_button.config(command=toggle_sync)

        # Right panel (Figure 2: Enhanced)
        right_panel = tk.Frame(main, bd=2, relief="groove")
        right_panel.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=10)
        right_panel.rowconfigure(0, weight=0)
        right_panel.rowconfigure(1, weight=1)
        right_panel.rowconfigure(2, weight=0)
        right_panel.columnconfigure(0, weight=1)
        tk.Label(right_panel, text="Figure 2: Enhanced", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, pady=(0, 6))

        # Canvas with scrollbars for right image
        right_canvas = tk.Canvas(right_panel, bg="#202020", highlightthickness=0)
        right_canvas.grid(row=1, column=0, sticky="nsew")
        right_img_id = [None]
        right_cached_photo = [None]  # Cache the PhotoImage to avoid recreating
        right_zoom_after_id = [None]  # For debouncing zoom

        right_controls = tk.Frame(right_panel)
        right_controls.grid(row=2, column=0, sticky="ew", pady=6)
        right_zoom_var = tk.DoubleVar(value=1.0)
        right_zoom_label = tk.Label(right_controls, text="Zoom: 1.00x")
        right_zoom_label.pack(side="left", padx=8)

        def apply_right_zoom(force=False) -> None:
            if Image is None or ImageTk is None:
                return
            try:
                z = max(0.1, min(4.0, float(right_zoom_var.get())))
            except Exception:
                z = 1.0
            
            # Update label immediately
            right_zoom_label.config(text=f"Zoom: {z:.2f}x")
            
            # Debounce: only recreate image after user stops adjusting
            if not force and right_zoom_after_id[0] is not None:
                right_canvas.after_cancel(right_zoom_after_id[0])
            
            def do_zoom():
                pil = Image.fromarray(enhanced_rgb)
                w, h = pil.size
                new_w, new_h = max(1, int(w * z)), max(1, int(h * z))
                # Use BILINEAR for faster resizing during zoom, LANCZOS for final
                resample = Image.BILINEAR if new_w * new_h > 4000000 else Image.LANCZOS
                pil_z = pil.resize((new_w, new_h), resample)
                photo = ImageTk.PhotoImage(pil_z)
                right_cached_photo[0] = photo
                
                # Update canvas
                right_canvas.delete("all")
                right_img_id[0] = right_canvas.create_image(0, 0, anchor="nw", image=photo)
                right_canvas.config(scrollregion=(0, 0, new_w, new_h))
                right_zoom_after_id[0] = None
            
            if force:
                do_zoom()
            else:
                right_zoom_after_id[0] = right_canvas.after(150, do_zoom)

        tk.Button(right_controls, text="-", width=3, command=lambda: (right_zoom_var.set(max(0.1, right_zoom_var.get() - 0.1)), apply_right_zoom())).pack(side="left")
        tk.Button(right_controls, text="+", width=3, command=lambda: (right_zoom_var.set(min(4.0, right_zoom_var.get() + 0.1)), apply_right_zoom())).pack(side="left", padx=4)
        ttk.Scale(right_controls, from_=0.1, to=4.0, orient="horizontal", variable=right_zoom_var, command=lambda e: apply_right_zoom()).pack(side="left", fill="x", expand=True, padx=8)

        def on_right_wheel(event) -> None:
            delta = event.delta or 0
            step = 0.1 if delta > 0 else -0.1
            right_zoom_var.set(max(0.1, min(4.0, right_zoom_var.get() + step)))
            apply_right_zoom()
            if sync_enabled[0]:
                left_zoom_var.set(right_zoom_var.get())
                apply_left_zoom()
        
        # Pan functionality - proper implementation
        def on_right_press(event) -> None:
            right_canvas.config(cursor="fleur")
            right_canvas.scan_mark(event.x, event.y)
            if sync_enabled[0]:
                left_canvas.config(cursor="fleur")
                left_canvas.scan_mark(event.x, event.y)
        
        def on_right_drag(event) -> None:
            right_canvas.scan_dragto(event.x, event.y, gain=1)
            if sync_enabled[0]:
                left_canvas.scan_dragto(event.x, event.y, gain=1)
        
        def on_right_release(event) -> None:
            right_canvas.config(cursor="")
            if sync_enabled[0]:
                left_canvas.config(cursor="")

        right_canvas.bind("<MouseWheel>", on_right_wheel)
        right_canvas.bind("<ButtonPress-1>", on_right_press)
        right_canvas.bind("<B1-Motion>", on_right_drag)
        right_canvas.bind("<ButtonRelease-1>", on_right_release)
        right_canvas.bind("<Configure>", lambda e: apply_right_zoom(force=True))

        # Save Controls (no caption label)
        save_frame = tk.Frame(win)
        save_frame.pack(pady=10, padx=10, fill="x")
        
        tk.Label(save_frame, text="Compression:", font=("Arial", 10)).pack(side="left", padx=5)
        
        compression_var = tk.StringVar(value="BALANCED")
        compression_dropdown = ttk.Combobox(save_frame, textvariable=compression_var,
                                           values=list(self.COMPRESSION_PRESETS.keys()),
                                           state="readonly", width=12)
        compression_dropdown.pack(side="left", padx=5)
        
        tk.Button(save_frame, text="Save Original", command=lambda: save_original(),
                 bg="#2196F3", fg="white", font=("Arial", 9, "bold"), padx=15).pack(side="left", padx=5)
        
        tk.Button(save_frame, text="Save Enhanced", command=lambda: save_enhanced(),
                 bg="#4CAF50", fg="white", font=("Arial", 9, "bold"), padx=15).pack(side="left", padx=5)
        
        save_status = tk.Label(save_frame, text="", fg="green", font=("Arial", 9))
        save_status.pack(side="left", padx=10)
        
        def save_original():
            from tkinter import filedialog
            
            original_name = os.path.basename(self.image_path)
            name_parts = os.path.splitext(original_name)
            suggested_name = f"{name_parts[0]}_original{name_parts[1]}"
            
            filetypes = [("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All files", "*.*")]
            save_path = filedialog.asksaveasfilename(
                title="Save Original Image",
                initialfile=suggested_name,
                filetypes=filetypes,
                defaultextension=".jpg"
            )
            
            if not save_path:
                return
            
            try:
                save_status.config(text="Saving...", fg="blue")
                win.update()
                
                preset_name = compression_var.get()
                orig_size, comp_size = self._compress_and_save(original_rgb, save_path, preset_name)
                
                reduction = ((orig_size - comp_size) / orig_size) * 100 if orig_size > 0 else 0
                save_status.config(
                    text=f"✓ Saved! ({comp_size:.2f}MB, {reduction:.1f}% reduction)",
                    fg="green"
                )
                print(f"Saved: {save_path} | Original: {orig_size:.2f}MB | Compressed: {comp_size:.2f}MB")
            except Exception as e:
                save_status.config(text=f"✗ Error: {str(e)[:50]}", fg="red")
                messagebox.showerror("Save Error", f"Failed to save image:\n{e}")
        
        def save_enhanced():
            from tkinter import filedialog
            
            # Suggest filename
            original_name = os.path.basename(self.image_path)
            name_parts = os.path.splitext(original_name)
            suggested_name = f"{name_parts[0]}_enhanced{name_parts[1]}"
            
            filetypes = [("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All files", "*.*")]
            save_path = filedialog.asksaveasfilename(
                title="Save Enhanced Image",
                initialfile=suggested_name,
                filetypes=filetypes,
                defaultextension=".jpg"
            )
            
            if not save_path:
                return
            
            try:
                save_status.config(text="Saving...", fg="blue")
                win.update()
                
                preset_name = compression_var.get()
                orig_size, comp_size = self._compress_and_save(enhanced_rgb, save_path, preset_name)
                
                reduction = ((orig_size - comp_size) / orig_size) * 100 if orig_size > 0 else 0
                save_status.config(
                    text=f"✓ Saved! ({comp_size:.2f}MB, {reduction:.1f}% reduction)",
                    fg="green"
                )
                print(f"Saved: {save_path} | Original: {orig_size:.2f}MB | Compressed: {comp_size:.2f}MB")
            except Exception as e:
                save_status.config(text=f"✗ Error: {str(e)[:50]}", fg="red")
                messagebox.showerror("Save Error", f"Failed to save image:\n{e}")

        # Initial render
        win.update_idletasks()
        apply_left_zoom(force=True)
        apply_right_zoom(force=True)

    # Nightscape: median (3x3) + CLAHE on LAB L
    def _run_night_enhance(self, path: str) -> np.ndarray:
        rgb = self._load_image_rgb(path)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        den = cv2.medianBlur(bgr, 3)
        lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)

    # Document: run DocScanner pipeline and display its final output
    def _run_document_enhance(self, path: str) -> np.ndarray:
        # Ensure OpenCV present
        if cv2 is None:
            raise RuntimeError("OpenCV is required to run document pipeline.")
        try:
            import DocScanner as DS
            res = DS.process_document(
                input_path=path,
                out_dir="outputs",
                page="A4",
                scale_long=1200,
                do_ocr=False,
                # Key tunables aligned with user command
                illum_method="divide",
                illum_blur_frac=0.05,
                block_size=31,
                C=3,
                canny_low=30,
                canny_high=100,
                morph_ksize=1,
                morph_iters=0,
                fallback_use_whole=True,
                min_quad_area_ratio=0.15,
            )

            final_bin = res.get("binary")
            if final_bin is None:
                raise RuntimeError("DocScanner pipeline returned no final binary result")
            # Display must match final pipeline output (scan_08_clean)
            if final_bin.ndim == 2:
                return cv2.cvtColor(final_bin, cv2.COLOR_GRAY2RGB)
            return final_bin
        except Exception:
            # Fallback: simple inline erosion -> Otsu -> closing
            rgb = self._load_image_rgb(path)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            eroded = cv2.erode(gray, kernel, iterations=1)
            _, binary = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            return cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)

    # Landscape: bilateral denoise + CLAHE with sky protection + unsharp mask
    def _run_landscape_enhance(self, path: str) -> np.ndarray:
        rgb = self._load_image_rgb(path)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        den = cv2.bilateralFilter(bgr, d=9, sigmaColor=100, sigmaSpace=75)
        lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
        l_orig, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_orig)
        l_norm = l_orig.astype(np.float32) / 255.0
        sky_power = 2.0
        blend_strength = 0.55
        protection_mask = np.power(l_norm, sky_power)
        enhance_weight = (1.0 - protection_mask) * blend_strength
        l_final = (l_clahe.astype(np.float32) * enhance_weight + l_orig.astype(np.float32) * (1.0 - enhance_weight)).astype(np.uint8)
        lab_enh = cv2.merge([l_final, a, b])
        bgr_enh = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
        # Unsharp
        radius = 1.0
        amount = 0.8
        blurred = cv2.GaussianBlur(bgr_enh, (0, 0), radius)
        sharp = cv2.addWeighted(bgr_enh, 1.0 + amount, blurred, -amount, 0)
        return cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)

    # Face: reuse FaceEnhancement functions if available; fallback to simple pipeline
    def _run_face_enhance(self, path: str) -> np.ndarray:
        try:
            import FaceEnhancement as FE
            import cv2
            import numpy as np

            # 1. Load & Prep
            original = FE.load_and_prep(path)
            
            # 2. Noise Classification
            noise_type = FE.classify_noise_type(original)
            
            # 3. Denoising
            if noise_type == "gaussian":
                # Gaussian Path: Light (5x5) & Strong (9x9) Gaussian Blur
                denoised_light = cv2.GaussianBlur(original, (5, 5), 0)
                denoised_strong = cv2.GaussianBlur(original, (9, 9), 0)
            elif noise_type == "impulse":
                print(f"Detected {noise_type.upper()} Noise -> Using Median Filter Strategy.")
                # Median Path: Face=3, BG=5
                denoised_light = cv2.medianBlur(original, 3)
                denoised_strong = cv2.medianBlur(original, 5)
            else:
                # Legacy Path: Smart Denoise (NLM)
                denoised_light, _ = FE.apply_smart_denoise(original, override_h=10)
                denoised_strong, _ = FE.apply_smart_denoise(original, override_h=30)
            
            # 4. Blending with Skin Mask
            skin_mask = FE.get_refined_skin_mask(denoised_light)
            mask_norm = skin_mask.astype(float) / 255.0
            mask_norm = cv2.merge([mask_norm, mask_norm, mask_norm])
            
            denoised_combined = (denoised_light * mask_norm) + (denoised_strong * (1 - mask_norm))
            denoised_combined = denoised_combined.astype(np.uint8)
            
            # 5. Skin Smoothing
            skin_enhanced = FE.apply_glamour_skin(denoised_combined, skin_mask)
            
            # 6. Eye Enhancement
            features_popped = FE.pixel_pop_eyes(skin_enhanced)
            
            # 7. Color & Tone
            # Base Saturation
            val_sat = 1.2 if noise_type == "gaussian" else 1.0
            color_corrected = FE.adjust_saturation(features_popped, saturation=val_sat)
            
            # Warmth (Red Boost)
            warmed = FE.apply_warmth(color_corrected, amount=15)
            
            if noise_type == "gaussian":
                # Gaussian Path: Skip Contrast Stretch, Use CLAHE, Skip Sharpen
                final_output = FE.apply_histogram_equalization(warmed)
            else:
                # Legacy Path: Stretch -> CLAHE -> Polish -> Sharpen
                stretched = FE.apply_contrast_stretching(warmed)
                clahe_result = FE.apply_histogram_equalization(stretched)
                polished = cv2.bilateralFilter(clahe_result, d=5, sigmaColor=20, sigmaSpace=20)
                final_output = FE.apply_masked_sharpening(polished, skin_mask, amount=2.0)

            return cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"Face Pipeline Error: {e}")
            # Fallback: bilateral on face regions not guaranteed, so apply globally but conservatively
            rgb = self._load_image_rgb(path)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            smooth = cv2.bilateralFilter(bgr, d=-1, sigmaColor=50, sigmaSpace=20)
            lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            lab2 = cv2.merge([l2, a, b])
            bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
            # mild unsharp
            blurred = cv2.GaussianBlur(bgr2, (0, 0), 1.5)
            sharp = cv2.addWeighted(bgr2, 1.5, blurred, -0.5, 0)
            return cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)

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

        # Disable UI during processing
        self.confirm_btn.config(state="disabled")
        self.upload_btn.config(state="disabled")
        self.progress.pack(side="left", padx=10)
        self.progress.start(10)

        # Start background thread
        thread = threading.Thread(target=self._enhancement_worker, args=(sel,))
        thread.start()
        self.root.after(100, self._monitor_enhancement, thread)

    def _enhancement_worker(self, sel: str) -> None:
        try:
            self._worker_result_rgb = None
            self._worker_error = None
            
            orig_rgb = self._load_image_rgb(self.image_path)
            if sel == "face":
                enh_rgb = self._run_face_enhance(self.image_path)
            elif sel == "document":
                enh_rgb = self._run_document_enhance(self.image_path)
            elif sel == "nightscape":
                enh_rgb = self._run_night_enhance(self.image_path)
            elif sel == "landscape":
                enh_rgb = self._run_landscape_enhance(self.image_path)
            else:
                raise ValueError(f"Unknown type: {sel}")
                
            self._worker_result_rgb = (orig_rgb, enh_rgb, sel)
        except Exception as e:
            self._worker_error = e

    def _monitor_enhancement(self, thread: threading.Thread) -> None:
        if thread.is_alive():
            # Check again in 100ms
            self.root.after(100, self._monitor_enhancement, thread)
        else:
            # Thread finished
            self.progress.stop()
            self.progress.pack_forget()
            self.confirm_btn.config(state="normal")
            self.upload_btn.config(state="normal")

            if self._worker_error:
                messagebox.showerror("Pipeline Error", f"Failed to run pipeline:\n{self._worker_error}")
            elif self._worker_result_rgb:
                orig_rgb, enh_rgb, sel = self._worker_result_rgb
                self._show_pair_window(orig_rgb, enh_rgb, title=f"{sel.capitalize()} Enhancement Result")
                print(f"CONFIRMED: path={self.image_path} type={sel}")


def main() -> None:
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
