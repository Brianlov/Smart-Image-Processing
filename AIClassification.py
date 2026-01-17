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
            win.state('zoomed')
        except Exception:
            pass

        main = tk.Frame(win)
        main.pack(fill="both", expand=True)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1, uniform="fig")
        main.columnconfigure(1, weight=0)
        main.columnconfigure(2, weight=1, uniform="fig")

        sync_enabled = [False]
        canvases = {}
        zoom_vars = {}
        zoom_funcs = {}
        panel_side = {}  # Track which side each panel is on
        last_interacted = ["left"]  # Track which side was last interacted with
        canvas_views = {"left": [0.5, 0.5], "right": [0.5, 0.5]}  # Track normalized center positions (0-1)

        # Helper to create image panel
        def create_panel(parent, col, label_text, img_rgb, is_left, side_name):
            panel = tk.Frame(parent, bd=2, relief="groove")
            panel.grid(row=0, column=col, sticky="nsew", padx=(10, 5) if is_left else (5, 10), pady=10)
            panel.rowconfigure(0, weight=0)
            panel.rowconfigure(1, weight=1)
            panel.rowconfigure(2, weight=0)
            panel.columnconfigure(0, weight=1)
            tk.Label(panel, text=label_text, font=("Segoe UI", 11, "bold")).grid(row=0, column=0, pady=(0, 6))

            canvas = tk.Canvas(panel, bg="#202020", highlightthickness=0)
            canvas.grid(row=1, column=0, sticky="nsew")
            cached = [None, None, None]  # photo, img_id, after_id

            controls = tk.Frame(panel)
            controls.grid(row=2, column=0, sticky="ew", pady=6)
            zoom_var = tk.DoubleVar(value=1.0)
            zoom_label = tk.Label(controls, text="Zoom: 1.00x")
            zoom_label.pack(side="left", padx=8)

            def apply_zoom(force=False, sync_other=True):
                if Image is None or ImageTk is None:
                    return
                z = max(0.1, min(4.0, zoom_var.get()))
                zoom_label.config(text=f"Zoom: {z:.2f}x")
                
                # Mark this side as last interacted when not syncing
                if not sync_other or not sync_enabled[0]:
                    last_interacted[0] = side_name
                
                if not force and cached[2] is not None:
                    canvas.after_cancel(cached[2])
                
                def do_zoom():
                    pil = Image.fromarray(img_rgb)
                    w, h = pil.size
                    new_w, new_h = max(1, int(w * z)), max(1, int(h * z))
                    resample = Image.BILINEAR if new_w * new_h > 4000000 else Image.LANCZOS
                    photo = ImageTk.PhotoImage(pil.resize((new_w, new_h), resample))
                    cached[0] = photo
                    canvas.delete("all")
                    
                    # Center the image in canvas
                    cw = canvas.winfo_width() or 1
                    ch = canvas.winfo_height() or 1
                    x_pos = max(0, (cw - new_w) // 2)
                    y_pos = max(0, (ch - new_h) // 2)
                    
                    cached[1] = canvas.create_image(x_pos, y_pos, anchor="nw", image=photo)
                    canvas.config(scrollregion=(0, 0, max(new_w, cw), max(new_h, ch)))
                    
                    # Restore pan position
                    if new_w > cw or new_h > ch:
                        # Image is larger, use scroll position
                        x_frac, y_frac = canvas_views[side_name]
                        canvas.xview_moveto(x_frac)
                        canvas.yview_moveto(y_frac)
                    
                    cached[2] = None
                    
                    # Sync with other side if enabled and we're the source
                    if sync_other and sync_enabled[0]:
                        other = "right" if side_name == "left" else "left"
                        if other in zoom_vars and other in zoom_funcs:
                            # Update other zoom var and apply
                            if zoom_vars[other].get() != z:
                                zoom_vars[other].set(z)
                                zoom_funcs[other](force=True, sync_other=False)
                
                if force:
                    do_zoom()
                else:
                    cached[2] = canvas.after(150, do_zoom)

            def on_button_zoom(delta):
                zoom_var.set(max(0.1, min(4.0, zoom_var.get() + delta)))
                apply_zoom()

            tk.Button(controls, text="-", width=3, command=lambda: on_button_zoom(-0.1)).pack(side="left")
            tk.Button(controls, text="+", width=3, command=lambda: on_button_zoom(0.1)).pack(side="left", padx=4)
            ttk.Scale(controls, from_=0.1, to=4.0, orient="horizontal", variable=zoom_var, command=lambda e: apply_zoom()).pack(side="left", fill="x", expand=True, padx=8)

            return canvas, zoom_var, apply_zoom

        left_canvas, left_zoom_var, apply_left_zoom = create_panel(main, 0, "Figure 1: Original", original_rgb, True, "left")
        right_canvas, right_zoom_var, apply_right_zoom = create_panel(main, 2, "Figure 2: Enhanced", enhanced_rgb, False, "right")
        
        canvases = {"left": left_canvas, "right": right_canvas}
        zoom_vars = {"left": left_zoom_var, "right": right_zoom_var}
        zoom_funcs = {"left": apply_left_zoom, "right": apply_right_zoom}
        panel_side = {"left": True, "right": True}  # Mark both as initialized

        # Event handlers with sync
        def make_handlers(canvas, side):
            other = "right" if side == "left" else "left"
            
            def on_wheel(event):
                last_interacted[0] = side
                step = 0.1 if event.delta > 0 else -0.1
                zoom_vars[side].set(max(0.1, min(4.0, zoom_vars[side].get() + step)))
                zoom_funcs[side](sync_other=True)
            
            def on_press(event):
                last_interacted[0] = side
                canvas.config(cursor="fleur")
                canvas.scan_mark(event.x, event.y)
                if sync_enabled[0]:
                    canvases[other].config(cursor="fleur")
                    canvases[other].scan_mark(event.x, event.y)
            
            def on_drag(event):
                canvas.scan_dragto(event.x, event.y, gain=1)
                # Update tracked view position (normalized 0-1)
                try:
                    x_view = canvas.xview()
                    y_view = canvas.yview()
                    if x_view and y_view and len(x_view) >= 2 and len(y_view) >= 2:
                        # Calculate center of visible region
                        canvas_views[side] = [(x_view[0] + x_view[1]) / 2, (y_view[0] + y_view[1]) / 2]
                except:
                    pass
                
                if sync_enabled[0]:
                    canvases[other].scan_dragto(event.x, event.y, gain=1)
                    try:
                        x_view = canvases[other].xview()
                        y_view = canvases[other].yview()
                        if x_view and y_view and len(x_view) >= 2 and len(y_view) >= 2:
                            canvas_views[other] = [(x_view[0] + x_view[1]) / 2, (y_view[0] + y_view[1]) / 2]
                    except:
                        pass
            
            def on_release(event):
                canvas.config(cursor="")
                # Save final view position
                try:
                    x_view = canvas.xview()
                    y_view = canvas.yview()
                    if x_view and y_view and len(x_view) >= 2 and len(y_view) >= 2:
                        canvas_views[side] = [(x_view[0] + x_view[1]) / 2, (y_view[0] + y_view[1]) / 2]
                except:
                    pass
                
                if sync_enabled[0]:
                    canvases[other].config(cursor="")
                    try:
                        x_view = canvases[other].xview()
                        y_view = canvases[other].yview()
                        if x_view and y_view and len(x_view) >= 2 and len(y_view) >= 2:
                            canvas_views[other] = [(x_view[0] + x_view[1]) / 2, (y_view[0] + y_view[1]) / 2]
                    except:
                        pass
            
            canvas.bind("<MouseWheel>", on_wheel)
            canvas.bind("<ButtonPress-1>", on_press)
            canvas.bind("<B1-Motion>", on_drag)
            canvas.bind("<ButtonRelease-1>", on_release)
            canvas.bind("<Configure>", lambda e: zoom_funcs[side](force=True, sync_other=False))

        make_handlers(left_canvas, "left")
        make_handlers(right_canvas, "right")

        # Sync button
        middle_panel = tk.Frame(main)
        middle_panel.grid(row=0, column=1, sticky="ns", padx=5, pady=10)
        sync_button = tk.Button(middle_panel, text="⛓\n\nLink", font=("Arial", 10, "bold"),
                               width=5, height=6, relief="raised", bg="#e0e0e0")
        sync_button.pack(expand=True)
        
        def toggle_sync():
            sync_enabled[0] = not sync_enabled[0]
            if sync_enabled[0]:
                sync_button.config(relief="sunken", bg="#4CAF50", fg="white", text="🔗\n\nLinked")
                # Sync from last interacted side to the other
                source = last_interacted[0]
                target = "right" if source == "left" else "left"
                
                # Sync zoom level
                zoom_vars[target].set(zoom_vars[source].get())
                zoom_funcs[target](force=True, sync_other=False)
                
                # Sync pan position after zoom is complete
                def sync_pan():
                    try:
                        # Get source canvas normalized center position
                        x_center, y_center = canvas_views[source]
                        
                        # Apply to target canvas
                        target_canvas = canvases[target]
                        x_view = target_canvas.xview()
                        y_view = target_canvas.yview()
                        
                        if x_view and y_view and len(x_view) >= 2 and len(y_view) >= 2:
                            # Calculate moveto position to center at the same point
                            x_width = x_view[1] - x_view[0]
                            y_height = y_view[1] - y_view[0]
                            
                            x_moveto = max(0, min(1 - x_width, x_center - x_width / 2))
                            y_moveto = max(0, min(1 - y_height, y_center - y_height / 2))
                            
                            target_canvas.xview_moveto(x_moveto)
                            target_canvas.yview_moveto(y_moveto)
                            canvas_views[target] = [x_center, y_center]
                    except Exception as e:
                        print(f"Sync pan error: {e}")
                
                canvases[target].after(250, sync_pan)
            else:
                sync_button.config(relief="raised", bg="#e0e0e0", fg="black", text="⛓\n\nLink")
        
        sync_button.config(command=toggle_sync)

        # Save Controls
        save_frame = tk.Frame(win)
        save_frame.pack(pady=10, padx=10, fill="x")
        
        tk.Label(save_frame, text="Compression:", font=("Arial", 10)).pack(side="left", padx=5)
        compression_var = tk.StringVar(value="BALANCED")
        ttk.Combobox(save_frame, textvariable=compression_var,
                    values=list(self.COMPRESSION_PRESETS.keys()),
                    state="readonly", width=12).pack(side="left", padx=5)
        
        save_status = tk.Label(save_frame, text="", fg="green", font=("Arial", 9))
        
        def save_image(img_rgb, img_type):
            original_name = os.path.basename(self.image_path)
            name_parts = os.path.splitext(original_name)
            suggested_name = f"{name_parts[0]}_{img_type}{name_parts[1]}"
            
            save_path = filedialog.asksaveasfilename(
                title=f"Save {img_type.capitalize()} Image",
                initialfile=suggested_name,
                filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All files", "*.*")],
                defaultextension=".jpg"
            )
            
            if not save_path:
                return
            
            try:
                save_status.config(text="Saving...", fg="blue")
                win.update()
                
                orig_size, comp_size = self._compress_and_save(img_rgb, save_path, compression_var.get())
                reduction = ((orig_size - comp_size) / orig_size) * 100 if orig_size > 0 else 0
                # Show absolute reduction
                saved_mb = orig_size - comp_size
                save_status.config(text=f"✓ Saved! {comp_size:.2f}MB (saved {saved_mb:.2f}MB, {reduction:.1f}%)", fg="green")
                print(f"Saved: {save_path} | Uncompressed: {orig_size:.2f}MB | Compressed: {comp_size:.2f}MB | Reduction: {reduction:.1f}%")
            except Exception as e:
                save_status.config(text=f"✗ Error: {str(e)[:50]}", fg="red")
                messagebox.showerror("Save Error", f"Failed to save image:\n{e}")
        
        tk.Button(save_frame, text="Save Original", command=lambda: save_image(original_rgb, "original"),
                 bg="#2196F3", fg="white", font=("Arial", 9, "bold"), padx=15).pack(side="left", padx=5)
        tk.Button(save_frame, text="Save Enhanced", command=lambda: save_image(enhanced_rgb, "enhanced"),
                 bg="#4CAF50", fg="white", font=("Arial", 9, "bold"), padx=15).pack(side="left", padx=5)
        save_status.pack(side="left", padx=10)

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

    # Document: 4-step morph sequence (import from morph_seq if available)
    def _run_document_enhance(self, path: str) -> np.ndarray:
        try:
            from morph_seq import process_morph_seq
            res = process_morph_seq(path, out_dir="outputs", save_intermediate=True)
            binary = res.get("step4_closed")
            if binary is None:
                raise RuntimeError("Document pipeline returned no binary result")
            # Convert to RGB for display
            if binary.ndim == 2:
                rgb_bin = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            else:
                rgb_bin = binary
            return rgb_bin
        except Exception:
            # Fallback: do erosion -> Otsu -> closing inline
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
            original = FE.load_and_prep(path)
            skin_mask = FE.get_refined_skin_mask(original)
            skin_enhanced = FE.apply_glamour_skin(original, skin_mask)
            sharpened = FE.enhance_details(skin_enhanced, amount=2.0)
            features_popped = FE.pixel_pop_eyes(sharpened)
            color_corrected = FE.adjust_saturation(features_popped, saturation=1.0)
            stretched = FE.apply_contrast_stretching(color_corrected)
            final_output = FE.apply_histogram_equalization(stretched)
            return cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)
        except Exception:
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