import cv2
import numpy as np
import pyrealsense2 as rs
import tkinter as tk
import threading
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import ttk
from PIL import Image, ImageTk
import time
import os

from live_segmentation import SegmentationEngine, mask_to_colour, analyse_vials, VIAL_VOLUME_ML
from live_classification import ClassificationEngine
from segmentation import CLASS_NAMES

# ============================================================
#  FRONT END APPLICATION (THREADED)
# ============================================================

class FrontEndApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Vial Analysis")

        # Keybinds
        self.root.bind("<space>", self.capture_photo_event)
        self.root.bind("<c>", self.capture_photo_event)

        # Mode selection variable
        self.mode = tk.StringVar(value="classification")

        # Engines
        self.seg_engine = SegmentationEngine()
        self.cls_engine = ClassificationEngine()

        # Camera setup
        self.use_realsense = False
        self.pipeline = None
        self.cap = None
        self.initialise_camera()

        # UI setup
        self.build_ui()

        # Shared thread buffers
        self.processed_frame = None
        self.last_frame = None
        self.last_info = ""
        self.thread_running = True

        threading.Thread(target=self.worker_loop, daemon=True).start()
        self.update_frame()

    # ============================================================
    # CAMERA SETUP
    # ============================================================

    def initialise_camera(self):
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
            self.use_realsense = True
            print("RealSense detected.")
        except Exception:
            print("Using webcam fallback.")
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise RuntimeError("No camera detected.")
            self.use_realsense = False

    # ============================================================
    # UI BUILD
    # ============================================================

    def build_ui(self):
        self.main_frame = tb.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        # VIDEO FEED
        self.video_label = tb.Label(self.main_frame)
        self.video_label.grid(row=0, column=0, padx=10, pady=10)

        # INFO PANEL
        self.info_panel = tb.Frame(self.main_frame)
        self.info_panel.grid(row=0, column=1, sticky="n", padx=10, pady=10)

        # Dynamic title
        self.info_title = tb.Label(self.info_panel, text="Information", font=("Segoe UI", 14, "bold"))
        self.info_title.pack(anchor="nw")

        # Main info text
        self.info_text = tb.Text(self.info_panel, width=40, height=18, font=("Consolas", 10))
        self.info_text.pack(anchor="nw", pady=(5, 10))

        ttk.Separator(self.info_panel, orient="horizontal").pack(fill="x", pady=5)

        # METRICS PANEL
        tb.Label(self.info_panel, text="Metrics", font=("Segoe UI", 12, "bold")).pack(anchor="nw")
        self.metrics_text = tb.Text(self.info_panel, width=40, height=6, font=("Consolas", 10))
        self.metrics_text.pack(anchor="nw", pady=(2, 10))

        # BOTTOM CONTROLS
        self.bottom_bar = tb.Frame(self.root)
        self.bottom_bar.pack(side="bottom", pady=10)

        tb.Label(self.bottom_bar, text="Mode:", font=("Segoe UI", 12)).pack(side="left", padx=10)

        self.mode_frame = tb.Frame(self.bottom_bar)
        self.mode_frame.pack(side="left")

        self.class_btn = tb.Button(self.mode_frame, text="Classification",
                                   bootstyle=SUCCESS,
                                   command=lambda: self.set_mode("classification"))
        self.class_btn.grid(row=0, column=0, padx=3)

        self.seg_btn = tb.Button(self.mode_frame, text="Segmentation",
                                 bootstyle=SECONDARY,
                                 command=lambda: self.set_mode("segmentation"))
        self.seg_btn.grid(row=0, column=1, padx=3)

        # CAPTURE BUTTON
        self.capture_button = tb.Button(self.bottom_bar, text="Capture Photo (Space / C)",
                                        bootstyle=INFO, padding=10,
                                        command=self.capture_photo)
        self.capture_button.pack(side="left", padx=40)

        # FEEDBACK
        self.capture_feedback = tb.Label(self.root, text="", font=("Segoe UI", 12, "bold"), bootstyle=SUCCESS)
        self.capture_feedback.pack(side="bottom", pady=(0, 5))

        # THUMBNAILS
        self.gallery_frame = tb.Frame(self.root, padding=5)
        self.gallery_frame.pack(side="bottom", fill="x")

        tb.Label(self.gallery_frame, text="Recent Captures:", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.thumbnail_container = tb.Frame(self.gallery_frame)
        self.thumbnail_container.pack(anchor="w")

    # ============================================================
    # MODE SWITCH
    # ============================================================

    def set_mode(self, mode):
        self.mode.set(mode)
        if mode == "classification":
            self.class_btn.config(bootstyle=SUCCESS)
            self.seg_btn.config(bootstyle=SECONDARY)
        else:
            self.seg_btn.config(bootstyle=SUCCESS)
            self.class_btn.config(bootstyle=SECONDARY)

    # ============================================================
    # FRAME ACQUISITION
    # ============================================================

    def acquire_frame(self):
        if self.use_realsense:
            try:
                frames = self.pipeline.wait_for_frames()
                colour = frames.get_color_frame()
                if colour:
                    return np.asanyarray(colour.get_data())
            except Exception:
                pass

        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=8)
        return frame

    # ============================================================
    # INFO + METRICS UPDATE
    # ============================================================

    def update_info_panel(self, text):
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, text)

    def update_metrics(self, fps, latency_ms, frame):
        h, w = frame.shape[:2]
        src = "RealSense" if self.use_realsense else "Webcam"
        msg = (
            f"FPS: {fps:.1f}\n"
            f"Latency: {latency_ms:.2f} ms\n"
            f"Source: {src}\n"
            f"Resolution: {w} x {h}\n"
        )
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, msg)

    # ============================================================
    # PROCESS FRAME (dynamic info)
    # ============================================================

    def process_frame(self, frame):
        mode = self.mode.get()

        # =====================================================
        # SEGMENTATION MODE (dynamic class list)
        # =====================================================
        if mode == "segmentation":
            self.info_title.config(text="Segmentation Results")

            pred = self.seg_engine.process_frame(frame)

            # Overlay mask
            coloured = mask_to_colour(pred)
            coloured = cv2.resize(coloured, (frame.shape[1], frame.shape[0]), cv2.INTER_NEAREST)
            blended = cv2.addWeighted(frame, 1.0, coloured, 0.4, 0)

            info = ""

            # Count appearing classes
            unique, counts = np.unique(pred, return_counts=True)
            total = pred.size

            for cid, count in zip(unique, counts):
                if cid == 0:
                    continue  # ignore background
                name = CLASS_NAMES.get(cid, f"class {cid}")
                pct = (count / total) * 100
                info += f"- {name}: {pct:.2f}% coverage\n"

            # VIAL ANALYSIS
            vials = analyse_vials(pred, VIAL_VOLUME_ML)
            info += f"Detected vials: {len(vials)}\n\n"

            for i, vial in enumerate(vials, 1):
                info += f"Vial {i}\n"
                if not vial["layers"]:
                    info += "  Empty\n\n"
                else:
                    for layer in vial["layers"]:
                        cid = layer["class_id"]
                        name = CLASS_NAMES.get(cid, f"class {cid}")
                        pct = layer["percentage"] * 100
                        vol = layer["volume_ml"]
                        info += f"  - {name}: {pct:.1f}% ({vol:.2f} mL)\n"
                    info += "\n"

            self.last_info = info
            return blended

        # =====================================================
        # CLASSIFICATION MODE (dynamic probability listing)
        # =====================================================
        else:
            self.info_title.config(text="Classification Results")

            label, conf = self.cls_engine.classify(frame)

            # Draw on frame
            cv2.putText(frame, f"{label} ({conf:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            info = f"Top Prediction: {label} ({conf:.2f})\n\n"

            # Try to get full probabilities
            try:
                probs = self.cls_engine.predict_proba(frame)
            except:
                probs = None

            if probs is not None:
                for cid, p in enumerate(probs):
                    cname = CLASS_NAMES.get(cid, f"class {cid}")
                    info += f"- {cname}: {p:.2f}\n"

            self.last_info = info
            return frame

    # ============================================================
    # BACKGROUND WORKER LOOP
    # ============================================================

    def worker_loop(self):
        while self.thread_running:
            frame = self.acquire_frame()
            if frame is not None:
                start = time.time()

                processed = self.process_frame(frame)
                self.processed_frame = processed.copy()

                latency = (time.time() - start) * 1000
                fps = 1000 / latency if latency > 0 else 0
                self.metrics_info = (fps, latency, frame.copy())

    # ============================================================
    # UI UPDATE LOOP
    # ============================================================

    def update_frame(self):
        if self.processed_frame is not None:
            rgb = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2RGB)
            tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))

            self.video_label.imgtk = tkimg
            self.video_label.config(image=tkimg)

            self.update_info_panel(self.last_info)

            if hasattr(self, "metrics_info"):
                fps, latency, frame = self.metrics_info
                self.update_metrics(fps, latency, frame)

            self.last_frame = self.processed_frame.copy()

        self.root.after(10, self.update_frame)

    # ============================================================
    # CAPTURE + ANNOTATION
    # ============================================================

    def annotate_frame(self, frame):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        mode = self.mode.get()

        cv2.putText(frame, f"{mode.upper()} CAPTURE", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, ts, (frame.shape[1] - 250, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def capture_photo(self):
        if self.last_frame is None:
            return

        os.makedirs("photos", exist_ok=True)

        frame = self.annotate_frame(self.last_frame.copy())
        filename = time.strftime("photos/capture_%Y%m%d_%H%M%S.png")

        cv2.imwrite(filename, frame)
        self.add_thumbnail(filename)

        self.capture_feedback.config(text=f"Saved: {os.path.basename(filename)}")
        self.root.after(1000, lambda: self.capture_feedback.config(text=""))

    def capture_photo_event(self, event):
        self.capture_button.state(["pressed"])
        self.root.update_idletasks()
        self.capture_photo()
        self.root.after(150, lambda: self.capture_button.state(["!pressed"]))

    # ============================================================
    # THUMBNAIL GALLERY
    # ============================================================

    def add_thumbnail(self, filepath):
        img = Image.open(filepath)
        img.thumbnail((100, 100))
        tkimg = ImageTk.PhotoImage(img)

        thumb = tb.Label(self.thumbnail_container, image=tkimg)
        thumb.img = tkimg
        thumb.pack(side="left", padx=4)

        thumb.bind("<Button-1>", lambda e, p=filepath: self.open_image(p))

        if len(self.thumbnail_container.winfo_children()) > 5:
            old = self.thumbnail_container.winfo_children()[0]
            old.destroy()

    def open_image(self, path):
        win = tb.Toplevel(self.root)
        win.title(os.path.basename(path))

        img = Image.open(path)
        tkimg = ImageTk.PhotoImage(img)

        lbl = tb.Label(win, image=tkimg)
        lbl.img = tkimg
        lbl.pack()

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def on_close(self):
        self.thread_running = False
        self.root.destroy()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    root = tb.Window(themename="darkly")
    app = FrontEndApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
