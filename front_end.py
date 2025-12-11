import cv2
import numpy as np
import pyrealsense2 as rs
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import os

from live_segmentation import SegmentationEngine, mask_to_colour, analyse_vials, VIAL_VOLUME_ML
from live_classification import ClassificationEngine


class FrontEndApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Vial Analysis")
        self.root.bind("<space>", self.capture_photo_event)

        # Mode selector variable
        self.mode = tk.StringVar(value="classification")

        # Engines
        self.seg_engine = SegmentationEngine()
        self.cls_engine = ClassificationEngine()

        # Camera hardware selection
        self.use_realsense = False
        self.pipeline = None
        self.cap = None
        self.initialise_camera()

        # UI Construction
        self.build_ui()

        # Last image saved
        self.last_frame = None
        self.last_info = ""

        # Start updating display
        self.update_frame()

    # ------------------------------------------------------------
    # Camera initialisation
    # ------------------------------------------------------------
    def initialise_camera(self):
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
            self.use_realsense = True
            print("RealSense detected and initialised.")
        except Exception:
            print("RealSense not available. Falling back to webcam.")

            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise RuntimeError("No RealSense and no webcam available.")

            print("Webcam initialised.")
            self.use_realsense = False

    # ------------------------------------------------------------
    # Build the complete UI
    # ------------------------------------------------------------
    def build_ui(self):
        # Main area (video + information panel)
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        # Video feed widget
        self.video_label = tk.Label(self.main_frame)
        self.video_label.grid(row=0, column=0, padx=10, pady=10)

        # Information panel
        self.info_panel = tk.Frame(self.main_frame, width=300)
        self.info_panel.grid(row=0, column=1, sticky="n", padx=10, pady=10)

        tk.Label(self.info_panel, text="Information", font=("Arial", 14, "bold")).pack(anchor="nw")

        self.info_text = tk.Text(self.info_panel, width=40, height=20, font=("Consolas", 10))
        self.info_text.pack(anchor="nw", pady=10)
        
        # Metrics panel title
        tk.Label(self.info_panel, text="Metrics", font=("Arial", 12, "bold")).pack(anchor="nw", pady=(10, 0))

        # Metrics text box
        self.metrics_text = tk.Text(self.info_panel, width=40, height=6, font=("Consolas", 10))
        self.metrics_text.pack(anchor="nw")

        # Bottom bar – mode selector + capture button
        self.bottom_bar = tk.Frame(self.root)
        self.bottom_bar.pack(side="bottom", pady=10)

        # Mode label
        tk.Label(self.bottom_bar, text="Mode: ").pack(side="left", padx=(0, 10))

        # Frame to hold the two toggle buttons
        self.mode_frame = tk.Frame(self.bottom_bar)
        self.mode_frame.pack(side="left")

        # Shared button style
        self.mode_active_style = {
            "bg": "#4CAF50",
            "fg": "white",
            "relief": "sunken",
            "bd": 3,
            "font": ("Arial", 12, "bold"),
            "padx": 10,
            "pady": 5
        }

        self.mode_inactive_style = {
            "bg": "#E0E0E0",
            "fg": "black",
            "relief": "raised",
            "bd": 1,
            "font": ("Arial", 12),
            "padx": 10,
            "pady": 5
        }

        # Buttons
        self.class_btn = tk.Button(
            self.mode_frame, text="Classification",
            command=lambda: self.set_mode("classification")
        )
        self.seg_btn = tk.Button(
            self.mode_frame, text="Segmentation",
            command=lambda: self.set_mode("segmentation")
        )

        self.class_btn.grid(row=0, column=0)
        self.seg_btn.grid(row=0, column=1)

        # Initial visual state
        self.update_mode_buttons()

        # Capture button
        self.capture_button = tk.Button(
            self.bottom_bar,
            text="Capture Photo (Space)",
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            activebackground="#45A049",
            relief="raised",
            bd=4,
            padx=20,
            pady=10,
            command=self.capture_photo
        )
        self.capture_button.pack(side="left", padx=40)

        # Temporary capture feedback message
        self.capture_feedback = tk.Label(
            self.root,
            text="",
            font=("Arial", 12, "bold"),
            fg="green"
        )
        self.capture_feedback.pack(side="bottom", pady=(0, 5))

    # ------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------
    def acquire_frame(self):
        if self.use_realsense:
            try:
                frames = self.pipeline.wait_for_frames()
                colour_frame = frames.get_color_frame()
                if colour_frame:
                    return np.asanyarray(colour_frame.get_data())
            except Exception:
                return None

        # Webcam fallback
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Normalise webcam appearance
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

        return frame

    # ------------------------------------------------------------
    # Update information panel
    # ------------------------------------------------------------
    def update_info_panel(self, text):
        self.last_info = text
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, text)
   
    def update_metrics(self, fps, latency_ms, frame):
        source = "RealSense" if self.use_realsense else "Webcam"
        h, w = frame.shape[:2]

        text = (
            f"FPS: {fps:.1f}\n"
            f"Latency: {latency_ms:.2f} ms\n"
            f"Source: {source}\n"
            f"Resolution: {w} x {h}\n"
        )

        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, text)

    # ------------------------------------------------------------
    # Process frame depending on mode
    # ------------------------------------------------------------
    def process_frame(self, frame):
        mode = self.mode.get()

        if mode == "segmentation":
            pred_mask = self.seg_engine.process_frame(frame)
            coloured = mask_to_colour(pred_mask)
            coloured = cv2.resize(coloured, (frame.shape[1], frame.shape[0]), cv2.INTER_NEAREST)
            blended = cv2.addWeighted(frame, 1.0, coloured, 0.4, 0)

            # Analyse vials
            vials = analyse_vials(pred_mask, VIAL_VOLUME_ML)
            info = f"Segmentation Mode\n\nDetected vials: {len(vials)}\n\n"

            for i, vial in enumerate(vials, start=1):
                info += f"Vial {i}\n"
                if not vial["layers"]:
                    info += "  Empty\n\n"
                else:
                    for layer in vial["layers"]:
                        cid = layer["class_id"]
                        pct = layer["percentage"] * 100
                        vol = layer["volume_ml"]
                        info += f"  Class {cid}: {pct:.1f}% | {vol:.2f} mL\n"
                    info += "\n"

            self.update_info_panel(info)
            return blended

        # Classification mode
        label, conf = self.cls_engine.classify(frame)
        cv2.putText(
            frame, f"{label}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 255, 0) if label != "none" else (0, 0, 255),
            2
        )

        self.update_info_panel(
            f"Classification Mode\n\nPrediction: {label}\nConfidence: {conf:.2f}"
        )

        return frame

    # ------------------------------------------------------------
    # Display loop
    # ------------------------------------------------------------
    def update_frame(self):
        start_time = time.time()

        frame = self.acquire_frame()
        if frame is not None:

            processed = self.process_frame(frame)
            self.last_frame = processed.copy()

            # Display
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            tkimg = ImageTk.PhotoImage(image=Image.fromarray(rgb))

            self.video_label.imgtk = tkimg
            self.video_label.configure(image=tkimg)

            # Compute metrics
            latency_ms = (time.time() - start_time) * 1000
            fps = 1000 / latency_ms if latency_ms > 0 else 0

            # Update metrics panel
            self.update_metrics(fps, latency_ms, frame)

        self.root.after(1, self.update_frame)


    # ------------------------------------------------------------
    # Save photos (with auto-labelling filenames)
    # ------------------------------------------------------------
    def capture_photo(self):
        if self.last_frame is None:
            return

        if not os.path.exists("photos"):
            os.makedirs("photos")

        mode = self.mode.get()

        # Auto-name depending on mode
        if mode == "classification":
            parts = self.last_info.split("\n")
            label_line = parts[2] if len(parts) > 2 else "unknown"
            label = label_line.replace("Prediction: ", "").strip()
            safe_label = label.replace(" ", "_")
            filename = time.strftime(f"photos/class_{safe_label}_%Y%m%d_%H%M%S.png")

        else:  # segmentation
            parts = self.last_info.split("\n")
            count_line = parts[1] if len(parts) > 1 else "Detected vials: 0"
            n_vials = count_line.replace("Detected vials:", "").strip()
            filename = time.strftime(f"photos/seg_{n_vials}vials_%Y%m%d_%H%M%S.png")

        cv2.imwrite(filename, self.last_frame)

        # Show feedback
        self.capture_feedback.config(text=f"Photo saved: {os.path.basename(filename)}")
        self.root.after(1000, lambda: self.capture_feedback.config(text=""))

    def capture_photo_event(self, event):
        self.capture_button.config(relief="sunken")
        self.root.after(150, lambda: self.capture_button.config(relief="raised"))
        self.capture_photo()

    def set_mode(self, mode):
        self.mode.set(mode)
        self.update_mode_buttons()

    def update_mode_buttons(self):
        mode = self.mode.get()

        if mode == "classification":
            self.class_btn.config(**self.mode_active_style)
            self.seg_btn.config(**self.mode_inactive_style)
        else:
            self.seg_btn.config(**self.mode_active_style)
            self.class_btn.config(**self.mode_inactive_style)

# ----------------------------------------------------------
# Run
# ----------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FrontEndApp(root)
    root.mainloop()
