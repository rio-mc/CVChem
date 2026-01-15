import pyrealsense2 as rs
import numpy as np
import cv2
import time

# ---------- User settings ----------
ACTIVATION_KEY = 'a'         # Start automatic saving when pressed
SAVE_KEY       = 's'         # Save an image
SAVE_INTERVAL  = 2           # Seconds between automatic saves
MAX_IMAGES     = 10          # Maximum saved images

AUTO_PREFIX    = "test_"       # Default output filename prefix (can change during stream)
MANUAL_PREFIX  = "image_"     # Manual save prefix
# -----------------------------------

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

print("Streaming...")
print("  Press 'a' to start automatic saving")
print("  Press 'p' to change output filename prefix")
print("  Press 's' to manually save a frame")
print("  Press 'q' to quit")

auto_enabled = False
auto_count = 1
manual_count = 1
last_auto_time = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # Show live stream
        cv2.imshow("RealSense Color Stream", color_image)

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):
            break

        # Save an image
        if key == ord(SAVE_KEY):
            filename = f"{MANUAL_PREFIX}{manual_count}.png"
            cv2.imwrite(filename, color_image)
            print(f"[MANUAL] Saved {filename}")
            manual_count += 1

        # Enable auto-saving
        if not auto_enabled and key == ord(ACTIVATION_KEY):
            auto_enabled = True
            last_auto_time = time.time()
            print(f"[INFO] Capturing {MAX_IMAGES} images every {SAVE_INTERVAL} seconds...")

        # Change prefix during streaming
        if key == ord('p'):
            new_prefix = input("Enter new auto-save prefix: ").strip()
            if new_prefix != "":
                AUTO_PREFIX = new_prefix
                auto_count = 1  # optional reset to default prefix
                print(f"[INFO] Filename prefix changed to '{AUTO_PREFIX}'", flush=True)
            else:
                print(f"[INFO] Prefix unchanged: {AUTO_PREFIX}")

        # Perform automatic saving after pressing activation key
        if auto_enabled and auto_count <= MAX_IMAGES:
            current_time = time.time()
            if current_time - last_auto_time >= SAVE_INTERVAL:
                filename = f"{AUTO_PREFIX}{auto_count}.png"
                cv2.imwrite(filename, color_image)
                print(f"[AUTO] Saved {filename} ({auto_count}/{MAX_IMAGES})")

                auto_count += 1
                last_auto_time = current_time

        # Check if automatic saving finished
        if auto_enabled and auto_count > MAX_IMAGES:
            print("[INFO] Completed capture.")
            auto_enabled = False


finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera stopped.")
