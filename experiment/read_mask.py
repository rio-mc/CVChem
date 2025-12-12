import os
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

# Given original images and their masks, make the masks human-viewable
# -------------------------------------------------------
def load_class_map(csv_path):
    pixel_values = []
    class_names = []

    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pv = int(row["Pixel Value"])
            cl = row["Class"]
            pixel_values.append(pv)
            class_names.append(cl)

    # Use Matplotlib colormap API
    cmap = mpl.colormaps["tab10"]

    class_map = {}
    for i, pv in enumerate(pixel_values):
        r, g, b, _ = cmap(i % 10)[:4]
        rgb = (int(r*255), int(g*255), int(b*255))
        class_map[pv] = rgb

    return class_map, dict(zip(pixel_values, class_names))

# -------------------------------------------------------
def colorize_mask(mask_array, color_map):
    h, w = mask_array.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for pv, rgb in color_map.items():
        color_mask[mask_array == pv] = rgb

    return color_mask

# -------------------------------------------------------
def generate_visualizations(folder):

    csv_path = os.path.join(folder, "_classes.csv")
    if not os.path.exists(csv_path):
        print("❌ _classes.csv not found.")
        return

    color_map, class_labels = load_class_map(csv_path)

    output_dir = os.path.join(folder, "visualisations")
    os.makedirs(output_dir, exist_ok=True)

    valid_ext = (".jpg", ".jpeg", ".png")

    # originals = .jpg/.jpeg/.png, but NOT mask files
    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith(valid_ext) and "_mask" not in f.lower()
    ]

    if len(images) == 0:
        print("❌ No original images found.")
        return

    for img_file in images:
        print(f"Processing: {img_file}")

        img_path = os.path.join(folder, img_file)

        # Mask must ALWAYS be PNG
        base = os.path.splitext(img_file)[0]
        mask_path = os.path.join(folder, base + "_mask.png")

        if not os.path.exists(mask_path):
            print(f"⚠️ No mask for {img_file}, skipping.")
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        mask_color = colorize_mask(mask, color_map)

        # Visualization
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image)
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(mask_color)
        ax[1].set_title("Mask (Colored)")
        ax[1].axis("off")

        out_path = os.path.join(output_dir, base + "_vis.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"✔ Saved: {out_path}")

# Run
generate_visualizations("annotated_data/")

