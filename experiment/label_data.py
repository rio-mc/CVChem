import os
import csv

# Label data for training based on mask classes
data_dir = "data/"
classes_file = "train/_classes.csv"
output_file = "labels.txt"

# ---------------------------
# Load class mapping from CSV
# ---------------------------
class_mapping = {}

with open(classes_file, "r", newline="") as f:
    reader = csv.DictReader(f)  # uses column names Pixel Value, Class
    for row in reader:
        class_name = row["Class"].strip()
        pixel_value = int(row["Pixel Value"].strip())
        class_mapping[class_name] = pixel_value

print("[INFO] Loaded class mapping:", class_mapping)

# ---------------------------
# Process data directory
# ---------------------------
files = [
    f for f in os.listdir(data_dir)
    if os.path.isfile(os.path.join(data_dir, f))
]

with open(output_file, "w") as f_out:
    for file_name in files:
        class_index = -1

        # Match filenames based on prefix
        for prefix, idx in class_mapping.items():
            if file_name.startswith(prefix):
                class_index = idx
                break

        if class_index == -1:
            continue  # skip files without valid prefix

        f_out.write(f"{file_name},{class_index}\n")

print(f"[INFO] Generated '{output_file}'.")

