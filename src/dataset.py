import os

DATASET_DIR = "data"
SPLITS = ["train", "val", "test"]
VALID_EXT = (".jpg", ".jpeg", ".png")
COUNTER_DIGITS = 3

for split in SPLITS:
    images_dir = os.path.join(DATASET_DIR, split, "images")
    labels_dir = os.path.join(DATASET_DIR, split, "labels")

    if not os.path.exists(images_dir):
        continue

    images = sorted(f for f in os.listdir(images_dir)
                    if f.lower().endswith(VALID_EXT))

    for i, img_name in enumerate(images, start=1):
        ext = os.path.splitext(img_name)[1]
        new_name = f"{i:0{COUNTER_DIGITS}d}"

        old_img_path = os.path.join(images_dir, img_name)
        new_img_path = os.path.join(images_dir, new_name + ext)
        os.rename(old_img_path, new_img_path)

        old_label_path = os.path.join(labels_dir,
                                      os.path.splitext(img_name)[0] + ".txt")
        new_label_path = os.path.join(labels_dir, new_name + ".txt")

        if os.path.exists(old_label_path):
            os.rename(old_label_path, new_label_path)
