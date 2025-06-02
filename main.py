import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import glob
from model import load_model, infer
from utils import sort_result
from config import ckpt_path, BATCH_SIZE, IMAGE_PATHS, OUTPUT_FILE, ERROR_FILE, SORT_IMAGES, SORT_DIR

# Data pipeline utils
error_images = []

@tf.py_function(Tout=(tf.string, tf.float32))
def load_image(file_path):
    try:
        img = Image.open(file_path.numpy().decode()).convert('RGB')
        img = img.resize((256, 256))
        img = tf.convert_to_tensor(np.array(img), dtype=tf.float32) / 255.0
        return file_path, img
    except Exception as e:
        # Return dummy tensor and mark with empty string path
        error_images.append(file_path.numpy().decode())
        return tf.constant('', dtype=tf.string), tf.zeros([256, 256, 3], dtype=tf.float32)

def filter_invalid(file_path, image):
    return tf.strings.length(file_path) > 0

def build_dataset(image_paths, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_image, num_parallel_calls=4)
    dataset = dataset.filter(filter_invalid)
    dataset = dataset.batch(batch_size).prefetch(4)
    return dataset

# Load model from checkpoint
model = load_model(ckpt_path)
if model is None:
    raise RuntimeError("Failed to load model. Check the checkpoint path.")
else: model.summary()

# Build dataset from image paths
image_paths = glob.glob(IMAGE_PATHS)
print(f"Found {len(image_paths)} images.")
dataset = build_dataset(image_paths, BATCH_SIZE)

# Run inference
predictions = infer(model, dataset)
# Save predictions to CSV
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv(OUTPUT_FILE, index=False)
print(f"Results saved to {OUTPUT_FILE}.")
# Save error images if any
if len(error_images) > 0:
    print(f"Found {len(error_images)} error images during processing.")
    with open(ERROR_FILE, 'w') as f:
        for img in error_images:
            f.write(f"{img}\n")
    print(f"Error images saved to {ERROR_FILE}")

# OPTIONAL: Sort images by color and density
if SORT_IMAGES:
    sort_result(predictions, SORT_DIR)