# Inference configuration
ckpt_path = './model/haze_classifier_tuned_v1.keras'
BATCH_SIZE = 64
IMAGE_PATHS = '/home/duynd/haze_attrs/Haze_attributes/0/*.*'
OUTPUT_FILE = 'haze_attributes_0.csv'
ERROR_FILE = 'error_images.txt'

#! OPTIONAL: Sort images by color and density
SORT_IMAGES = False
SORT_DIR = './results'