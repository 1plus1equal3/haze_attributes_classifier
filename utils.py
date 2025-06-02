import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import shutil
import os

# Visualization utility for color clusters
COLOR_CLUSTERS = {
    1: [
        (246.48, 254.53),
        (213.99, 251.55),
        (149.79, 239.74)
    ],
    2: [
        (146.16, 217.47),
        (181.47, 230.52),
        (205.76, 241.87)
    ],
    3: [
        (218.99, 255.0),
        (217.01, 255.0),
        (215.25, 255.0)
    ]
}


# Initialize densities
HAZE_DENSITIES = {
    0: (0.01, 0.075), # Light
    1: (0.075, 0.25),  # Medium
    2: (0.25, 0.5)    # Heavy
}

def viz_cluster_color_range(cluster_id='0', step=5):
    cluster = COLOR_CLUSTERS[cluster_id] # Get cluster
    # Get value range for each color channel
    red_range = np.linspace(cluster[0][0], cluster[0][1], step)
    green_range = np.linspace(cluster[1][0], cluster[1][1], step)
    blue_range = np.linspace(cluster[2][0], cluster[2][1], step)
    # Get color list
    colors = []
    for i in range(step):
        colors.append((
            int(red_range[i]),
            int(green_range[i]),
            int(blue_range[i])
        ))
    f, ax = plt.subplots(1, step, figsize=(10,10), tight_layout=True)
    for i in range(step):
        color_arr = np.ones((100, 100, 3), dtype=np.uint8) * np.asarray(colors[i])
        ax[i].imshow(color_arr)
        ax[i].axis('off')
        ax[i].set_title(str(colors[i]))
    plt.show()

# Sort images by color and density
def sort_result(predictions, save_dir='./results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    new_predictions = []

    # Create directories for each haze color and density
    for color in COLOR_CLUSTERS.keys():
        color_dir = os.path.join(save_dir, 'color', f'color_{color}')
        if not os.path.exists(color_dir):
            os.makedirs(color_dir)
    for density in HAZE_DENSITIES.keys():
        density_dir = os.path.join(save_dir, 'density', f'density_{density}')
        if not os.path.exists(density_dir):
            os.makedirs(density_dir)

    index = 0
    for prediction in tqdm(predictions):
        image_path = prediction['image_path']
        haze_color = prediction['haze_color']
        haze_density = prediction['haze_density']
        
        # Copy the image to the appropriate directory
        color_dest_path = os.path.join(save_dir, 'color', f'color_{haze_color}', f'{index:06d}.jpg')
        os.makedirs(os.path.dirname(color_dest_path), exist_ok=True)
        shutil.copy(image_path, color_dest_path)
        
        density_dest_path = os.path.join(save_dir, 'density', f'density_{haze_density}', f'{index:06d}.jpg')
        os.makedirs(os.path.dirname(density_dest_path), exist_ok=True)
        shutil.copy(image_path, density_dest_path)
        index += 1

        # Append to new predictions list
        new_predictions.append({
            'image_path': image_path,
            'haze_color': haze_color,
            'haze_density': haze_density,
            'color_dest_path': color_dest_path,
            'density_dest_path': density_dest_path
        })

    # Save new predictions to CSV
    new_predictions_df = pd.DataFrame(new_predictions)
    new_predictions_df.to_csv(os.path.join(save_dir, 'haze_attributes_sorted.csv'), index=False)
    print(f"Results saved to {save_dir} with sorted images by haze color and density.")