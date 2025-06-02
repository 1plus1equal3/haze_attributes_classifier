import tensorflow as tf
import keras
import numpy as np
from tqdm import tqdm


# Load model from checkpoint
def load_model(ckpt):
    try:
        model = keras.models.load_model(ckpt)
        print(f"Model loaded successfully from {ckpt}")
        # model.summary()
        return model
    except Exception as e:
        print(f"Error loading model from {ckpt}: {e}")
        return None
    
def infer(model, dataset):
    predictions = []
    for batch in tqdm(dataset):
        paths, imgs = batch # Unpack paths and images
        # Inference
        preds = model.predict(imgs, verbose=0)
        # Process predictions
        for i in range(len(paths)):
            haze_color = np.argmax(preds[0][i])
            haze_density = np.argmax(preds[1][i])
            predictions.append({
                'image_path': paths[i].numpy().decode(),
                'haze_color': haze_color,
                'haze_density': haze_density,
            })
        del paths, imgs, preds # Clear memory
    return predictions