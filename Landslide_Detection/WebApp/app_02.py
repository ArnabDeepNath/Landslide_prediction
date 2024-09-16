import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

# Define custom metrics (if used in your model)
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Load the saved model
model_path = 'path/to/your/model_save.h5'
model = tf.keras.models.load_model(model_path, custom_objects={
    'recall_m': recall_m,
    'precision_m': precision_m,
    'f1_m': f1_m
})

def process_image(filename):
    f_data = np.zeros((1, 128, 128, 6))
    with h5py.File(filename, 'r') as hdf:
        data = np.array(hdf.get('img'))
        
        data_red = data[:, :, 3]
        data_green = data[:, :, 2]
        data_blue = data[:, :, 1]
        data_nir = data[:, :, 7]
        
        # Avoid division by zero
        data_ndvi = np.divide(data_nir - data_red, data_nir + data_red, 
                              out=np.zeros_like(data_nir), where=(data_nir + data_red) != 0)
        
        f_data[0, :, :, 0] = data_ndvi
        f_data[0, :, :, 1] = data[:, :, 12]
        f_data[0, :, :, 2] = data[:, :, 13]
        f_data[0, :, :, 3] = data_red
        f_data[0, :, :, 4] = data_green
        f_data[0, :, :, 5] = data_blue

    return f_data, data[:, :, 3:0:-1]  # Return processed data and RGB image

def predict_and_display(image_path):
    # Process the image
    processed_data, rgb_image = process_image(image_path)
    
    # Make prediction
    raw_prediction = model.predict(processed_data)
    
    # Apply threshold
    threshold = 0.21
    prediction = (raw_prediction > threshold).astype(np.uint8)

    # Display results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax1.imshow(rgb_image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Raw prediction
    im2 = ax2.imshow(raw_prediction[0, :, :, 0], cmap='viridis')
    ax2.set_title('Raw Prediction')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)
    
    # Thresholded prediction
    im3 = ax3.imshow(prediction[0, :, :, 0], cmap='binary')
    ax3.set_title('Thresholded Prediction')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"Raw prediction range: [{np.min(raw_prediction)}, {np.max(raw_prediction)}]")
    print(f"Raw prediction mean: {np.mean(raw_prediction)}")
    print(f"Thresholded prediction mean: {np.mean(prediction)}")

# Use the function
image_path = 'image.h5'
predict_and_display(image_path)