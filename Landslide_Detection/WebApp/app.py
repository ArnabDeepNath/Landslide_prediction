# app.py
from flask import Flask, request, render_template, jsonify, send_file
import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras import backend as K
import os
import matplotlib.pyplot as plt
import io
import base64
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the metrics functions
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

# Load the model
try:
    model = tf.keras.models.load_model('model_save.h5', custom_objects={
        'recall_m': recall_m,
        'precision_m': precision_m,
        'f1_m': f1_m
    })
    logging.info(f"Model summary: {model.summary()}")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

def generate_plots(prediction, rgb_image):
    try:
        def plot_to_base64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        # Prediction plot
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        im1 = ax1.imshow(prediction[0, :, :, 0], cmap='viridis')
        ax1.set_title("Thresholded Predictions")
        ax1.axis('off')
        fig1.colorbar(im1)
        prediction_plot = plot_to_base64(fig1)
        plt.close(fig1)

        # Original image plot
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.imshow(rgb_image)
        ax2.set_title('Original Image')
        ax2.axis('off')
        original_plot = plot_to_base64(fig2)
        plt.close(fig2)

        logging.info("Plots generated successfully")
        return prediction_plot, original_plot
    except Exception as e:
        logging.error(f"Error generating plots: {str(e)}")
        return None, None

def process_file(filename):
    try:
        f_data = np.zeros((1, 128, 128, 6))
        with h5py.File(filename, 'r') as hdf:
            data = np.array(hdf.get('img'))

            # Replace NaN values
            data[np.isnan(data)] = 0.000001

            # Compute midpoints for normalization
            mid_rgb = data[:, :, 1:4].max() / 2.0
            mid_slope = data[:, :, 12].max() / 2.0
            mid_elevation = data[:, :, 13].max() / 2.0

            # Extract RGB and other bands
            data_red = data[:, :, 3]
            data_green = data[:, :, 2]
            data_blue = data[:, :, 1]
            data_nir = data[:, :, 7]

            # NDVI calculation
            data_ndvi = np.divide(data_nir - data_red, data_nir + data_red, 
                                  out=np.zeros_like(data_nir), where=(data_nir + data_red) != 0)

            # Populate final processed data with normalization
            f_data[0, :, :, 0] = data_ndvi  # NDVI
            f_data[0, :, :, 1] = 1 - data[:, :, 12] / mid_slope  # Slope (normalized)
            f_data[0, :, :, 2] = 1 - data[:, :, 13] / mid_elevation  # Elevation (normalized)
            f_data[0, :, :, 3] = 1 - data_red / mid_rgb  # Red (normalized)
            f_data[0, :, :, 4] = 1 - data_green / mid_rgb  # Green (normalized)
            f_data[0, :, :, 5] = 1 - data_blue / mid_rgb  # Blue (normalized)

        # Log input data statistics
        logging.info(f"Input data shape: {f_data.shape}")
        logging.info(f"Input data range: [{np.min(f_data)}, {np.max(f_data)}]")
        logging.info(f"Input data mean: {np.mean(f_data)}")
        logging.info(f"Input data std: {np.std(f_data)}")

        # Make prediction
        raw_prediction = model.predict(f_data)
        
        # Log raw prediction statistics
        logging.info(f"Raw prediction shape: {raw_prediction.shape}")
        logging.info(f"Raw prediction range: [{np.min(raw_prediction)}, {np.max(raw_prediction)}]")
        logging.info(f"Raw prediction mean: {np.mean(raw_prediction)}")
        logging.info(f"Raw prediction std: {np.std(raw_prediction)}")

        # Apply threshold
        threshold = 0.0021
        prediction = (raw_prediction > threshold).astype(np.uint8)
        
        # Log thresholded prediction statistics
        logging.info(f"Thresholded prediction range: [{np.min(prediction)}, {np.max(prediction)}]")
        logging.info(f"Thresholded prediction mean: {np.mean(prediction)}")

        rgb_image = f_data[0, :, :, 3:6]  # Extract normalized RGB channels
        logging.info("File processed successfully")
        return prediction, rgb_image
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return None, None


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and file.filename.endswith('.h5'):
            try:
                filename = file.filename
                file.save(filename)
                prediction, rgb_image = process_file(filename)
                os.remove(filename)  # Remove the file after processing
                
                if prediction is None or rgb_image is None:
                    return jsonify({'error': 'Error processing file'}), 500
                
                # Generate plots
                prediction_plot, original_plot = generate_plots(prediction, rgb_image)
                
                if prediction_plot is None or original_plot is None:
                    return jsonify({'error': 'Error generating plots'}), 500
                
                return jsonify({
                    'prediction': prediction.tolist(),
                    'prediction_plot': prediction_plot,
                    'original_plot': original_plot
                })
            except Exception as e:
                logging.error(f"Error in upload_file: {str(e)}")
                return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
