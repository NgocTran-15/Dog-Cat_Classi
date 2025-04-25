from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64
import os
import pickle

app = Flask(__name__)

# Define image size
IMG_SIZE = (128, 128)
# Confidence threshold for classification
CONFIDENCE_THRESHOLD = 0.85
# Number of Monte Carlo samples - reduced for better performance
MC_SAMPLES = 40  # Reduced from 50
# Variance threshold for OOD detection
VARIANCE_THRESHOLD = 0.1

# Enable GPU memory growth to prevent memory issues
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        print("Memory growth setting failed")

# Load model with better error handling
def load_model_safely():
    model_path = 'model_250425.pkl'
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        return None
        
    try:
        # Try loading with Keras first
        model = load_model(model_path)
        # Optimize model for inference
        model.make_predict_function()
        print("Model loaded successfully with Keras")
        return model
    except Exception as e:
        print(f"Error loading with Keras: {str(e)}")
        try:
            # If that fails, try loading with pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            # Optimize model for inference
            model.make_predict_function()
            print("Model loaded successfully with pickle")
            return model
        except Exception as e:
            print(f"Error loading with pickle: {str(e)}")
            return None

model = load_model_safely()

def monte_carlo_predictions(model, img_array):
    predictions = np.zeros(MC_SAMPLES)
    
    # Perform multiple forward passes with dropout
    for i in range(MC_SAMPLES):
        pred = model.predict(img_array, verbose=0)
        predictions[i] = pred[0][0]
    
    # Calculate mean and variance using numpy operations
    mean_pred = np.mean(predictions)
    var_pred = np.var(predictions)
    
    # Calculate entropy-based uncertainty
    uncertainty = -mean_pred * np.log2(mean_pred + 1e-10) - (1 - mean_pred) * np.log2(1 - mean_pred + 1e-10)
    
    return mean_pred, var_pred, uncertainty

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if model is None:
        return render_template('error.html', 
            message="Lỗi: Không thể tải model. Vui lòng kiểm tra file model và phiên bản TensorFlow/Keras.")
        
    if request.method == 'POST':
        try:
            # Read image from request
            file = request.files['file']
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            
            # Convert image to base64 for display
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Preprocess image
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            try:
                # Make Monte Carlo predictions
                mean_pred, var_pred, uncertainty = monte_carlo_predictions(model, img_array)
                
                # Enhanced OOD detection using both variance and uncertainty
                is_ood = (var_pred > VARIANCE_THRESHOLD) or (uncertainty > 0.8) or \
                        (mean_pred > 0.4 and mean_pred < 0.6)
                
                # Determine result based on OOD detection and mean prediction
                if is_ood:
                    label = 'Không phải chó/mèo'
                    # Calculate confidence based on how far from decision boundary
                    confidence = 1.0 - (2 * abs(mean_pred - 0.5))
                else:
                    if mean_pred >= 0.6:
                        label = 'Chó'
                        confidence = mean_pred
                    elif mean_pred <= 0.4:
                        label = 'Mèo'
                        confidence = 1 - mean_pred
                    else:
                        label = 'Không phải chó/mèo'
                        confidence = 1.0 - (2 * abs(mean_pred - 0.5))
                
                confidence_percent = round(max(min(confidence * 100, 100), 0), 2)
                
            except Exception as e:
                print(f"Monte Carlo prediction failed: {str(e)}")
                # Fallback to single prediction if Monte Carlo fails
                prediction = model.predict(img_array, verbose=0)[0][0]
                
                # Enhanced single prediction OOD detection
                if 0.4 <= prediction <= 0.6:
                    label = 'Không phải chó/mèo'
                    confidence = 1.0 - (2 * abs(prediction - 0.5))
                elif prediction > 0.6:
                    label = 'Chó'
                    confidence = prediction
                else:
                    label = 'Mèo'
                    confidence = 1 - prediction
                
                confidence_percent = round(max(min(confidence * 100, 100), 0), 2)
            
            return render_template('result.html', 
                                prediction=label, 
                                confidence=confidence_percent,
                                image_url=f"data:image/jpeg;base64,{img_base64}")
        except Exception as e:
            return render_template('error.html', message=f"Lỗi xử lý ảnh: {str(e)}")
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)