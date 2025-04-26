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
# Điều chỉnh các threshold để cân bằng giữa OOD và classification
CONFIDENCE_THRESHOLD = 0.65  # Giảm để dễ phân loại hơn
VARIANCE_THRESHOLD = 0.15    # Tăng để phát hiện OOD tốt hơn
ENTROPY_THRESHOLD = 0.65     # Cân bằng giữa nhạy và đặc hiệu
MC_SAMPLES = 50             # Giữ nguyên số lượng samples

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
    predictions = np.zeros((MC_SAMPLES, 1))
    features = np.zeros((MC_SAMPLES, 512))  # Lấy feature từ layer cuối cùng
    
    for i in range(MC_SAMPLES):
        pred = model.predict(img_array, verbose=0)
        predictions[i] = pred[0][0]
        
        # Lấy features từ layer Dense cuối để phát hiện OOD
        feature_model = tf.keras.Model(model.input, model.layers[-2].output)
        features[i] = feature_model.predict(img_array, verbose=0)
    
    mean_pred = np.mean(predictions)
    var_pred = np.var(predictions)
    
    # Tính toán các metrics mới cho OOD
    feature_variance = np.mean(np.var(features, axis=0))
    prediction_range = np.max(predictions) - np.min(predictions)
    entropy = -mean_pred * np.log2(mean_pred + 1e-10) - (1 - mean_pred) * np.log2(1 - mean_pred + 1e-10)
    
    
    confidence_score = (1 - var_pred) * (1 - feature_variance) * (1 - entropy)
    
    # Điều kiện phân loại được tinh chỉnh
    is_dog = mean_pred > 0.65 and confidence_score > CONFIDENCE_THRESHOLD
    is_cat = mean_pred < 0.35 and confidence_score > CONFIDENCE_THRESHOLD
    
    # OOD detection với nhiều tiêu chí
    is_ood = (
        (0.35 <= mean_pred <= 0.65) or            # Vùng không chắc chắn
        (feature_variance > VARIANCE_THRESHOLD) or # Feature variance cao
        (prediction_range > 0.4) or               # Range dự đoán lớn
        (confidence_score < CONFIDENCE_THRESHOLD)  # Confidence thấp
    )
    
    return mean_pred, confidence_score, is_dog, is_cat, is_ood

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if model is None:
        return render_template('error.html', 
            message="Cannot load model. Please check the model file and TensorFlow/Keras version.")
        
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
                mean_pred, confidence_score, is_dog, is_cat, is_ood = monte_carlo_predictions(model, img_array)
                
                if is_ood:
                    label = 'Không phải chó/mèo'
                    confidence = confidence_score * 100
                elif is_dog:
                    label = 'Chó'
                    confidence = mean_pred * confidence_score * 100
                elif is_cat:
                    label = 'Mèo'
                    confidence = (1 - mean_pred) * confidence_score * 100
                else:
                    label = 'Không phải chó/mèo'
                    confidence = confidence_score * 100
                
                confidence_percent = round(max(min(confidence, 100), 0), 2)
                
            except Exception as e:
                print(f"Monte Carlo prediction failed: {str(e)}")
                # Fallback với single prediction được cải thiện
                prediction = model.predict(img_array, verbose=0)[0][0]
                
                # Thêm kiểm tra OOD cho single prediction
                if 0.35 <= prediction <= 0.65:
                    label = 'Không phải chó/mèo'
                    confidence = (1 - 2 * abs(prediction - 0.5)) * 100
                elif prediction > 0.65:
                    label = 'Chó'
                    confidence = prediction * 100
                else:
                    label = 'Mèo'
                    confidence = (1 - prediction) * 100
                
                confidence_percent = round(max(min(confidence, 100), 0), 2)
            
            return render_template('result.html', 
                                prediction=label, 
                                confidence=confidence_percent,
                                image_url=f"data:image/jpeg;base64,{img_base64}")
        except Exception as e:
            return render_template('error.html', message=f"Lỗi xử lý ảnh: {str(e)}")
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)