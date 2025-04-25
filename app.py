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
CONFIDENCE_THRESHOLD = 0.7

# Load model with better error handling
def load_model_safely():
    model_path = 'model_2504.pkl'
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        return None
        
    try:
        # Try loading with Keras first
        model = load_model(model_path)
        print("Model loaded successfully with Keras")
        return model
    except Exception as e:
        print(f"Error loading with Keras: {str(e)}")
        try:
            # If that fails, try loading with pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded successfully with pickle")
            return model
        except Exception as e:
            print(f"Error loading with pickle: {str(e)}")
            return None

model = load_model_safely()

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
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Preprocess image
            img = img.resize(IMG_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = model.predict(img_array)[0][0]
            
            # Determine result based on confidence threshold
            if prediction > 0.5:
                confidence = prediction
                if confidence > CONFIDENCE_THRESHOLD:
                    label = "Chó"
                else:
                    label = "Không phải chó/mèo"
            else:
                confidence = 1 - prediction
                if confidence > CONFIDENCE_THRESHOLD:
                    label = "Mèo"
                else:
                    label = "Không phải chó/mèo"
            
            confidence_percent = round(confidence * 100, 2)
            
            return render_template('result.html', 
                                prediction=label, 
                                confidence=confidence_percent,
                                image_url=f"data:image/jpeg;base64,{img_base64}")
        except Exception as e:
            return render_template('error.html', message=f"Lỗi xử lý ảnh: {str(e)}")
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)