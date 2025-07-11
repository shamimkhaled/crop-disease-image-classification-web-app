import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from django.contrib import messages
from .recommendations import recommendations  
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf

# Global model variable
model = None
model_loaded = False

class_names = {
    0: 'Corn___Common_Rust', 1: 'Corn___Gray_Leaf_Spot',
    2: 'Corn___Healthy', 3: 'Corn___Leaf_Blight', 
    4: 'Potato___Early_Blight', 5: 'Potato___Healthy', 
    6: 'Potato___Late_Blight', 7: 'Rice___Brown_Spot', 
    8: 'Rice___Healthy', 9: 'Rice___Hispa', 10: 'Rice___Leaf_Blast', 
    11: 'Wheat___Brown_Rust', 12: 'Wheat___Healthy', 13: 'Wheat___Yellow_Rust'
}

def load_model_safely():
    """Safely load the model with error handling"""
    global model, model_loaded
    
    if model_loaded:
        return model is not None
    
    # Try different model files in order of preference
    model_files = [
        'model/model_InceptionV3.h5',
        'model/model_DenseNet121.h5', 
        'model/model_MobileNetV2.h5'
    ]
    
    for model_file in model_files:
        model_path = os.path.join(settings.BASE_DIR, model_file)
        
        if os.path.exists(model_path):
            try:
                # Check if file is actually a valid model file (not LFS pointer)
                file_size = os.path.getsize(model_path)
                if file_size < 1000:  # LFS pointer files are very small
                    print(f"Warning: {model_file} appears to be a Git LFS pointer file")
                    continue
                    
                print(f"Loading model from {model_file}...")
                model = tf.keras.models.load_model(model_path)
                model_loaded = True
                print(f"✓ Model loaded successfully from {model_file}")
                return True
                
            except Exception as e:
                print(f"✗ Failed to load {model_file}: {e}")
                continue
    
    model_loaded = True
    print("✗ No valid model found. Please check your model files.")
    return False

def create_demo_prediction(image_path):
    """Create a demo prediction when no model is available"""
    # Analyze filename or use random prediction for demo
    filename = os.path.basename(image_path).lower()
    
    # Simple keyword-based demo prediction
    if any(word in filename for word in ['corn', 'maize']):
        if any(word in filename for word in ['rust', 'disease', 'sick']):
            predicted_class = 'Corn___Common_Rust'
            confidence = 85.5
        else:
            predicted_class = 'Corn___Healthy'
            confidence = 92.3
    elif any(word in filename for word in ['potato']):
        if any(word in filename for word in ['blight', 'disease', 'sick']):
            predicted_class = 'Potato___Early_Blight'
            confidence = 78.9
        else:
            predicted_class = 'Potato___Healthy'
            confidence = 89.2
    elif any(word in filename for word in ['rice']):
        if any(word in filename for word in ['blast', 'disease', 'sick']):
            predicted_class = 'Rice___Leaf_Blast'
            confidence = 82.1
        else:
            predicted_class = 'Rice___Healthy'
            confidence = 94.6
    elif any(word in filename for word in ['wheat']):
        if any(word in filename for word in ['rust', 'disease', 'sick']):
            predicted_class = 'Wheat___Brown_Rust'
            confidence = 76.8
        else:
            predicted_class = 'Wheat___Healthy'
            confidence = 91.4
    else:
        # Default demo prediction
        predicted_class = 'Corn___Healthy'
        confidence = 88.5
    
    return predicted_class, confidence

def home(request):
    """Main view for home page and prediction"""
    
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        
        # Validate file type
        if not uploaded_file.content_type.startswith('image/'):
            messages.error(request, 'Please upload a valid image file.')
            return render(request, 'home.html')
        
        # Save uploaded file
        try:
            # Create unique filename to avoid conflicts
            import time
            timestamp = str(int(time.time()))
            file_extension = os.path.splitext(uploaded_file.name)[1]
            unique_filename = f"crop_image_{timestamp}{file_extension}"
            
            img_path = default_storage.save(unique_filename, uploaded_file)
            full_img_path = os.path.join(settings.MEDIA_ROOT, img_path)
            
            # Ensure media directory exists
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            
            # Try to load and process image
            img = Image.open(full_img_path)
            img = img.convert('RGB')  # Ensure RGB format
            img = img.resize((224, 224))
            
            print(f"Image saved to: {full_img_path}")
            print(f"Image URL will be: {settings.MEDIA_URL}{img_path}")
            
        except Exception as e:
            messages.error(request, f'Error processing image: {e}')
            return render(request, 'home.html')
        
        # Try to make prediction with model
        if load_model_safely() and model is not None:
            try:
                # Preprocess image for model
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # Normalize
                
                # Make prediction
                predictions = model.predict(img_array, verbose=0)
                score = predictions[0]
                
                predicted_class = class_names[np.argmax(score)]
                confidence = 100 * np.max(score)
                
                print(f"✓ Model prediction: {predicted_class} ({confidence:.2f}%)")
                
            except Exception as e:
                print(f"✗ Error during prediction: {e}")
                # Fallback to demo prediction
                predicted_class, confidence = create_demo_prediction(full_img_path)
                messages.warning(request, 'Using demo prediction due to model error.')
        else:
            # Use demo prediction when no model is available
            predicted_class, confidence = create_demo_prediction(full_img_path)
            messages.info(request, 'Demo mode: Using simulated predictions. Please load a trained model for accurate results.')
        
        # Format confidence
        confidence_format = '{:.2f}'.format(confidence)
        
        # Get recommendations
        recommendation = recommendations.get(predicted_class, {
            'solution': 'No specific recommendation available for this classification.',
            'prevention': 'Follow general crop care practices.',
        })
        
        # Prepare context for template
        # Fix the image URL - just use the filename with MEDIA_URL
        image_url = f"{settings.MEDIA_URL}{img_path}"
        
        # Ensure no double slashes
        image_url = image_url.replace('//', '/')
        
        context = {
            'predicted_class': predicted_class.replace('___', ' - ').replace('_', ' '),
            'confidence_format': confidence_format,
            'image_url': image_url,
            'image_path': img_path,  # Add this for debugging
            'solution': recommendation.get('solution', 'No solution available.'),
            'prevention': recommendation.get('prevention', 'No prevention measures available.'),
            'disease': recommendation.get('disease', ''),
            'text': recommendation.get('text', ''),
            'medicine_products': recommendation.get('medicine_products', [])
        }
        
        print(f"Image path: {img_path}")
        print(f"Final image URL: {image_url}")  # Debug print
        print(f"Media root: {settings.MEDIA_ROOT}")
        print(f"Media URL: {settings.MEDIA_URL}")
        
        return render(request, 'prediction.html', context)
    
    return render(request, 'home.html')