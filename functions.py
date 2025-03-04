import os
import pickle
import numpy as np
import cv2  # OpenCV for image processing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy import stats  # For outlier detection

# Get absolute path of the project directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Load Deep Learning Model
def get_model(path):
    """ Load a pre-trained Keras model from a given path. """
    try:
        model = load_model(path, compile=False)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# **Noisy Image Preprocessing**
def preprocess_image(image_path):
    """ Load, resize, denoise, and normalize an image for model prediction. """
    try:
        # Load image and convert to array
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)

        # Convert image to OpenCV format (for noise removal)
        image_cv = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2BGR)

        # Apply Gaussian Blur (reduces noise)
        image_cv = cv2.GaussianBlur(image_cv, (5, 5), 0)

        # Convert back to Keras format & normalize
        image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image = img_to_array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return None

# **Image Prediction**
def img_predict(path, crop):
    """ Predict the disease class from an image using a trained model. """
    try:
        image = preprocess_image(path)
        if image is None:
            return None

        model_path = os.path.join(BASE_DIR, 'models', 'DL_models', f'{crop}_model.h5')
        model = get_model(model_path)

        if model:
            prediction = model.predict(image)[0]
            predicted_index = np.argmax(prediction) if len(crop_diseases_classes[crop]) > 2 else int(np.round(prediction[0]))
            return predicted_index
        else:
            return None
    except Exception as e:
        print(f"Error in image prediction: {e}")
        return None

# **Get Disease Class from Prediction**
def get_diseases_classes(crop, prediction):
    """ Get the disease class based on model's prediction index. """
    try:
        crop_classes = crop_diseases_classes[crop]
        return crop_classes[prediction][1].replace("_", " ")
    except Exception as e:
        print(f"Error fetching disease class: {e}")
        return "Unknown"

# **Handle Noisy Data (Missing Values & Scaling)**
def clean_data(data):
    """ Clean noisy data by handling missing values and outliers. """
    try:
        # Convert to NumPy array
        data = np.array(data, dtype=np.float64)

        # **1. Handle Missing Values**
        imputer = SimpleImputer(strategy="mean")  # Fill NaN with mean
        data = imputer.fit_transform(data.reshape(-1, 1)).flatten()

        # **2. Detect & Remove Outliers using Z-score**
        z_scores = np.abs(stats.zscore(data))
        data = data[z_scores < 3]  # Remove values beyond 3 std deviations

        # **3. Normalize Data (Min-Max Scaling)**
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

        return data
    except Exception as e:
        print(f"Error in data cleaning: {e}")
        return None

# **Crop Recommendation (with Noisy Data Handling)**
def get_crop_recommendation(item):
    """ Predict the best crop based on soil and environmental conditions with noise handling. """
    try:
        item = clean_data(item)  # Clean noisy input data

        scaler_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'crop_scaler.pkl')
        model_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'crop_model.pkl')

        with open(scaler_path, 'rb') as f:
            crop_scaler = pickle.load(f)
        with open(model_path, 'rb') as f:
            crop_model = pickle.load(f)

        scaled_item = crop_scaler.transform([item])
        prediction = crop_model.predict(scaled_item)[0]
        return crops[prediction]
    except Exception as e:
        print(f"Error in crop recommendation: {e}")
        return None

# **Fertilizer Recommendation (with Noisy Data Handling)**
def get_fertilizer_recommendation(num_features, cat_features):
    """ Recommend the best fertilizer based on numerical & categorical features. """
    try:
        num_features = clean_data(num_features)  # Clean numerical features

        scaler_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'fertilizer_scaler.pkl')
        model_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'fertilizer_model.pkl')

        with open(scaler_path, 'rb') as f:
            fertilizer_scaler = pickle.load(f)
        with open(model_path, 'rb') as f:
            fertilizer_model = pickle.load(f)

        scaled_features = fertilizer_scaler.transform([num_features])
        item = np.concatenate([scaled_features, [cat_features]], axis=1)
        prediction = fertilizer_model.predict(item)[0]
        return fertilizer_classes[prediction]
    except Exception as e:
        print(f"Error in fertilizer recommendation: {e}")
        return None

# **Crop Disease Classes**
crop_diseases_classes = {
    'strawberry': [(0, 'Leaf_scorch'), (1, 'healthy')],
    'potato': [(0, 'Early_blight'), (1, 'Late_blight'), (2, 'healthy')],
    'corn': [(0, 'Cercospora_leaf_spot Gray_leaf_spot'), (1, 'Common_rust_'),
             (2, 'Northern_Leaf_Blight'), (3, 'healthy')],
    'apple': [(0, 'Apple_scab'), (1, 'Black_rot'), (2, 'Cedar_apple_rust'), (3, 'healthy')],
    'cherry': [(0, 'Powdery_mildew'), (1, 'healthy')],
    'grape': [(0, 'Black_rot'), (1, 'Esca_(Black_Measles)'), (2, 'Leaf_blight_(Isariopsis_Leaf_Spot)'),
              (3, 'healthy')],
    'peach': [(0, 'Bacterial_spot'), (1, 'healthy')],
    'pepper': [(0, 'Bacterial_spot'), (1, 'healthy')],
    'tomato': [(0, 'Bacterial_spot'), (1, 'Early_blight'), (2, 'Late_blight'),
               (3, 'Leaf_Mold'), (4, 'Septoria_leaf_spot'), (5, 'Spider_mites Two-spotted_spider_mite'),
               (6, 'Target_Spot'), (7, 'Tomato_Yellow_Leaf_Curl_Virus'), (8, 'Tomato_mosaic_virus'),
               (9, 'healthy')]
}

# **Crop List**
crop_list = list(crop_diseases_classes.keys())

# **Crop Names**
crops = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute',
         'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange',
         'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

# **Soil & Crop Types**
soil_types = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
Crop_types = ['Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets', 'Oil seeds', 'Paddy',
              'Pulses', 'Sugarcane', 'Tobacco', 'Wheat']

# **Fertilizer Classes**
fertilizer_classes = ['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP', 'Urea']
