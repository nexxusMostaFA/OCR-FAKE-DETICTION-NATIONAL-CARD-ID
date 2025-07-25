import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import logging
import requests
import re
import time

logging.basicConfig(level=logging.DEBUG,   
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("id_verification.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

IMG_SIZE = (224, 224)
MODEL_PATH = "id_detection_model.keras"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

NATIONAL_ID_API_BASE = "http://127.0.0.1:8000/national-id"   
ID_EXTRACTION_API = "http://localhost:5678/process_id"

MAX_RETRIES = 3
RETRY_DELAY = 1  

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    def f1_score(y_true, y_pred):
        y_pred = tf.round(y_pred)
        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32))
        predicted_positives = tf.reduce_sum(tf.cast(tf.equal(y_pred, 1), tf.float32))
        actual_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), tf.float32))
        
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
        
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return f1

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            f1_score
        ]
    )
    
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image from {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img

    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def sanitize_national_id(id_string):
    """
    Clean and validate the national ID string
    - Remove any non-digit characters
    - Ensure it's exactly 14 digits
    """
    if not id_string:
        return None
        
    cleaned_id = re.sub(r'\D', '', id_string)
    
    if len(cleaned_id) == 14:
        logger.info(f"Successfully sanitized national ID: {cleaned_id}")
        return cleaned_id
    else:
        logger.warning(f"Invalid national ID format after sanitization: {cleaned_id} (length: {len(cleaned_id)})")
        return None

def extract_national_id_from_api(image_path, max_retries=MAX_RETRIES):
    """
    Send image to external API for national ID extraction with retry mechanism
    Returns both the national_id and the full OCR response
    """
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            logger.info(f"Attempting to extract national ID from image (attempt {retry_count + 1}/{max_retries})")
            
            with open(image_path, 'rb') as img_file:
                files = {'file': (os.path.basename(image_path), img_file, 'image/jpeg')}
                
                logger.debug(f"Sending request to {ID_EXTRACTION_API}")
                
                response = requests.post(ID_EXTRACTION_API, files=files, timeout=10)
                
                logger.debug(f"API response status: {response.status_code}")
                logger.debug(f"API response content: {response.text}")
                
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        logger.info(f"ID extraction API response: {response_data}")
                        
                        if 'data' in response_data and 'national_id' in response_data['data'] and response_data['data']['national_id']:
                            sanitized_id = sanitize_national_id(response_data['data']['national_id'])
                            return sanitized_id, response_data
                        else:
                            logger.warning("No national ID found in API response")
                            return None, response_data
                    except ValueError as json_err:
                        logger.error(f"Failed to parse JSON response: {json_err}")
                        logger.error(f"Raw response: {response.text}")
                else:
                    logger.warning(f"ID extraction API returned status code: {response.status_code}")
            
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error during national ID extraction: {e}")
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"Unexpected error extracting national ID from API: {e}")
            break
    
    return None, None

def verify_national_id(national_id, max_retries=MAX_RETRIES):
    """
    Verify if the national ID is valid using the national ID API with retry mechanism
    """
    if not national_id or len(national_id) != 14:
        logger.warning(f"Invalid ID format before verification: {national_id}")
        return False, {"error": "Invalid ID format"}
    
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            api_url = f"{NATIONAL_ID_API_BASE}/{national_id}/"
            
            api_url = api_url.replace("//", "/").replace(":/", "://")
            
            logger.info(f"Verifying national ID: {national_id}")
            logger.debug(f"Verification API URL: {api_url}")
            
            response = requests.get(api_url, timeout=10)
            
            logger.debug(f"API response status: {response.status_code}")
            logger.debug(f"API response content: {response.text}")
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    logger.info(f"National ID API validation successful: {response_data}")
                    return True, response_data
                except ValueError as json_err:
                    logger.error(f"Failed to parse JSON response: {json_err}")
                    logger.error(f"Raw response: {response.text}")
            else:
                logger.warning(f"National ID API returned status code: {response.status_code}")
                logger.warning(f"Response content: {response.text}")
            
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error during national ID verification: {e}")
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"Unexpected error verifying national ID: {e}")
            break
    
    return False, {"error": f"Failed to verify national ID after {max_retries} attempts"}

def cleanup_file(filepath):
    """Helper function to clean up uploaded files"""
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            logger.info(f"Deleted file {filepath}")
        except Exception as e:
            logger.warning(f"Could not delete file {filepath}: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Three-stage verification process:
    1. Extract national ID from image using external OCR API
    2. Verify national ID validity using national ID API
    3. Predict if the ID is real or fake using the AI model
    """
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded properly'
        }), 500

    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file part in the request'
        }), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    filepath = None
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved at {filepath}")
        
        result = {
            'stage1_extraction': None,
            'stage2_validation': None,
            'stage3_ai_verification': None,
            'ocr_response': None,  
            'final_result': None
        }
        
        logger.info("Starting Stage 1: National ID extraction")
        national_id, ocr_response = extract_national_id_from_api(filepath)
        
        result['ocr_response'] = ocr_response
        
        result['stage1_extraction'] = {
            'extracted_id': national_id,
            'success': national_id is not None and len(national_id) == 14 if national_id else False
        }
        
        if not national_id or len(national_id) != 14:
            logger.warning(f"Stage 1 failed: Could not extract valid national ID")
            result['final_result'] = 'Fake ID (Failed to extract valid national ID)'
            
            cleanup_file(filepath)
                
            return jsonify({
                'status': 'fake',
                'message': 'Invalid National ID - Could not extract valid 14-digit ID from the image',
                'process_results': result
            }), 200
        
        logger.info(f"Stage 1 successful - Extracted National ID: {national_id}")
        
        logger.info("Starting Stage 2: National ID validation")
        is_valid_id, api_response = verify_national_id(national_id)
        
        result['stage2_validation'] = {
            'is_valid': is_valid_id,
            'api_response': api_response
        }
        
        if not is_valid_id:
            logger.warning(f"Stage 2 failed: Invalid national ID")
            result['final_result'] = 'Fake ID (Invalid national ID)'
            
            cleanup_file(filepath)
                
            return jsonify({
                'status': 'fake',
                'message': 'Invalid National ID',
                'process_results': result
            }), 200
        
        logger.info(f"Stage 2 successful - National ID is valid")
        
        logger.info("Starting Stage 3: AI model verification")
        preprocessed_img = preprocess_image(filepath)
        if preprocessed_img is None:
            logger.error("Stage 3 failed: Could not preprocess image")
            result['final_result'] = 'Unknown (Failed to process image for AI verification)'
            
            cleanup_file(filepath)
                
            return jsonify({
                'status': 'error',
                'message': 'Failed to preprocess image for model prediction',
                'process_results': result
            }), 500
        
        prediction = model.predict(preprocessed_img)[0][0]
        
        is_fake = bool(prediction >= 0.5)
        confidence = float(prediction) if is_fake else float(1 - prediction)
        
        result['stage3_ai_verification'] = {
            'is_fake': is_fake,
            'is_real': not is_fake,
            'confidence': round(confidence * 100, 2),
            'raw_prediction': float(prediction)
        }
        
        logger.info(f"Stage 3 complete - AI model prediction: {'Fake' if is_fake else 'Real'} with {round(confidence * 100, 2)}% confidence")
        
        if is_fake:
            result['final_result'] = "Fake ID (AI model detection)"
            status = "fake"
        else:
            result['final_result'] = "Real ID (Passed all verification stages)"
            status = "real"
        
        cleanup_file(filepath)
        
        return jsonify({
            'status': status,
            'process_results': result
        })

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        cleanup_file(filepath)
            
        return jsonify({
            'status': 'error',
            'message': f'Error processing image: {str(e)}'
        }), 500

@app.route('/validate-id', methods=['POST'])
def validate_id():
    return jsonify({
        'status': 'deprecated',
        'message': 'This endpoint is deprecated. Please use /predict instead.'
    }), 308

@app.route('/health', methods=['GET'])
def health_check():
    services_status = {
        'app': 'running',
        'model': 'loaded' if model is not None else 'not_loaded',
        'ocr_api': 'unknown',
        'id_api': 'unknown'
    }
    
    try:
        response = requests.get(ID_EXTRACTION_API.replace('/process_id_card', '/health'), timeout=2)
        services_status['ocr_api'] = 'running' if response.status_code == 200 else 'error'
    except:
        services_status['ocr_api'] = 'not_running'
    
    try:
        response = requests.get(f"{NATIONAL_ID_API_BASE}/health/", timeout=2)
        services_status['id_api'] = 'running' if response.status_code == 200 else 'error'
    except:
        services_status['id_api'] = 'not_running'
    
    overall_status = 'healthy' if all(status in ['running', 'loaded'] for status in services_status.values()) else 'degraded'
    
    return jsonify({
        'status': overall_status,
        'services': services_status
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9000))
    app.run(host='0.0.0.0', port=port, debug=False)