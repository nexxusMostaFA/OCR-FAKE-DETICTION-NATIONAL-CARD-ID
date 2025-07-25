import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import cv2
import subprocess
import sys
import platform

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Increased batch size for better gradient estimates
EPOCHS = 25  # More epochs with early stopping
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2

# Paths
REAL_PATH = "images/real"
FAKE_PATH = "images/fake"
MODEL_PATH = "id_detection_model.keras"

# Custom Tesseract path
TESSERACT_PATH = r"C:\Users\mostafa\Desktop\graduation projects\card_id_fake_detection\Tesseract-OCR"

# Install Tesseract if needed
def setup_tesseract():
    """Install and set up Tesseract OCR based on the platform"""
    try:
        import pytesseract
        
        # Set the tesseract path
        pytesseract.pytesseract.tesseract_cmd = os.path.join(TESSERACT_PATH, "tesseract.exe")
        
        # Test if tesseract works
        pytesseract.get_tesseract_version()
        print("Tesseract is configured and working properly.")
        return True
    except Exception as e:
        print(f"Tesseract setup error: {e}")
        system = platform.system().lower()
        
        if system == "windows":
            print(f"Using custom Tesseract path: {TESSERACT_PATH}")
            try:
                import pytesseract
                # Set the Tesseract executable path
                pytesseract.pytesseract.tesseract_cmd = os.path.join(TESSERACT_PATH, "tesseract.exe")
                # Test if it works now
                pytesseract.get_tesseract_version()
                print("Tesseract is now configured and working properly.")
                return True
            except Exception as e:
                print(f"Failed to configure Tesseract: {e}")
                print("Please ensure the path is correct and Tesseract is properly installed.")
                return False
        elif system == "linux":
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pytesseract"])
                subprocess.check_call(["apt-get", "update"])
                subprocess.check_call(["apt-get", "install", "-y", "tesseract-ocr", "tesseract-ocr-ara", "tesseract-ocr-eng"])
                print("Tesseract installed successfully on Linux")
                return True
            except Exception as e:
                print(f"Failed to install Tesseract: {e}")
                return False
        elif system == "darwin":  # macOS
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pytesseract"])
                subprocess.check_call(["brew", "install", "tesseract", "tesseract-lang"])
                print("Tesseract installed successfully on macOS")
                return True
            except Exception as e:
                print(f"Failed to install Tesseract: {e}")
                print("On macOS, try installing manually with: brew install tesseract tesseract-lang")
                return False
        return False

def get_image_paths(real_dir, fake_dir):
    """Get balanced paths for real and fake images"""
    real_paths = []
    for dirpath, _, filenames in os.walk(real_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                real_paths.append(os.path.join(dirpath, filename))
    
    fake_paths = []
    for dirpath, _, filenames in os.walk(fake_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                fake_paths.append(os.path.join(dirpath, filename))
    
    print(f"Found {len(real_paths)} real images and {len(fake_paths)} fake images")
    
    # Balance the dataset by limiting the majority class
    # min_count = min(len(real_paths), len(fake_paths))
    # if len(real_paths) > min_count:
    #    real_paths = real_paths[:min_count]
    # if len(fake_paths) > min_count:
    #    fake_paths = fake_paths[:min_count]
    
    # Instead of balancing by limiting samples, we'll use class weights
    
    return real_paths, fake_paths

def create_dataset(real_paths, fake_paths):
    """Create and split the dataset"""
    # Combine paths and create labels
    all_paths = real_paths + fake_paths
    # 0 for real, 1 for fake
    labels = [0] * len(real_paths) + [1] * len(fake_paths)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        all_paths, labels, test_size=VALIDATION_SPLIT, random_state=42, stratify=labels
    )
    
    # Compute class weights to address imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"Class weights: {class_weight_dict}")
    
    return X_train, X_test, y_train, y_test, class_weight_dict

def create_data_generators(X_train, X_test, y_train, y_test):
    """Create data generators with augmentation"""
    
    # Function to load and preprocess images
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        
        # Reshape the label to match the expected output shape
        label = tf.reshape(label, (1,))
        return img, label
    
    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Data augmentation layer
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ])
    
    # Apply augmentation to training data
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Configure datasets for performance
    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds

def create_efficient_model():
    """Create an EfficientNetV2B0 model with improved architecture"""
    # Use a more recent EfficientNet version
    base_model = applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_preprocessing=True
    )
    
    # Freeze most of the base model
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Add more sophisticated head for better feature extraction
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Define custom F1 score metric that works with binary classification
    def f1_score(y_true, y_pred):
        y_pred = tf.round(y_pred)  # Threshold at 0.5
        
        # Calculate precision and recall
        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32))
        predicted_positives = tf.reduce_sum(tf.cast(tf.equal(y_pred, 1), tf.float32))
        actual_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), tf.float32))
        
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return f1
    
    # Compile with appropriate metrics
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            f1_score  # Custom F1 score metric
        ]
    )
    
    return model

def train_model(model, train_ds, test_ds, class_weights):
    """Train the model with proper callbacks"""
    
    # Define callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_f1_score',  # Updated monitor to match the new metric name
            mode='max',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_f1_score',  # Updated monitor to match the new metric name
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train in two phases
    print("Phase 1: Training with frozen base layers...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=10,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Unfreeze all layers for fine-tuning
    print("Phase 2: Fine-tuning all layers...")
    for layer in model.layers[0].layers:
        layer.trainable = True
    
    # Recompile with lower learning rate
    # Define custom F1 score metric that works with binary classification
    def f1_score(y_true, y_pred):
        y_pred = tf.round(y_pred)  # Threshold at 0.5
        
        # Calculate precision and recall
        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32))
        predicted_positives = tf.reduce_sum(tf.cast(tf.equal(y_pred, 1), tf.float32))
        actual_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), tf.float32))
        
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return f1
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            f1_score  # Custom F1 score metric
        ]
    )
    
    # Continue training
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        initial_epoch=history.epoch[-1] + 1,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return model

def evaluate_model(model, test_ds):
    """Evaluate the model and show detailed metrics"""
    print("Evaluating model...")
    results = model.evaluate(test_ds, verbose=1)
    
    metrics = {name: value for name, value in zip(model.metrics_names, results)}
    
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Get predictions
    all_predictions = []
    all_labels = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        predictions = predictions.flatten()
        all_predictions.extend(predictions)
        all_labels.extend(labels.numpy().flatten())  # Flattening because labels are now (batch_size, 1)
    
    # Convert to binary predictions
    binary_predictions = [1 if p >= 0.5 else 0 for p in all_predictions]
    
    # Calculate per-class accuracy
    real_indices = [i for i, label in enumerate(all_labels) if label == 0]
    fake_indices = [i for i, label in enumerate(all_labels) if label == 1]
    
    real_correct = sum(1 for i in real_indices if binary_predictions[i] == 0)
    fake_correct = sum(1 for i in fake_indices if binary_predictions[i] == 1)
    
    real_accuracy = real_correct / len(real_indices) if real_indices else 0
    fake_accuracy = fake_correct / len(fake_indices) if fake_indices else 0
    
    print(f"\nReal ID Accuracy: {real_accuracy:.4f} ({real_correct}/{len(real_indices)})")
    print(f"Fake ID Accuracy: {fake_accuracy:.4f} ({fake_correct}/{len(fake_indices)})")
    
    # Find optimal threshold
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        predictions_at_threshold = [1 if p >= threshold else 0 for p in all_predictions]
        
        # Calculate metrics at this threshold
        true_positives = sum(1 for i, label in enumerate(all_labels) if label == 1 and predictions_at_threshold[i] == 1)
        false_positives = sum(1 for i, label in enumerate(all_labels) if label == 0 and predictions_at_threshold[i] == 1)
        false_negatives = sum(1 for i, label in enumerate(all_labels) if label == 1 and predictions_at_threshold[i] == 0)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nOptimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    return best_threshold

def extract_text_with_ocr(image_path):
    """Extract text from an image using OCR with better preprocessing"""
    try:
        import pytesseract
        
        # Ensure tesseract path is set
        pytesseract.pytesseract.tesseract_cmd = os.path.join(TESSERACT_PATH, "tesseract.exe")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return ""
        
        # Enhanced preprocessing pipeline
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Apply deskewing if needed (simplified here)
        angle = 0
        try:
            coords = np.column_stack(np.where(denoised > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
        except:
            angle = 0
            
        if abs(angle) > 0.5:
            (h, w) = denoised.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            denoised = cv2.warpAffine(denoised, M, (w, h), 
                               flags=cv2.INTER_CUBIC, 
                               borderMode=cv2.BORDER_REPLICATE)
        
        # Extract text with multiple languages (Arabic + English)
        text = pytesseract.image_to_string(denoised, lang='ara+eng', config='--psm 11')
        return text
    
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

def main():
    """Main function to run the ID card verification system"""
    print("Starting improved ID card verification system...")
    
    # Setup Tesseract
    tesseract_ready = setup_tesseract()
    if not tesseract_ready:
        print("Warning: Tesseract is not properly set up. OCR functions will be limited.")
    
    # Get image paths
    real_paths, fake_paths = get_image_paths(REAL_PATH, FAKE_PATH)
    
    # Create dataset
    X_train, X_test, y_train, y_test, class_weights = create_dataset(real_paths, fake_paths)
    
    # Create data generators
    train_ds, test_ds = create_data_generators(X_train, X_test, y_train, y_test)
    
    # Create model
    model = create_efficient_model()
    
    # Train the model
    model = train_model(model, train_ds, test_ds, class_weights)
    
    # Evaluate the model
    threshold = evaluate_model(model, test_ds)
    
    # Save the final model using native Keras format
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Test OCR if available
    if tesseract_ready:
        print("\nTesting OCR on sample images:")
        for i, path in enumerate(X_test[:3]):
            print(f"Sample {i+1}:")
            text = extract_text_with_ocr(path)
            print(f"  Extracted text length: {len(text)} characters")
            if len(text) > 0:
                print(f"  First 100 chars: {text[:100]}")
            print()
    
    print("Finished!")

if __name__ == "__main__":
    main()