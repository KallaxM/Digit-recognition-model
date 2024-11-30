import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('two_digit_recognition_model.h5')

# Function to preprocess a new image for digit recognition
def preprocess_new_image(image, digit_area=(545, 196, 581, 220), new_digit_size=(14, 20), bg_size=(36, 24)):
    """
    Preprocesses a new image for prediction.
    Args:
        image (np.ndarray): Input image of size 640x360.
        digit_area (tuple): Coordinates of the digit area as (x_start, y_start, x_end, y_end).
        new_digit_size (tuple): Size of each digit as (width, height).
        bg_size (tuple): Size of the background (24x36).
    Returns:
        np.ndarray: Preprocessed image ready for prediction.
    """
    x_start, y_start, x_end, y_end = digit_area
    cropped = image[y_start:y_end, x_start:x_end]  # Crop the digit area

    # Resize to background size while preserving aspect ratio
    resized = cv2.resize(cropped, bg_size, interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    preprocessed = resized.astype('float32') / 255.0
    
    # Visualize the preprocessing step
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for visualization
    plt.title("Preprocessed Image")
    plt.axis('off')
    plt.show()

    return np.expand_dims(preprocessed, axis=0)  # Add batch dimension

# Function to predict digits from a new image
def predict_digits(image_path):
    """
    Predicts the two digits in a new image.
    Args:
        image_path (str): Path to the input image.
    Returns:
        tuple: Predicted digits as (digit1, digit2).
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Preprocess the image
    preprocessed_image = preprocess_new_image(image)

    # Predict
    predictions = model.predict(preprocessed_image)

    # Extract predictions for each digit
    digit1 = np.argmax(predictions[0], axis=1)[0]
    digit2 = np.argmax(predictions[1], axis=1)[0]

    return digit1, digit2

# Example usage
image_path = 'frames1-10/frame_0005.png'  # Replace with the actual path to the image
try:
    digit1, digit2 = predict_digits(image_path)
    print(f"Predicted Digits: {digit1}, {digit2}")
except Exception as e:
    print(f"Error: {e}")
