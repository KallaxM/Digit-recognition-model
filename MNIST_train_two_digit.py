"""Preprocessing of MNIST data"""

import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
import random
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Function to generate a random background color
def random_background_color():
    color_type = random.choice(['gray', 'green', 'red'])
    if color_type == 'gray':
        color = random.randint(50, 200)  # Shades of gray
        return (color, color, color)
    elif color_type == 'green':
        return (random.randint(0, 50), random.randint(100, 200), random.randint(0, 50))
    elif color_type == 'red':
        return (random.randint(100, 200), random.randint(0, 50), random.randint(0, 50))

# Function to preprocess MNIST for two digits
def preprocess_two_digits(images, labels, new_digit_size=(14, 20), bg_size=(36, 24)):
    processed_images = []
    processed_labels = []
    for _ in range(len(images)):
        # Select two random digits
        idx1, idx2 = random.sample(range(len(images)), 2)
        digit1, label1 = images[idx1], labels[idx1]
        digit2, label2 = images[idx2], labels[idx2]
        
        # Resize digits to 14x20
        resized_digit1 = cv2.resize(digit1, new_digit_size, interpolation=cv2.INTER_AREA)
        resized_digit2 = cv2.resize(digit2, new_digit_size, interpolation=cv2.INTER_AREA)
        
        # Normalize digits to binary
        _, binary_digit1 = cv2.threshold(resized_digit1, 128, 255, cv2.THRESH_BINARY)
        _, binary_digit2 = cv2.threshold(resized_digit2, 128, 255, cv2.THRESH_BINARY)
        
        # Create a random background
        bg_color = random_background_color()
        background = np.zeros((bg_size[1], bg_size[0], 3), dtype=np.uint8)
        background[:, :] = bg_color
        
        # Fixed placement
        x1_offset = (bg_size[0] - 2 * new_digit_size[0]) // 2  # Center the pair of digits horizontally
        y_offset = (bg_size[1] - new_digit_size[1]) // 2       # Center the digits vertically
        x2_offset = x1_offset + new_digit_size[0]              # Place second digit immediately to the right
        
        # Place the digits on the background
        for c in range(3):
            background[y_offset:y_offset+new_digit_size[1], x1_offset:x1_offset+new_digit_size[0], c] = \
                np.where(binary_digit1 > 0, 255, background[y_offset:y_offset+new_digit_size[1], x1_offset:x1_offset+new_digit_size[0], c])
            background[y_offset:y_offset+new_digit_size[1], x2_offset:x2_offset+new_digit_size[0], c] = \
                np.where(binary_digit2 > 0, 255, background[y_offset:y_offset+new_digit_size[1], x2_offset:x2_offset+new_digit_size[0], c])
        
        processed_images.append(background)
        processed_labels.append((label1, label2))  # Store the pair of labels
    
    return np.array(processed_images), np.array(processed_labels)



# Preprocess the MNIST training and test sets
processed_train_images, processed_train_labels = preprocess_two_digits(x_train, y_train)
processed_test_images, processed_test_labels = preprocess_two_digits(x_test, y_test)

# Normalize images (scaling pixel values between 0 and 1)
processed_train_images = processed_train_images.astype('float32') / 255.0
processed_test_images = processed_test_images.astype('float32') / 255.0

# Visualize some examples from the preprocessed dataset
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(processed_train_images[i])
    ax.axis('off')
    ax.set_title(f"Labels: {processed_train_labels[i]}")
plt.tight_layout()
plt.show()

"""Model Definition"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define a CNN model for two-digit recognition
def build_two_digit_model(input_shape=(24, 36, 3)):
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # Shared feature extractor
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Separate outputs for each digit
    digit1_output = Dense(10, activation='softmax', name='digit1')(x)
    digit2_output = Dense(10, activation='softmax', name='digit2')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=[digit1_output, digit2_output])
    model.compile(
        optimizer=Adam(),
        loss={'digit1': 'sparse_categorical_crossentropy', 'digit2': 'sparse_categorical_crossentropy'},
        metrics={'digit1': 'accuracy', 'digit2': 'accuracy'}
    )
    return model


"""Training the Model"""

from sklearn.model_selection import train_test_split

# Prepare data for training
X_train, X_val, y_train, y_val = train_test_split(
    processed_train_images, processed_train_labels, test_size=0.2, random_state=42
)

# Separate labels for two outputs
y_train_digit1 = np.array([label[0] for label in y_train])
y_train_digit2 = np.array([label[1] for label in y_train])
y_val_digit1 = np.array([label[0] for label in y_val])
y_val_digit2 = np.array([label[1] for label in y_val])

# Build the model
model = build_two_digit_model(input_shape=(24, 36, 3))

# Train the model
history = model.fit(
    X_train, 
    {'digit1': y_train_digit1, 'digit2': y_train_digit2},  # Outputs
    epochs=10, 
    batch_size=32,
    validation_data=(X_val, {'digit1': y_val_digit1, 'digit2': y_val_digit2})
)


"""Evaluation"""

# Plot accuracy for both outputs
plt.plot(history.history['digit1_accuracy'], label='digit1_train_accuracy')
plt.plot(history.history['val_digit1_accuracy'], label='digit1_val_accuracy')
plt.plot(history.history['digit2_accuracy'], label='digit2_train_accuracy')
plt.plot(history.history['val_digit2_accuracy'], label='digit2_val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss for both outputs
plt.plot(history.history['digit1_loss'], label='digit1_train_loss')
plt.plot(history.history['val_digit1_loss'], label='digit1_val_loss')
plt.plot(history.history['digit2_loss'], label='digit2_train_loss')
plt.plot(history.history['val_digit2_loss'], label='digit2_val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Save the trained model
model.save('two_digit_recognition_model.h5')