# Two-Digit Recognition using MNIST Dataset

Student project

## Project Results and Overview

This project aims to develop a deep learning-based digit recognition system to extract acceleration data from YouTube videos of opponent cars in an autonomous car racing competition. Specifically, the system will recognize and process the digits displayed on the dash cam of these videos, allowing the extraction of speed data. Than we can differentiate the time-based velocity data to obtain acceleration data.

In the present state, the project is able to recognize two-digit numbers, where each digit is sourced from the MNIST dataset, with randomly assigned backgrounds in shades of gray, green, or red. The model preprocesses the MNIST dataset to form pairs of digits, resizes them, normalizes them, and places them on a background. It then trains a Convolutional Neural Network (CNN) to predict each digit in the pair. The prediction is specific to the images of the frames folder.

**Key objectives**:
- Preprocess MNIST data to form two-digit images.
- Train a CNN to predict the first and second digits of the pair.
- Evaluate the model on its accuracy for both digit outputs.

**Results**:
- The model achieves training effectively with two-digits
- The dataset augmentation with random background colors improved the accuracy of the model
- The model isn't very accurate on the new frames. It has a 50% success rate

## Source Code

The source code is organized in two python files, the training file and the prediction file :

```
- MNIST_train_two_digit.py          # Training file
- Run_two_digit.py                  # Prediction file
- two_digit_recognition_model.h5    # Trained model
- MNIST_train_two_digit_Results     # Results of training
- README.md                         # Project overview and documentation
```

### Dependencies

The project requires the following libraries:
- `numpy`
- `opencv-python`
- `tensorflow`
- `matplotlib`
- `sklearn`
  

## Performance Metrics

The model's performance is evaluated on both accuracy and loss metrics for each digit prediction (digit1 and digit2). The following charts show the accuracy and loss over training epochs:

### Accuracy
- **Train Accuracy** for both digits increased steadily during training, reaching over 95% accuracy for both digits on the validation set.

### Loss
- **Loss** decreased for both digits across epochs, confirming that the model was learning efficiently.

You can visualize these metrics int the MNIST_train_two_digit_Results folder.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/two-digit-recognition.git
   cd two-digit-recognition
   ```

2. Install the dependencies

   
4. Run the project:
   - Run Run_two_digit.py file
   - Feel free to modify the input images

## References and Documentation

This project is based on the MNIST dataset and standard Convolutional Neural Network (CNN) techniques.

1. Kaggle course: https://www.kaggle.com/learn/intro-to-deep-learning
3. MNIST Dataset: [Yann LeCun, et al., 1998.](http://yann.lecun.com/exdb/mnist/)
4. Keras documentation for CNN models: [Keras Documentation](https://keras.io/)

## Issues and Contributions

### Known Issues
- The dataset isn't the most suitable for this application.
- The prediction code is predicting only for 1 frame.
- The prediction code is using known coordinates of the digit area to predict.
- As the dataset isn't the best one, performance of the model on the images of the frames folder isn't great.

### Contributing
Contributions are welcome! You can contribute by solving the issues shown before.

## Future Work

- Changing the dataset.
- Fine tuning for better performance.
- Improvement of the prediction code to make it usable on any youtube video showing a speed monitor.
