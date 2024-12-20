# Computer Vision Projects

## 1. Traffic Light Classifier

The Traffic Light Classifier project employs classical computer vision methods to classify traffic light states (red, yellow, green) using image processing techniques.

### Key Components
- **Image Preprocessing**: The input traffic light images are converted to RGB color space, and color thresholds are applied to isolate red, yellow, and green components of the traffic light using color masks.
- **Color Detection**: Separate masks are applied to detect red, yellow, and green lights by examining the pixel intensities in each color’s masked region.
- **Classification**: Based on the most prominent color detected, the system classifies the traffic light state as red, yellow, or green using rule-based logic.

This classifier is lightweight and suitable for real-time traffic scene analysis for autonomous vehicles.

### Future Works
- Implement a deep learning model to classify the images in order to achieve higher accuracy.

---

## 2. Facial Key Points Detection

The Facial Key Points Detection project focuses on detecting and localizing 68 facial landmarks using a deep learning model.

### Key Components
- **Dataset**: Consists of images annotated with 68 facial key points, representing important facial features like the eyes, nose, mouth, and jawline.
- **Preprocessing**: Images are resized to 224x224 pixels, normalized, and augmented with random transformations for robust learning.
- **CNN Architecture**: A Convolutional Neural Network (CNN) extracts spatial features from the face images. This model architecture includes multiple convolutional layers followed by fully connected layers to regress the 68 facial key points.
- **Loss Function**: The Smooth L1 loss function is used for optimizing the regression of the facial key points.
- **Optimizer**: The Adam optimizer is employed for efficient training and fast convergence.

This project demonstrates a deep learning approach to facial landmark detection, crucial for applications in facial recognition and emotion analysis.

### Future Works
- Load a pretrained model like ResNet or VGG and fine-tune the model to achieve better results.
- Implement early stopping to avoid overfitting.

---

## 3. Image Captioning with LSTMs

The Image Captioning with LSTMs project generates descriptive captions for images by combining a Convolutional Neural Network (CNN) with Long Short-Term Memory (LSTM) networks.

### Key Components
- **Feature Extraction**: A pre-trained CNN extracts feature vectors from images, which act as input to the LSTM.
- **LSTM-based Caption Generation**: The LSTM network generates a sequence of words (captions) based on the extracted image features.
- **Text Processing**: Captions are tokenized, padded, and converted to sequences of integer indices for training. Special tokens like “start” and “end” are used for structured captions.
- **Training**: The model is trained using categorical cross-entropy loss to compare predicted captions with ground truth captions.

This project illustrates how deep learning methods can be applied to image-to-text tasks, bridging the gap between computer vision and natural language processing.

---

## 4. Landmark detection using SLAM

This project demonstrates my capability in implementing landmark detection and tracking using SLAM (Simultaneous Localization and Mapping) techniques. The goal was to detect and track specific landmarks across a series of images, enabling real-time navigation and environmental mapping applications.

## Project Overview

- **Objective**: To develop a robust system that identifies and continuously tracks landmarks in a dynamic environment.
- **Techniques Used**: Feature extraction, SLAM-based tracking, and real-time visualization of detected landmarks.
- **Tools and Libraries**: Python, OpenCV, NumPy, and Matplotlib.
  
## Key Features

- **Landmark Detection**: Employed image processing methods to detect notable landmarks, leveraging feature extraction techniques like edge detection and corner detection.
- **Tracking**: Used SLAM algorithms to maintain the positions of each detected landmark over time, even with minor changes in camera viewpoint.
- **Visualization**: Implemented a real-time visualization to display the detected and tracked landmarks, providing insights into tracking accuracy and stability.

## Results

The project successfully tracked landmarks across frames, showing robustness against moderate occlusions and variations in perspective. This implementation serves as a foundation for applications in autonomous navigation and mapping.

## Key Learnings

This project enhanced my understanding of:
- SLAM and its applications in computer vision.
- Feature detection and tracking for dynamic environments.
- Practical experience in visualizing real-time tracking data.


