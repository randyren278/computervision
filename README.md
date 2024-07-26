# Facial Analysis Attendance App

## Description

This project is a Facial Analysis Attendance application that utilizes advanced computer vision and machine learning algorithms to recognize faces and mark attendance. The application captures images from a webcam, detects faces, recognizes them from a pre-loaded database, and logs new faces with titles.

### Features

- Real-time face recognition using webcam input
- Automatic attendance marking with timestamps
- Option to add new faces to the database
- Simple GUI for user interaction

## Gallery
![alt text](<distance metric.jpeg>)
Shown on the left is the intial image and on the right the test image along with what's known as the distance metric (the larger the value, the less similar the images).
![alt text](<no tracking.png>)
Shwon on the left is an example of the algorithim detecing a face but not finding it in the database. On the right is an example of a face the algorithim found within its database. Also pictured, is the save face button in the top left corner.


## Technologies Used

### Python
Python is chosen for its simplicity and the extensive range of libraries available for machine learning and computer vision. It allows for quick development and easy integration of different modules required for this project.

### OpenCV
OpenCV (Open Source Computer Vision Library) is used for image and video processing. Key functionalities utilized in this project include:
- **Image Reading and Display**: OpenCV reads images from the file system and captures video frames from the webcam.
- **Color Space Conversion**: Converts images from BGR (default in OpenCV) to RGB for compatibility with the `face_recognition` library.
- **Drawing**: Used to draw rectangles around detected faces.

### face_recognition
The `face_recognition` library leverages deep learning models for face detection and recognition. It simplifies the process by providing high-level functions to:
- **Face Detection**: Uses a Histogram of Oriented Gradients (HOG) combined with a linear classifier for robust face detection.
- **Facial Feature Encoding**: Converts facial features into a 128-dimensional vector using deep convolutional neural networks (CNNs). This encoding represents the unique features of a face.
- **Face Comparison**: Compares these 128-dimensional vectors to determine if two faces are of the same person. The Euclidean distance between vectors is calculated, with smaller distances indicating higher similarity.

### dlib
dlib is a toolkit that provides machine learning algorithms and tools. The `face_recognition` library builds on dlib's capabilities, particularly:
- **Face Landmark Detection**: dlib detects facial landmarks which are then used to align the face to a standard position before encoding.
- **Deep Learning**: dlib's implementation of deep CNNs is used for generating facial encodings, ensuring high accuracy in face recognition.

### Tkinter
Tkinter is the standard GUI library for Python, used to create the user interface of the application. Key components include:
- **Buttons**: For user actions such as saving a new face.
- **Dialogs**: Custom dialogs to prompt users for input (e.g., entering a name for a captured face).

### PIL (Python Imaging Library)
PIL, now maintained under the name Pillow, is used for handling images within the Tkinter interface. It converts OpenCV images to a format that can be displayed in the Tkinter window.

### NumPy
NumPy is used for numerical operations, particularly handling arrays and matrices which are common in image processing tasks.

### Detailed Workflow and Algorithms

1. **Loading and Encoding Known Faces**:
   - The application loads images from the `ImageDatabase` directory.
   - Each image is converted to RGB and passed to `face_recognition.face_encodings` to generate a 128-dimensional vector representing the face.
   - These encodings are stored in a list for comparison during real-time face recognition.

2. **Real-time Face Recognition**:
   - The webcam captures video frames, which are processed in real-time.
   - Each frame is resized and converted to RGB.
   - Faces are detected using `face_recognition.face_locations`, which employs the HOG algorithm for fast and accurate detection.
   - The detected faces are then encoded using the same CNN-based method, generating 128-dimensional vectors.
   - These vectors are compared to the known face encodings using `face_recognition.compare_faces` and `face_recognition.face_distance`. The Euclidean distance between vectors helps determine the closest match.

3. **Marking Attendance**:
   - When a face is recognized, the person's name is logged along with the current timestamp in an `Attendance.csv` file.
   - The application ensures that each person is only logged once per session by checking existing entries in the CSV file.

4. **Adding New Faces**:
   - Users can add new faces to the database through the GUI.
   - The application captures an image from the webcam, prompts the user for a name, and saves the image to the `ImageDatabase` directory.
   - The new face is then encoded and added to the list of known faces, updating the real-time recognition capability.

## Installation

1. **Install the required libraries:**
    pip install -r requirements.txt

2. **Run the app:**
    python main.py