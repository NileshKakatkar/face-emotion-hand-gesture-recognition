# Face Emotion & Hand Gesture Recognition

A computer-vision application that combines **facial emotion recognition** and **real-time hand gesture recognition** in a Flask web application.

The project contains both the **machine-learning training workflows** and the **trained models used by the live application**.

## Features

- Real-time facial emotion detection from a webcam.
- Real-time hand gesture recognition from a webcam.
- Facial emotion classification using a CNN trained on 48×48 grayscale face images.
- Hand landmark extraction using MediaPipe.
- Hand gesture classification using a trained TensorFlow/Keras model.
- Flask web interface for displaying live camera streams.
- Jupyter notebooks and Python scripts for model training and experimentation.

## High-Level Architecture

```text
                         Web Browser
                              |
                              v
                        Flask (app.py)
                         /          \
                        /            \
                       v              v
              Hand Gesture       Emotion Detection
              /video_Hand        /video_emotion
                    |                   |
                    v                   v
                Camera.py           Camera.py
                    |                   |
          +---------+---------+         |
          |                   |         |
          v                   v         v
      OpenCV Webcam       MediaPipe   OpenCV
          |               Hand        Haar Cascade
          |               Landmarks       |
          v                   |           v
    Video Frame          Landmark Data  Face ROI
                              |           |
                              v           v
                    Hand Gesture Model  CNN Emotion Model
                              |           |
                              +-----+-----+
                                    |
                                    v
                              Live Prediction
```

## Machine-Learning Components

### 1. Facial Emotion Recognition

The emotion-recognition pipeline is implemented primarily in:

```text
code/Face_Emotion_Recognition.ipynb
```

The notebook contains the CNN training workflow for facial emotion classification. The application uses seven emotion classes:

```text
Angry
Disgust
Fear
Happy
Sad
Surprise
Neutral
```

The trained model is stored as:

```text
models/face_emotion/model.json
models/face_emotion/model_weights.h5
```

At runtime, `Camera.py` reconstructs the model architecture from `model.json`, loads the learned weights from `model_weights.h5`, detects faces with OpenCV's Haar Cascade, resizes the face region to 48×48 grayscale pixels, normalizes it, and performs prediction.

### 2. Hand Gesture Recognition

The hand-gesture pipeline uses **MediaPipe Hands** to extract hand landmarks.

The preprocessing script is:

```text
code/data.py
```

It converts hand images/video frames into landmark data and stores the processed sequences under:

```text
MP_Data/
```

The available gesture classes in the deployed application include:

```text
big
call
catch
eat
hello
help
iloveyou
nice
no
ok
peace
please
result
stop
washroom
```

The project also contains different training experiments. In particular:

```text
code/train.ipynb
train.ipynb
```

contain LSTM-based training experiments, while:

```text
code/a.py
```

contains the hand-gesture model training implementation corresponding to the model used by the current application.

The trained deployed model is stored under:

```text
models/hand_gesture/
```

## Application Runtime Flow

The application starts from:

```text
app.py
```

The Flask application imports the two camera/inference classes from `Camera.py`:

```text
Handgesture
EmotionDetection
```

The main routes are:

| Route | Purpose |
|---|---|
| `/` | Main application page |
| `/handgesture` | Hand gesture detection page |
| `/video_Hand` | Live hand gesture video stream |
| `/EmotionDetection` | Emotion detection page |
| `/video_emotion` | Live emotion detection video stream |

`Camera.py` captures webcam frames using OpenCV and returns JPEG frames to Flask. Flask exposes those frames as an MJPEG stream to the browser.

## Project Structure

```text
FaceEmotion_Hand Gesture/
│
├── app.py                              # Flask application entry point
├── Camera.py                           # Webcam capture and ML inference
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
│
├── code/
│   ├── a.py                            # Hand gesture model training
│   ├── data.py                          # MediaPipe landmark preprocessing
│   ├── function.py                      # Hand-processing utilities
│   ├── train.ipynb                      # Hand gesture training experiments
│   ├── Face_Emotion_Recognition.ipynb   # CNN emotion training
│   ├── All_Emotions_Detection.jpg       # Emotion detection visualization
│   └── Logs/                             # Emotion training logs
│
├── train.ipynb                          # Additional training experiment
│
├── MP_Data/                             # Processed hand landmark dataset
│   ├── big/
│   ├── call/
│   ├── catch/
│   └── ...
│
├── models/
│   ├── face_emotion/
│   │   ├── model.json                   # Emotion model architecture
│   │   └── model_weights.h5             # Learned emotion weights
│   │
│   └── hand_gesture/                    # Saved TensorFlow/Keras model
│
├── Logs/                                # Hand gesture training logs
│
├── templates/
│   ├── index.html
│   ├── main.html
│   └── video.html
│
└── static/
    └── images/
        ├── emotion_detection.jpg
        └── hand_gesture.jpg
```

## Requirements

The dependencies are listed in:

```text
requirements.txt
```

The project uses Python, Flask, TensorFlow/Keras, OpenCV, MediaPipe, NumPy, Pillow, and Gunicorn.

A Python 3.10 environment was used while setting up and running this project.

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "FaceEmotion_Hand Gesture"
```

### 2. Create a virtual environment

Windows:

```bash
py -3.10 -m venv venv
```

Activate it:

```bash
venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> TensorFlow, NumPy, MediaPipe, and OpenCV versions can be sensitive to one another. If the application is being reproduced on another machine, use a compatible Python/dependency combination rather than arbitrarily upgrading individual packages.

### 4. Start the application

```bash
python app.py
```

Flask will start the local development server. Open the address printed in the terminal, typically:

```text
http://127.0.0.1:5000
```

## Using the Application

1. Open the main page in a browser.
2. Select hand gesture recognition or emotion detection.
3. Allow webcam access if the browser requests it.
4. The Flask server captures webcam frames through OpenCV.
5. The selected ML pipeline processes each frame.
6. The prediction is rendered on the live video stream.

## Training the Models

Training is **not required to run the existing application**, because trained model artifacts are already included in the project.

### Emotion model

The CNN training workflow is available in:

```text
code/Face_Emotion_Recognition.ipynb
```

The resulting model architecture and weights used by the application are stored in:

```text
models/face_emotion/
```

### Hand gesture model

The preprocessing workflow is:

```text
code/data.py
```

The processed landmark data is stored under:

```text
MP_Data/
```

The project contains multiple hand-gesture training experiments. The current deployed hand-gesture model is trained by the workflow in:

```text
code/a.py
```

and saved under:

```text
models/hand_gesture/
```

The notebooks contain additional/earlier training experiments and are retained as part of the project's development history.

## Important Runtime Files

The minimum application path is:

```text
app.py
   ↓
Camera.py
   ↓
OpenCV + MediaPipe + TensorFlow/Keras
   ↓
models/
   ↓
Live predictions
```

The `app.py` file does not train a model when the server starts. It starts Flask and creates the inference objects used for the live video routes.

## Development Notes

- `venv/` should not be committed to Git; dependencies should be reproduced using `requirements.txt`.
- Python `__pycache__/` and `.pyc` files are generated artifacts and should not be committed.
- The model files and processed dataset are included because they are required to reproduce/use the supplied project state.
- Training logs are retained to document the training runs and experiments.

## Stopping the Application

Press:

```text
Ctrl + C
```

in the terminal running Flask.

## License

No project-specific license is included in the supplied project files. Add an appropriate license before distributing the project publicly if required.
