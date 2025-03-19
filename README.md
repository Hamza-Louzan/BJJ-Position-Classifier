# BJJ Position Detection


<p align="center">
  <img src="images/9nta28.gif" />
</p>

This project uses computer vision and machine learning to automatically detect and classify Brazilian Jiu-Jitsu (BJJ) positions from images. The system detects the poses of two athletes and predicts the BJJ position they are in.

## Features

- Automated detection of two BJJ athletes in a single image
- Pose estimation using MediaPipe
- Classification of common BJJ positions (standing, mount, guard, etc.)
- Training pipeline for creating your own position classifier
- Visualization of detected poses and predicted positions

## Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/bjj-position-detection.git
cd bjj-position-detection
```

2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data Structure

The expected data structure for training is:
```
data/
  ├── annotations.json  # Contains pose and position annotations
  └── images/          # Contains BJJ images
      ├── 0000001.jpg
      ├── 0000002.jpg
      └── ...
```

The annotations.json file should contain entries with the following structure:
```json
[
  {
    "position": "standing",
    "image": "0000007",
    "frame": 7,
    "pose1": [...keypoints...],
    "pose2": [...keypoints...]
  },
  ...
]
```

## Usage

### Training a Model

To train a new model using your own data:

```bash
python main.py train --annotations data/annotations.json --images data/images/ --output models/
```

This will create three files in the specified output directory:
- `bjj_position_model.h5`: The trained model
- `bjj_scaler.pkl`: The feature scaler
- `bjj_label_encoder.pkl`: The label encoder

### Running Prediction

To predict the BJJ position in a new image:

```bash
python main.py predict --image path/to/image.jpg --model models/bjj_position_model.h5 --scaler models/bjj_scaler.pkl --encoder models/bjj_label_encoder.pkl
```

## How It Works

1. **Pose Detection**: The system uses MediaPipe Pose to detect the keypoints of both athletes in the image.
2. **Feature Extraction**: The keypoints are normalized and interaction features (like distances between athletes) are computed.
3. **Position Classification**: The extracted features are fed into a neural network to predict the BJJ position.

## Project Structure

- `main.py`: Main script for running training and prediction
- `pose_utils.py`: Utilities for pose detection and processing
- `model_training.py`: Neural network model and training pipeline
- `inference.py`: Functions for running predictions on new images
- `requirements.txt`: Package dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The [MediaPipe](https://github.com/google/mediapipe) project for pose estimation
- TensorFlow and Keras for the neural network framework
- - Vicos.si resources
