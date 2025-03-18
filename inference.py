#!/usr/bin/env python3
'''
Inference Module for BJJ Position Detection
'''

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from pose_utils import detect_two_bjj_athletes, normalize_pose, compute_interaction_features

def predict_bjj_position(image_path, model_path='bjj_position_model.h5', 
                         scaler_path='bjj_scaler.pkl', encoder_path='bjj_label_encoder.pkl'):
    """
    Predict BJJ position from an image.
    
    Args:
        image_path: Path to the image
        model_path: Path to the trained model
        scaler_path: Path to the fitted scaler
        encoder_path: Path to the label encoder
        
    Returns:
        Tuple of (position_name, confidence)
    """
    # Load model and preprocessing components
    try:
        model = load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
            
        print("Model and preprocessing components loaded successfully")
    except Exception as e:
        print(f"Error loading model or preprocessing components: {e}")
        return None, None
    
    # Detect poses in the image
    pose1, pose2 = detect_two_bjj_athletes(image_path)
    
    if pose1 is None or pose2 is None:
        print("Failed to detect two athletes in the image")
        return None, None
    
    # Process the detected poses
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None, None
    
    img_height, img_width, _ = img.shape
    
    # Print the number of keypoints detected
    print(f"Detected pose1 with {len(pose1)} keypoints")
    print(f"Detected pose2 with {len(pose2)} keypoints")
    
    # Ensure we have exactly 17 keypoints (COCO format)
    # If more are detected, truncate to first 17
    pose1 = pose1[:17] if len(pose1) > 17 else pose1
    pose2 = pose2[:17] if len(pose2) > 17 else pose2
    
    # If fewer than 17 keypoints, pad with zeros
    while len(pose1) < 17:
        pose1.append([0.0, 0.0, 0.0])
    while len(pose2) < 17:
        pose2.append([0.0, 0.0, 0.0])
    
    # Normalize poses
    normalized_pose1 = normalize_pose(pose1, img_width, img_height)
    normalized_pose2 = normalize_pose(pose2, img_width, img_height)
    
    # Compute interaction features
    interaction_features = compute_interaction_features(normalized_pose1, normalized_pose2)
    
    # Prepare input for the model
    pose1_flat = [coord for kp in normalized_pose1 for coord in kp]  # 17*3 = 51 features
    pose2_flat = [coord for kp in normalized_pose2 for coord in kp]  # 17*3 = 51 features
    
    # Combine all features
    features = pose1_flat + pose2_flat + interaction_features
    features_array = np.array([features], dtype=np.float32)
    
    # Debug print the shape
    print(f"Features shape: {features_array.shape}")
    
    # Handle feature count mismatch
    if features_array.shape[1] != scaler.n_features_in_:
        print(f"WARNING: Feature count mismatch. Model expects {scaler.n_features_in_} features, but got {features_array.shape[1]}")
        
        # If we have too many features, truncate
        if features_array.shape[1] > scaler.n_features_in_:
            print("Truncating features to match model's expected input")
            features_array = features_array[:, :scaler.n_features_in_]
        else:
            # If we have too few features, pad with zeros
            padding = np.zeros((1, scaler.n_features_in_ - features_array.shape[1]))
            features_array = np.hstack([features_array, padding])
            print(f"Padded features to match model's expected input. New shape: {features_array.shape}")
    
    # Scale features
    scaled_features = scaler.transform(features_array)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    predicted_class_idx = np.argmax(prediction[0])
    
    # Get the position name and confidence
    position_name = label_encoder.inverse_transform([predicted_class_idx])[0]
    confidence = prediction[0][predicted_class_idx] * 100
    
    return position_name, confidence

def visualize_prediction(image_path, position, confidence):
    """
    Visualize BJJ position prediction on an image.
    
    Args:
        image_path: Path to the image
        position: Predicted position
        confidence: Confidence score (0-100)
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Convert from BGR to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Add text with prediction
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Position: {position} ({confidence:.1f}%)"
    cv2.putText(image_rgb, text, (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title(f"BJJ Position: {position}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()