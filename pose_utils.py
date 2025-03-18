#!/usr/bin/env python3
'''
Pose Utility Functions for BJJ Position Detection
'''

import json
import cv2
import mediapipe as mp
import numpy as np

def load_annotations(annotation_file):
    """
    Load annotations from a JSON file.
    
    Args:
        annotation_file: Path to the annotations JSON file
        
    Returns:
        A list of annotation dictionaries
    """
    with open(annotation_file, "r") as f:
        annotations = json.load(f)
    return annotations

def is_valid_pose(pose, confidence_threshold=0.3):
    """
    Check if a pose has sufficient confidence.
    
    Args:
        pose: A list of keypoints (x, y, confidence)
        confidence_threshold: Minimum average confidence
        
    Returns:
        True if the pose is valid, False otherwise
    """
    return np.mean([kp[2] for kp in pose]) > confidence_threshold

def normalize_pose(pose, img_width, img_height):
    """
    Normalize pose keypoints to 0-1 range.
    
    Args:
        pose: A list of keypoints (x, y, confidence)
        img_width: Width of the image
        img_height: Height of the image
        
    Returns:
        A list of normalized keypoints
    """
    normalized = []
    for kp in pose:
        x = float(kp[0]) / img_width  # Cast to float
        y = float(kp[1]) / img_height
        c = float(kp[2])
        normalized.append([x, y, c])
    return normalized

def compute_interaction_features(pose1, pose2):
    """
    Compute interaction features between two poses.
    
    Args:
        pose1: List of keypoints for the first athlete
        pose2: List of keypoints for the second athlete
        
    Returns:
        A list containing hip_distance, shoulder_distance, and hip_height_diff
    """
    # Keypoint indices (MS-COCO format)
    NOSE = 0
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
    LEFT_HIP, RIGHT_HIP = 11, 12
    
    # Athlete 1 features
    shoulder1 = np.mean([np.array(pose1[LEFT_SHOULDER][:2]), np.array(pose1[RIGHT_SHOULDER][:2])], axis=0)
    hip1 = np.mean([np.array(pose1[LEFT_HIP][:2]), np.array(pose1[RIGHT_HIP][:2])], axis=0)
    
    # Athlete 2 features
    shoulder2 = np.mean([np.array(pose2[LEFT_SHOULDER][:2]), np.array(pose2[RIGHT_SHOULDER][:2])], axis=0)
    hip2 = np.mean([np.array(pose2[LEFT_HIP][:2]), np.array(pose2[RIGHT_HIP][:2])], axis=0)
    
    # Interaction features - EXACTLY 3 features as in training
    hip_distance = np.linalg.norm(hip1 - hip2)
    shoulder_distance = np.linalg.norm(shoulder1 - shoulder2)
    hip_height_diff = hip1[1] - hip2[1]
    
    return [
        float(hip_distance), 
        float(shoulder_distance), 
        float(hip_height_diff)
    ]

def detect_two_bjj_athletes(image_path, visualize=False):
    """
    Detect two BJJ athletes in an image.
    
    Args:
        image_path: Path to the image
        visualize: Whether to visualize the detection
        
    Returns:
        Two lists of keypoints, one for each athlete
    """
    # Initialize MediaPipe pose
    mp_pose = mp.solutions.pose
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return None, None
        
    h, w, _ = frame.shape
    
    # Detect first athlete
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    ) as pose:
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            print("No athlete detected in the image")
            return None, None
            
        # Extract first athlete keypoints
        athlete1 = []
        for landmark in results.pose_landmarks.landmark:
            x = landmark.x * w
            y = landmark.y * h
            confidence = landmark.visibility
            athlete1.append([x, y, confidence])
        
        # Create a mask to hide the first person
        mask = np.ones((h, w), dtype=np.uint8) * 255
        if results.segmentation_mask is not None:
            mask = (1 - (results.segmentation_mask > 0.5).astype(np.uint8)) * 255
        else:
            # Create mask based on keypoints
            for i in range(len(athlete1)):
                x, y = int(athlete1[i][0]), int(athlete1[i][1])
                if athlete1[i][2] > 0.5:  # If confidence is high enough
                    cv2.circle(mask, (x, y), 30, 0, -1)  # Black circle
        
        # Apply mask to hide first athlete
        masked_frame = frame.copy()
        for c in range(3):
            masked_frame[:,:,c] = cv2.bitwise_and(masked_frame[:,:,c], mask)
    
    # Detect second athlete in the masked image
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.3  # Lower threshold for second athlete
    ) as pose:
        # Convert to RGB for MediaPipe
        masked_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(masked_rgb)
        
        if not results.pose_landmarks:
            print("Could not detect second athlete")
            return athlete1, None
            
        # Extract second athlete keypoints
        athlete2 = []
        for landmark in results.pose_landmarks.landmark:
            x = landmark.x * w
            y = landmark.y * h
            confidence = landmark.visibility
            athlete2.append([x, y, confidence])
    
    # Visualize if requested
    if visualize:
        vis_frame = frame.copy()
        
        # Draw first athlete in green
        for kp in athlete1:
            x, y, c = int(kp[0]), int(kp[1]), kp[2]
            if c > 0.3:
                cv2.circle(vis_frame, (x, y), 5, (0, 255, 0), -1)
                
        # Draw second athlete in red
        for kp in athlete2:
            x, y, c = int(kp[0]), int(kp[1]), kp[2]
            if c > 0.3:
                cv2.circle(vis_frame, (x, y), 5, (0, 0, 255), -1)
                
        # Show visualization
        cv2.imshow("Detected Athletes", vis_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return athlete1, athlete2

def split_label(label):
    """
    Split position label into position type and athlete role.
    
    Args:
        label: Position label string (e.g. "mount2")
        
    Returns:
        A tuple of (position, athlete_role)
    """
    position = ''.join([c for c in label if not c.isdigit()])
    athlete_role = int(label[-1]) if label[-1].isdigit() else 0
    return position, athlete_role