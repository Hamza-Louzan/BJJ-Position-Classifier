#!/usr/bin/env python3
'''
Main script for BJJ Position Detection
This script orchestrates the different components of the BJJ position detection system.
'''

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from pose_utils import load_annotations, is_valid_pose, normalize_pose
from model_training import train_bjj_model
from inference import predict_bjj_position, visualize_prediction

def main():
    parser = argparse.ArgumentParser(description='BJJ Position Detection')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--annotations', type=str, default='data/annotations.json',
                             help='Path to annotations file')
    train_parser.add_argument('--images', type=str, default='data/images/',
                             help='Path to images directory')
    train_parser.add_argument('--output', type=str, default='./',
                             help='Directory to save the model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run prediction on an image')
    predict_parser.add_argument('--image', type=str, required=True,
                               help='Path to image for prediction')
    predict_parser.add_argument('--model', type=str, default='bjj_position_model.h5',
                               help='Path to trained model')
    predict_parser.add_argument('--scaler', type=str, default='bjj_scaler.pkl',
                               help='Path to fitted scaler')
    predict_parser.add_argument('--encoder', type=str, default='bjj_label_encoder.pkl',
                               help='Path to label encoder')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Load annotations
        print(f"Loading annotations from {args.annotations}")
        annotations = load_annotations(args.annotations)
        
        # Get sample image to determine dimensions
        image_files = list(Path(args.images).glob('*.jpg'))
        if not image_files:
            print(f"No images found in {args.images}")
            return
            
        sample_img = cv2.imread(str(image_files[0]))
        if sample_img is None:
            print(f"Could not read sample image {image_files[0]}")
            return
            
        img_height, img_width, _ = sample_img.shape
        
        # Filter and normalize poses
        valid_annotations = []
        for ann in annotations:
            if "pose1" in ann and "pose2" in ann and is_valid_pose(ann["pose1"]) and is_valid_pose(ann["pose2"]):
                valid_annotations.append(ann)
                
        for ann in valid_annotations:
            ann["pose1"] = normalize_pose(ann["pose1"], img_width, img_height)
            ann["pose2"] = normalize_pose(ann["pose2"], img_width, img_height)
        
        # Train model
        train_bjj_model(valid_annotations, args.output)
        
    elif args.command == 'predict':
        # Check if image exists
        if not os.path.exists(args.image):
            print(f"Image file not found: {args.image}")
            return
            
        # Check if model files exist
        if not all(os.path.exists(f) for f in [args.model, args.scaler, args.encoder]):
            print("Model files not found. Please ensure model, scaler, and encoder files exist.")
            return
            
        # Run prediction
        position, confidence = predict_bjj_position(args.image, args.model, args.scaler, args.encoder)
        
        if position:
            print(f"Predicted position: {position} with {confidence:.2f}% confidence")
            
            # Visualize prediction
            visualize_prediction(args.image, position, confidence)
        else:
            print("Could not predict BJJ position")

if __name__ == "__main__":
    main()