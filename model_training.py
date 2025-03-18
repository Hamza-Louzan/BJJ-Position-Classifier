#!/usr/bin/env python3
'''
Model Training for BJJ Position Detection
'''

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from pose_utils import split_label, compute_interaction_features

def prepare_data(annotations):
    """
    Prepare data for model training.
    
    Args:
        annotations: List of annotation dictionaries
        
    Returns:
        X_train, X_test, y_position_train, y_position_test, scaler, label_encoder
    """
    # Update annotations with position type and athlete role
    for ann in annotations:
        ann["PositionType"], ann["AthleteRole"] = split_label(ann["position"])
    
    # Extract video sequence groups for proper train/test splitting
    groups = [ann["image"][:2] for ann in annotations]
    
    # Split data into training and testing sets
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(annotations, groups=groups))
    
    train_data = [annotations[i] for i in train_idx]
    test_data = [annotations[i] for i in test_idx]
    
    print(f"Training data: {len(train_data)} samples")
    print(f"Testing data: {len(test_data)} samples")
    
    # Prepare training data
    X_train = []
    y_position_train = []
    
    for ann in train_data:
        # Ensure all required keys are present
        if all(key in ann for key in ["pose1", "pose2", "PositionType"]):
            # Flatten poses
            pose1_flat = [coord for kp in ann["pose1"] for coord in kp]
            pose2_flat = [coord for kp in ann["pose2"] for coord in kp]
            
            # Compute interaction features
            interaction_features = compute_interaction_features(ann["pose1"], ann["pose2"])
            
            # Combine all features
            combined = pose1_flat + pose2_flat + interaction_features
            X_train.append(combined)
            y_position_train.append(ann["PositionType"])
    
    # Prepare testing data
    X_test = []
    y_position_test = []
    
    for ann in test_data:
        if all(key in ann for key in ["pose1", "pose2", "PositionType"]):
            pose1_flat = [coord for kp in ann["pose1"] for coord in kp]
            pose2_flat = [coord for kp in ann["pose2"] for coord in kp]
            interaction_features = compute_interaction_features(ann["pose1"], ann["pose2"])
            
            combined = pose1_flat + pose2_flat + interaction_features
            X_test.append(combined)
            y_position_test.append(ann["PositionType"])
    
    # Convert to numpy arrays
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_position_train = label_encoder.fit_transform(y_position_train)
    y_position_test = label_encoder.transform(y_position_test)
    
    # Compute class weights to handle imbalance
    class_counts = np.bincount(y_position_train)
    total_samples = len(y_position_train)
    
    position_class_weights = {}
    for i in range(len(class_counts)):
        if class_counts[i] > 0:
            weight = total_samples / (len(class_counts) * class_counts[i])
            position_class_weights[i] = weight
    
    return (X_train_scaled, X_test_scaled, y_position_train, y_position_test, 
            scaler, label_encoder, position_class_weights)

def build_model(input_shape, num_classes):
    """
    Build a neural network model for BJJ position classification.
    
    Args:
        input_shape: Shape of the input features
        num_classes: Number of classes to predict
        
    Returns:
        Compiled Keras model
    """
    # Define model architecture
    input_layer = Input(shape=(input_shape,))
    
    # Add batch normalization to input
    x = BatchNormalization()(input_layer)
    
    # First hidden layer with L2 regularization and dropout
    x = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second hidden layer
    x = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Third hidden layer
    x = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    output_position = Dense(num_classes, activation="softmax", name='position')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_position)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_bjj_model(annotations, output_dir):
    """
    Train a BJJ position detection model.
    
    Args:
        annotations: List of annotation dictionaries
        output_dir: Directory to save model and artifacts
    """
    # Prepare data
    (X_train_scaled, X_test_scaled, y_position_train, y_position_test, 
     scaler, label_encoder, position_class_weights) = prepare_data(annotations)
    
    # Build model
    input_shape = X_train_scaled.shape[1]
    num_classes = len(np.unique(y_position_train))
    model = build_model(input_shape, num_classes)
    
    # Print model summary
    model.summary()
    
    # Set up callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(output_dir, 'best_bjj_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train_scaled,
        y_position_train,
        epochs=100,  # We'll use early stopping
        batch_size=32,
        validation_split=0.15,
        class_weight=position_class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Evaluate on test data
    test_loss, test_acc = model.evaluate(X_test_scaled, y_position_test)
    print(f'Test accuracy: {test_acc:.4f}')
    
    # Confusion matrix and classification report
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_position_test, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)
    
    # Display class names in classification report
    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_position_test, y_pred_classes, 
                              labels=np.arange(len(class_names)), 
                              target_names=class_names))
    
    # Save the model, scaler, and label encoder
    model.save(os.path.join(output_dir, 'bjj_position_model.h5'))
    
    with open(os.path.join(output_dir, 'bjj_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    with open(os.path.join(output_dir, 'bjj_label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Model, scaler, and label encoder saved to {output_dir}")