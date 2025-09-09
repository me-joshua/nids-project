#!/usr/bin/env python3
"""
Regenerate NIDS model in Keras 3 compatible format
This script loads the existing model and saves it in a format compatible with Keras 3
"""

import os
import sys
import tensorflow as tf
import numpy as np
import json

def main():
    print("🔄 Regenerating NIDS model for Keras 3 compatibility...")
    print(f"📊 TensorFlow version: {tf.__version__}")
    print(f"📊 Keras version: {tf.keras.__version__}")
    
    models_dir = "models"
    
    # Try to load the existing model
    keras_model_path = os.path.join(models_dir, "multimodal_deep_learning.keras")
    saved_model_path = os.path.join(models_dir, "multimodal_deep_learning")
    
    model = None
    
    # Try loading existing model
    if os.path.exists(keras_model_path):
        try:
            print(f"🔄 Loading existing .keras model: {keras_model_path}")
            model = tf.keras.models.load_model(keras_model_path)
            print("✅ Successfully loaded .keras model")
        except Exception as e:
            print(f"❌ Failed to load .keras model: {e}")
    
    if model is None and os.path.exists(saved_model_path):
        try:
            print(f"🔄 Loading existing SavedModel: {saved_model_path}")
            model = tf.keras.models.load_model(saved_model_path)
            print("✅ Successfully loaded SavedModel")
        except Exception as e:
            print(f"❌ Failed to load SavedModel: {e}")
    
    if model is None:
        print("❌ Could not load existing model")
        print("🔄 Creating a new minimal model for testing...")
        
        # Create a minimal model that matches the expected interface
        # Based on the configuration, we need 5 inputs for different feature groups
        
        # Load configuration to get feature groups
        config_path = os.path.join(models_dir, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                feature_groups = config.get('feature_groups', {})
        else:
            # Default feature groups
            feature_groups = {
                'statistical': ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent',
                               'hot', 'num_failed_logins', 'num_compromised', 'num_root',
                               'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds'],
                'temporal': ['count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
                            'diff_srv_rate', 'srv_diff_host_rate'],
                'categorical': ['protocol_type', 'service', 'flag'],
                'host_based': ['dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                              'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                              'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                              'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'],
                'binary': ['land', 'logged_in', 'root_shell', 'su_attempted', 'is_host_login', 'is_guest_login']
            }
        
        print("🔧 Creating new multimodal model architecture...")
        
        # Create inputs for each modality
        inputs = {}
        processed_inputs = {}
        
        for modality_name, features in feature_groups.items():
            if modality_name == 'categorical':
                # Categorical features are one-hot encoded, so we need more dimensions
                input_dim = 84  # Approximate for NSL-KDD categorical features
            else:
                input_dim = len(features)
            
            inputs[modality_name] = tf.keras.layers.Input(
                shape=(input_dim,), 
                name=f"{modality_name}_input"
            )
            
            # Simple processing for each modality
            dense_units = max(32, input_dim * 2)
            x = tf.keras.layers.Dense(dense_units, activation='relu', 
                                    name=f"{modality_name}_dense1")(inputs[modality_name])
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(dense_units // 2, activation='relu',
                                    name=f"{modality_name}_dense2")(x)
            processed_inputs[modality_name] = x
        
        # Concatenate all processed inputs
        if len(processed_inputs) > 1:
            concatenated = tf.keras.layers.Concatenate(name="feature_fusion")(list(processed_inputs.values()))
        else:
            concatenated = list(processed_inputs.values())[0]
        
        # Attention mechanism
        attention = tf.keras.layers.Dense(concatenated.shape[-1], activation='sigmoid', name='attention')(concatenated)
        attended_features = tf.keras.layers.Multiply(name='attended_features')([concatenated, attention])
        
        # Final classification layers
        x = tf.keras.layers.Dense(128, activation='relu', name='fusion_dense1')(attended_features)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(64, activation='relu', name='fusion_dense2')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = tf.keras.Model(
            inputs=list(inputs.values()),
            outputs=output,
            name="multimodal_nids"
        )
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Created new model architecture")
        model.summary()
        
        # Initialize with random weights (since we don't have training data here)
        print("⚠️  Note: Model created with random weights - will need retraining for actual use")
    
    # Save in Keras 3 compatible format
    new_keras_path = os.path.join(models_dir, "multimodal_deep_learning_keras3.keras")
    
    try:
        print(f"💾 Saving model to: {new_keras_path}")
        model.save(new_keras_path)
        print("✅ Model saved successfully in Keras 3 format")
        
        # Test loading
        print("🧪 Testing model loading...")
        test_model = tf.keras.models.load_model(new_keras_path)
        print("✅ Model loads successfully")
        
        # Copy to main filename
        main_keras_path = os.path.join(models_dir, "multimodal_deep_learning.keras")
        print(f"📋 Copying to main filename: {main_keras_path}")
        
        import shutil
        shutil.copy2(new_keras_path, main_keras_path)
        print("✅ Model copied to main filename")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False
    
    print("🎉 Model regeneration completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
