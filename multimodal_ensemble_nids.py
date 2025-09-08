#!/usr/bin/env python3
"""
Multi-Modal Deep Learning and Ensemble Machine Learning for NSL-KDD
===================================================================

This implementation demonstrates advanced techniques for intrusion detection:
1. Multi-Modal Deep Learning with different data representations
2. Ensemble Machine Learning with multiple algorithms
3. Feature fusion and cross-modal learning
4. Advanced ensemble techniques (stacking, voting, boosting)

Architecture Overview:
- Modal 1: Statistical Features (connection statistics)
- Modal 2: Temporal Features (time-based patterns) 
- Modal 3: Categorical Features (protocols, services)
- Modal 4: Host-based Features (network behavior)
- Ensemble: Combines multiple ML algorithms
"""

# Configure CUDA for GPU acceleration BEFORE importing TensorFlow
import cuda_setup
cuda_setup.check_gpu_availability()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                             GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import warnings
warnings.filterwarnings('ignore')
import os
import joblib
import json
from datetime import datetime

class MultiModalDataProcessor:
    """
    Processes NSL-KDD data into multiple modalities for multi-modal learning
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_groups = self._define_feature_groups()
        
    def _define_feature_groups(self):
        """Define different modalities based on feature types"""
        return {
            'statistical': [
                'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent',
                'hot', 'num_failed_logins', 'num_compromised', 'num_root',
                'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds'
            ],
            'temporal': [
                'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
                'diff_srv_rate', 'srv_diff_host_rate'
            ],
            'categorical': [
                'protocol_type', 'service', 'flag'
            ],
            'host_based': [
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
            ],
            'binary': [
                'land', 'logged_in', 'root_shell', 'su_attempted', 'is_host_login', 'is_guest_login'
            ]
        }
    
    def load_and_prepare_data(self, train_file, test_file):
        """Load and prepare multi-modal data"""
        print("Loading NSL-KDD dataset...")
        
        # Define column names
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
        ]
        
        # Load data
        train_df = pd.read_csv(train_file, names=columns, header=None)
        test_df = pd.read_csv(test_file, names=columns, header=None)
        
        print(f"Train data: {train_df.shape}")
        print(f"Test data: {test_df.shape}")
        
        # Create binary labels (normal vs attack)
        train_df['is_attack'] = (train_df['attack_type'] != 'normal').astype(int)
        test_df['is_attack'] = (test_df['attack_type'] != 'normal').astype(int)
        
        return train_df, test_df
    
    def create_multi_modal_features(self, df, fit_transform=True):
        """Create different modality representations"""
        modalities = {}
        
        for modality_name, features in self.feature_groups.items():
            print(f"Processing {modality_name} modality...")
            
            # Extract features that exist in the dataframe
            available_features = [f for f in features if f in df.columns]
            
            if not available_features:
                continue
                
            modal_data = df[available_features].copy()
            
            if modality_name == 'categorical':
                # One-hot encode categorical features
                if fit_transform:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(modal_data)
                    self.encoders[modality_name] = encoder
                else:
                    encoded_data = self.encoders[modality_name].transform(modal_data)
                
                modalities[modality_name] = encoded_data
                
            else:
                # Scale numerical features
                if fit_transform:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(modal_data)
                    self.scalers[modality_name] = scaler
                else:
                    scaled_data = self.scalers[modality_name].transform(modal_data)
                
                modalities[modality_name] = scaled_data
        
        return modalities

class MultiModalDeepLearning:
    """
    Multi-Modal Deep Learning architecture for intrusion detection
    """
    
    def __init__(self, modality_shapes, num_classes=2):
        self.modality_shapes = modality_shapes
        self.num_classes = num_classes
        self.model = None
        
    def build_modal_networks(self):
        """Build separate networks for each modality"""
        modal_inputs = {}
        modal_features = {}
        
        for modality_name, input_shape in self.modality_shapes.items():
            # Input layer for this modality
            modal_input = keras.Input(shape=(input_shape,), name=f'{modality_name}_input')
            modal_inputs[modality_name] = modal_input
            
            # Modality-specific processing
            if modality_name == 'statistical':
                # Dense layers for statistical features
                x = layers.Dense(128, activation='relu', name=f'{modality_name}_dense1')(modal_input)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(64, activation='relu', name=f'{modality_name}_dense2')(x)
                
            elif modality_name == 'temporal':
                # Process temporal patterns
                x = layers.Dense(96, activation='relu', name=f'{modality_name}_dense1')(modal_input)
                x = layers.Dropout(0.2)(x)
                x = layers.Dense(48, activation='relu', name=f'{modality_name}_dense2')(x)
                
            elif modality_name == 'categorical':
                # Embedding-like processing for categorical
                x = layers.Dense(64, activation='relu', name=f'{modality_name}_dense1')(modal_input)
                x = layers.Dropout(0.2)(x)
                x = layers.Dense(32, activation='relu', name=f'{modality_name}_dense2')(x)
                
            elif modality_name == 'host_based':
                # Host behavior analysis
                x = layers.Dense(80, activation='relu', name=f'{modality_name}_dense1')(modal_input)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(40, activation='relu', name=f'{modality_name}_dense2')(x)
                
            else:  # binary features
                x = layers.Dense(32, activation='relu', name=f'{modality_name}_dense1')(modal_input)
                x = layers.Dense(16, activation='relu', name=f'{modality_name}_dense2')(x)
            
            modal_features[modality_name] = x
        
        return modal_inputs, modal_features
    
    def build_fusion_network(self, modal_features):
        """Build fusion network to combine modality features"""
        
        # Concatenate all modality features
        if len(modal_features) > 1:
            fused_features = layers.Concatenate(name='feature_fusion')(list(modal_features.values()))
        else:
            fused_features = list(modal_features.values())[0]
        
        # Cross-modal attention mechanism
        attention_weights = layers.Dense(fused_features.shape[-1], activation='softmax', name='attention')(fused_features)
        attended_features = layers.Multiply(name='attended_features')([fused_features, attention_weights])
        
        # Final classification layers
        x = layers.Dense(128, activation='relu', name='fusion_dense1')(attended_features)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu', name='fusion_dense2')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        if self.num_classes == 2:
            output = layers.Dense(1, activation='sigmoid', name='output')(x)
        else:
            output = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        return output
    
    def build_model(self):
        """Build complete multi-modal model"""
        print("Building multi-modal deep learning model...")
        
        # Build modality-specific networks
        modal_inputs, modal_features = self.build_modal_networks()
        
        # Build fusion network
        output = self.build_fusion_network(modal_features)
        
        # Create model
        self.model = Model(inputs=list(modal_inputs.values()), outputs=output, name='multimodal_nids')
        
        # Compile model
        if self.num_classes == 2:
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            self.model.compile(
                optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        print(f"Model built with {self.model.count_params():,} parameters")
        return self.model

class EnsembleMachineLearning:
    """
    Advanced Ensemble Machine Learning for intrusion detection
    """
    
    def __init__(self):
        self.base_models = {}
        self.ensemble_models = {}
        self.scaler = StandardScaler()
        
    def create_base_models(self):
        """Create diverse base models for ensemble"""
        base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            ),
            'svm': SVC(
                kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), max_iter=500, 
                learning_rate_init=0.001, random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=1.0, max_iter=1000, random_state=42
            )
        }
        
        return base_models
    
    def create_ensemble_models(self, base_models):
        """Create different types of ensemble models"""
        ensemble_models = {}
        
        # Voting Classifier (Hard and Soft voting)
        voting_models = [(name, model) for name, model in base_models.items() 
                        if name != 'svm']  # Exclude SVM for speed
        
        ensemble_models['hard_voting'] = VotingClassifier(
            estimators=voting_models, voting='hard'
        )
        
        ensemble_models['soft_voting'] = VotingClassifier(
            estimators=voting_models, voting='soft'
        )
        
        # Stacking Classifier would go here (requires more setup)
        # For now, we'll use a simple weighted ensemble
        
        return ensemble_models
    
    def train_base_models(self, X_train, y_train):
        """Train all base models"""
        print("Training base models...")
        
        base_models = self.create_base_models()
        trained_models = {}
        
        for name, model in base_models.items():
            print(f"  Training {name}...")
            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                print(f"    CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"    Error training {name}: {e}")
        
        self.base_models = trained_models
        return trained_models
    
    def train_ensemble_models(self, X_train, y_train):
        """Train ensemble models"""
        print("Training ensemble models...")
        
        ensemble_models = self.create_ensemble_models(self.base_models)
        trained_ensembles = {}
        
        for name, model in ensemble_models.items():
            print(f"  Training {name}...")
            try:
                model.fit(X_train, y_train)
                trained_ensembles[name] = model
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                print(f"    CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"    Error training {name}: {e}")
        
        self.ensemble_models = trained_ensembles
        return trained_ensembles
    
    def predict_with_ensemble(self, X_test):
        """Make predictions using all models and combine them"""
        predictions = {}
        
        # Base model predictions
        for name, model in self.base_models.items():
            try:
                pred = model.predict(X_test)
                pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                predictions[name] = {'pred': pred, 'proba': pred_proba}
            except Exception as e:
                print(f"Prediction error for {name}: {e}")
        
        # Ensemble model predictions
        for name, model in self.ensemble_models.items():
            try:
                pred = model.predict(X_test)
                pred_proba = model.predict_proba(X_test)[:, 1]
                predictions[f'ensemble_{name}'] = {'pred': pred, 'proba': pred_proba}
            except Exception as e:
                print(f"Prediction error for ensemble {name}: {e}")
        
        return predictions

def main():
    """Main execution function"""
    print("Multi-Modal Deep Learning and Ensemble ML for NSL-KDD")
    print("=" * 60)
    
    # Initialize processors
    data_processor = MultiModalDataProcessor()
    
    # Load data
    train_df, test_df = data_processor.load_and_prepare_data(
        'data/raw/KDDTrain+.txt',
        'data/raw/KDDTest+.txt'
    )
    
    # Create multi-modal features
    print("\nCreating multi-modal features...")
    train_modalities = data_processor.create_multi_modal_features(train_df, fit_transform=True)
    test_modalities = data_processor.create_multi_modal_features(test_df, fit_transform=False)
    
    # Prepare labels
    y_train = train_df['is_attack'].values
    y_test = test_df['is_attack'].values
    
    print(f"\nDataset Summary:")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Attack ratio (train): {y_train.mean():.3f}")
    print(f"Attack ratio (test): {y_test.mean():.3f}")
    
    # Multi-Modal Deep Learning
    print(f"\n{'='*60}")
    print("MULTI-MODAL DEEP LEARNING")
    print("=" * 60)
    
    modality_shapes = {name: data.shape[1] for name, data in train_modalities.items()}
    print(f"Modality shapes: {modality_shapes}")
    
    # Build and train multi-modal model
    mm_model = MultiModalDeepLearning(modality_shapes)
    model = mm_model.build_model()
    
    print(f"\nModel Architecture:")
    model.summary()
    
    # For demonstration, we'll train on a subset due to computational constraints
    sample_size = min(10000, len(y_train))
    train_indices = np.random.choice(len(y_train), sample_size, replace=False)
    
    train_modal_sample = [data[train_indices] for data in train_modalities.values()]
    y_train_sample = y_train[train_indices]
    
    print(f"\nTraining multi-modal model on {sample_size} samples...")
    history = model.fit(
        train_modal_sample, y_train_sample,
        batch_size=256, epochs=10, validation_split=0.2,
        verbose=1
    )
    
    # Ensemble Machine Learning
    print(f"\n{'='*60}")
    print("ENSEMBLE MACHINE LEARNING") 
    print("=" * 60)
    
    # Combine all modalities for traditional ML
    X_train_combined = np.hstack(list(train_modalities.values()))
    X_test_combined = np.hstack(list(test_modalities.values()))
    
    print(f"Combined feature matrix: {X_train_combined.shape}")
    
    # Initialize ensemble learning
    ensemble_ml = EnsembleMachineLearning()
    
    # Scale data
    X_train_scaled = ensemble_ml.scaler.fit_transform(X_train_combined)
    X_test_scaled = ensemble_ml.scaler.transform(X_test_combined)
    
    # Train models on sample for demonstration
    train_sample_ml = X_train_scaled[train_indices]
    
    # Train base models
    trained_base = ensemble_ml.train_base_models(train_sample_ml, y_train_sample)
    
    # Train ensemble models
    trained_ensembles = ensemble_ml.train_ensemble_models(train_sample_ml, y_train_sample)
    
    print(f"\n{'='*60}")
    print("IMPLEMENTATION COMPLETED!")
    print("=" * 60)
    print(f"✅ Multi-Modal Deep Learning: {len(modality_shapes)} modalities processed")
    print(f"✅ Ensemble Learning: {len(trained_base)} base models + {len(trained_ensembles)} ensemble models")
    print(f"✅ Ready for full-scale training and evaluation")
    
    # Save all trained models
    print(f"\n{'='*60}")
    print("SAVING TRAINED MODELS")
    print("=" * 60)
    save_all_models(model, trained_base, trained_ensembles, data_processor, modality_shapes)

def save_all_models(dl_model, base_models, ensemble_models, data_processor, modality_shapes):
    """Save all trained models and preprocessing objects"""
    
    # Create models directory
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Saving models to: {model_dir}")
    
    # 1. Save TensorFlow multimodal model
    tf_model_path = os.path.join(model_dir, "multimodal_deep_learning")
    dl_model.save(tf_model_path)
    print(f"✅ Saved TensorFlow model: {tf_model_path}")
    
    # 2. Save sklearn base models
    base_models_dir = os.path.join(model_dir, "base_models")
    os.makedirs(base_models_dir, exist_ok=True)
    for name, model in base_models.items():
        model_path = os.path.join(base_models_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Saved {name}: {model_path}")
    
    # 3. Save ensemble models
    ensemble_dir = os.path.join(model_dir, "ensemble_models") 
    os.makedirs(ensemble_dir, exist_ok=True)
    for name, model in ensemble_models.items():
        model_path = os.path.join(ensemble_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Saved ensemble {name}: {model_path}")
    
    # 4. Save preprocessing objects
    preprocessing_dir = os.path.join(model_dir, "preprocessing")
    os.makedirs(preprocessing_dir, exist_ok=True)
    
    # Save scalers
    scalers_path = os.path.join(preprocessing_dir, "scalers.pkl")
    joblib.dump(data_processor.scalers, scalers_path)
    print(f"✅ Saved scalers: {scalers_path}")
    
    # Save encoders
    encoders_path = os.path.join(preprocessing_dir, "encoders.pkl")
    joblib.dump(data_processor.encoders, encoders_path)
    print(f"✅ Saved encoders: {encoders_path}")
    
    # 5. Save configuration and metadata
    config = {
        "modality_shapes": modality_shapes,
        "feature_groups": data_processor.feature_groups,
        "model_info": {
            "tensorflow_model": "multimodal_deep_learning (SavedModel format)",
            "base_models": list(base_models.keys()),
            "ensemble_models": list(ensemble_models.keys()),
            "total_parameters": dl_model.count_params()
        }
    }
    
    config_path = os.path.join(model_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved configuration: {config_path}")
    
    # 6. Create loading instructions
    instructions = f"""
# Loading Instructions for Multimodal Ensemble NIDS Models

## Load TensorFlow Model:
```python
import tensorflow as tf
model = tf.keras.models.load_model('{tf_model_path}')
```

## Load Sklearn Models:
```python
import joblib
base_models = {{}}
for model_name in {list(base_models.keys())}:
    base_models[model_name] = joblib.load(f'{base_models_dir}/{{model_name}}.pkl')

ensemble_models = {{}}
for model_name in {list(ensemble_models.keys())}:
    ensemble_models[model_name] = joblib.load(f'{ensemble_dir}/{{model_name}}.pkl')
```

## Load Preprocessing:
```python
scalers = joblib.load('{scalers_path}')
encoders = joblib.load('{encoders_path}')
```

Total Models Saved: {len(base_models) + len(ensemble_models) + 1}
Total Size: ~{estimate_model_size(dl_model, base_models, ensemble_models)} MB
"""
    
    instructions_path = os.path.join(model_dir, "loading_instructions.md")
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    print(f"✅ Saved loading instructions: {instructions_path}")
    
    print(f"\n🎉 ALL MODELS SAVED SUCCESSFULLY!")
    print(f"📂 Location: {os.path.abspath(model_dir)}")
    print(f"📊 Total files: {count_files_in_directory(model_dir)}")

def estimate_model_size(dl_model, base_models, ensemble_models):
    """Rough estimate of total model size"""
    # TensorFlow model: ~0.5-2MB for this architecture
    tf_size = dl_model.count_params() * 4 / (1024*1024)  # 4 bytes per float32 parameter
    
    # Sklearn models: varies widely
    sklearn_size = len(base_models) * 2 + len(ensemble_models) * 5  # rough estimate
    
    return round(tf_size + sklearn_size, 1)

def count_files_in_directory(directory):
    """Count total files in directory recursively"""
    total = 0
    for root, dirs, files in os.walk(directory):
        total += len(files)
    return total

if __name__ == "__main__":
    main()
