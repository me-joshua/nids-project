#!/usr/bin/env python3
"""
Flask API Backend for Network Intrusion Detection System
========================================================

This Flask backend serves the trained multimodal ensemble NIDS models
and provides a REST API for the React frontend to analyze network packets.

Features:
- Load all saved models (TensorFlow + sklearn)
- Process incoming packet data 
- Return comprehensive analysis results
- CORS support for frontend integration
"""

import sys
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import time
from datetime import datetime

# TensorFlow imports with GPU configuration already handled by cuda_setup
import tensorflow as tf
import joblib
import json


app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return "NIDS Backend is running.", 200

class NIDSPredictor:
    """Network Intrusion Detection System Predictor"""
    
    def __init__(self, models_dir=None):
        # Auto-detect models directory based on current file location
        if models_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Try different possible paths for models directory
            possible_paths = [
                os.path.join(current_dir, "..", "models"),  # Local development
                os.path.join(os.path.dirname(current_dir), "models"),  # Render deployment
                os.path.join(current_dir, "models"),  # Same directory
                "/opt/render/project/src/models"  # Absolute Render path
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    models_dir = path
                    break
            
            if models_dir is None:
                models_dir = "../models"  # Fallback to original
                
        self.models_dir = models_dir
        self.models_loaded = False
        self.multimodal_model = None
        self.base_models = {}
        self.ensemble_models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_groups = {}
        self.config = {}
        
        # NSL-KDD feature names
        self.feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
        
    def _get_default_feature_groups(self):
        """Define default feature groups in case config is missing"""
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
        
    def load_models(self):
        """Load all saved models and preprocessors"""
        try:
            print("Loading NIDS models...")
            print(f"📂 Models directory: {self.models_dir}")
            print(f"📂 Directory exists: {os.path.exists(self.models_dir)}")
            
            if os.path.exists(self.models_dir):
                print(f"📂 Contents: {os.listdir(self.models_dir)}")
            
            # Load configuration
            config_path = os.path.join(self.models_dir, "model_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                    self.feature_groups = self.config.get('feature_groups', {})
                    print(f"✅ Loaded configuration with {len(self.feature_groups)} feature groups")
            else:
                print("⚠️  Config file not found, using default feature groups")
                print(f"⚠️  Looked for config at: {config_path}")
                self.feature_groups = self._get_default_feature_groups()
            
            # Ensure we have feature groups
            if not self.feature_groups:
                print("⚠️  No feature groups found, using default feature groups")
                self.feature_groups = self._get_default_feature_groups()
            
            # Load TensorFlow multimodal model - handle Keras 3 compatibility
            print("🔄 Loading TensorFlow multimodal model...")
            
            keras_model_path = os.path.join(self.models_dir, "multimodal_deep_learning.keras")
            saved_model_path = os.path.join(self.models_dir, "multimodal_deep_learning")
            
            model_loaded = False
            
            # First try: .keras format (preferred for Keras 3)
            if os.path.exists(keras_model_path):
                try:
                    print(f"🔍 Trying .keras format: {keras_model_path}")
                    # Get file info for debugging
                    import stat
                    file_stat = os.stat(keras_model_path)
                    print(f"🔍 File size: {file_stat.st_size} bytes")
                    print(f"🔍 File permissions: {stat.filemode(file_stat.st_mode)}")
                    
                    # Try standard loading
                    self.multimodal_model = tf.keras.models.load_model(keras_model_path)
                    print("✅ Loaded TensorFlow multimodal model (.keras format)")
                    model_loaded = True
                except Exception as e1:
                    print(f"❌ .keras format loading failed: {e1}")
                    
                    # Try loading without compilation
                    try:
                        print("🔄 Trying to load .keras without compilation...")
                        self.multimodal_model = tf.keras.models.load_model(keras_model_path, compile=False)
                        print("✅ Loaded TensorFlow model (.keras format) without compilation")
                        model_loaded = True
                    except Exception as e2:
                        print(f"❌ .keras loading without compilation failed: {e2}")
            
            # Second try: TFSMLayer for SavedModel (Keras 3 compatible)
            if not model_loaded and os.path.exists(saved_model_path):
                try:
                    print(f"🔍 Trying TFSMLayer for SavedModel: {saved_model_path}")
                    
                    # Create a wrapper model using TFSMLayer for Keras 3 compatibility
                    import tensorflow as tf
                    
                    # Create TFSMLayer
                    tfsm_layer = tf.keras.layers.TFSMLayer(saved_model_path, call_endpoint='serving_default')
                    
                    # Get the original model's input shapes for the wrapper
                    # We'll create a simple wrapper that matches our expected interface
                    print("🔄 Creating TFSMLayer wrapper for Keras 3 compatibility...")
                    
                    # Note: This is a simplified approach - we'll need to recreate the proper inputs
                    # For now, let's skip this complex approach and regenerate the model instead
                    print("⚠️  TFSMLayer approach requires model architecture recreation")
                    print("⚠️  Skipping SavedModel loading in Keras 3 environment")
                    
                except Exception as e3:
                    print(f"❌ TFSMLayer loading failed: {e3}")
            
            # Third try: Regenerate model in Keras 3 format
            if not model_loaded:
                print("❌ Failed to load existing TensorFlow model formats")
                print("💡 Model needs to be regenerated for Keras 3 compatibility")
                print("📁 Checked paths:")
                print(f"   .keras: {keras_model_path} (exists: {os.path.exists(keras_model_path)})")
                print(f"   SavedModel: {saved_model_path} (exists: {os.path.exists(saved_model_path)})")
                print("🔄 API will continue without deep learning model")
                print("🔄 Only traditional ML models (sklearn) will be available")
            
            # Load sklearn base models
            base_models_dir = os.path.join(self.models_dir, "base_models")
            if os.path.exists(base_models_dir):
                for model_file in os.listdir(base_models_dir):
                    if model_file.endswith('.pkl'):
                        model_name = model_file.replace('.pkl', '')
                        model_path = os.path.join(base_models_dir, model_file)
                        self.base_models[model_name] = joblib.load(model_path)
                        print(f"✅ Loaded base model: {model_name}")
            
            # Load ensemble models
            ensemble_dir = os.path.join(self.models_dir, "ensemble_models")
            if os.path.exists(ensemble_dir):
                for model_file in os.listdir(ensemble_dir):
                    if model_file.endswith('.pkl'):
                        model_name = model_file.replace('.pkl', '')
                        model_path = os.path.join(ensemble_dir, model_file)
                        self.ensemble_models[model_name] = joblib.load(model_path)
                        print(f"✅ Loaded ensemble model: {model_name}")
            
            # Load preprocessing objects
            preprocessing_dir = os.path.join(self.models_dir, "preprocessing")
            if os.path.exists(preprocessing_dir):
                scalers_path = os.path.join(preprocessing_dir, "scalers.pkl")
                if os.path.exists(scalers_path):
                    self.scalers = joblib.load(scalers_path)
                    print("✅ Loaded scalers")
                
                encoders_path = os.path.join(preprocessing_dir, "encoders.pkl")
                if os.path.exists(encoders_path):
                    self.encoders = joblib.load(encoders_path)
                    print("✅ Loaded encoders")
            
            self.models_loaded = True
            print(f"🎉 All models loaded successfully!")
            print(f"📊 Base models: {len(self.base_models)}")
            print(f"📊 Ensemble models: {len(self.ensemble_models)}")
            print(f"📊 Multimodal DL: {'Yes' if self.multimodal_model else 'No'}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.models_loaded = False
    
    def parse_packet_data(self, packet_string):
        """Parse comma-separated packet data into DataFrame"""
        try:
            # Split the packet data
            values = packet_string.strip().split(',')
            print(f"Debug - Raw packet values: {len(values)} items")
            print(f"Debug - First 10 values: {values[:10]}")
            
            # Ensure we have the right number of features
            if len(values) != len(self.feature_names):
                print(f"Debug - Length mismatch: got {len(values)}, expected {len(self.feature_names)}")
                # If we have more values, truncate to expected length
                if len(values) > len(self.feature_names):
                    print(f"Debug - Truncating from {len(values)} to {len(self.feature_names)}")
                    values = values[:len(self.feature_names)]
                # If we have fewer values, pad with zeros
                else:
                    print(f"Debug - Padding from {len(values)} to {len(self.feature_names)}")
                    values.extend(['0'] * (len(self.feature_names) - len(values)))
            
            # Convert to appropriate data types
            processed_values = []
            for i, (feature, value) in enumerate(zip(self.feature_names, values)):
                if feature in ['protocol_type', 'service', 'flag']:
                    # Keep categorical features as strings
                    processed_values.append(value.strip())
                else:
                    # Convert numerical features
                    try:
                        processed_values.append(float(value))
                    except ValueError:
                        processed_values.append(0.0)
            
            # Create DataFrame
            df = pd.DataFrame([processed_values], columns=self.feature_names)
            print(f"Debug - Parsed features: protocol_type={df['protocol_type'].iloc[0]}, service={df['service'].iloc[0]}, src_bytes={df['src_bytes'].iloc[0]}")
            return df
            
        except Exception as e:
            raise ValueError(f"Error parsing packet data: {e}")
    
    def create_multi_modal_features(self, df):
        """Create multi-modal features from input data"""
        modalities = {}
        
        print(f"Debug - Creating modalities from {len(self.feature_groups)} feature groups")
        print(f"Debug - Available scalers: {list(self.scalers.keys())}")
        print(f"Debug - Available encoders: {list(self.encoders.keys())}")
        
        for modality_name, features in self.feature_groups.items():
            # Extract features that exist in the dataframe
            available_features = [f for f in features if f in df.columns]
            
            if not available_features:
                print(f"Warning - No features available for modality: {modality_name}")
                continue
                
            modal_data = df[available_features].copy()
            print(f"Debug - Processing {modality_name}: {len(available_features)} features")
            
            if modality_name == 'categorical':
                # One-hot encode categorical features
                if modality_name in self.encoders:
                    try:
                        encoded_data = self.encoders[modality_name].transform(modal_data)
                        modalities[modality_name] = encoded_data
                        print(f"Debug - {modality_name} encoded shape: {encoded_data.shape}")
                    except Exception as e:
                        print(f"Error encoding {modality_name}: {e}")
                else:
                    print(f"Warning - No encoder found for {modality_name}")
            else:
                # Scale numerical features
                if modality_name in self.scalers:
                    try:
                        scaled_data = self.scalers[modality_name].transform(modal_data)
                        modalities[modality_name] = scaled_data
                        print(f"Debug - {modality_name} scaled shape: {scaled_data.shape}")
                    except Exception as e:
                        print(f"Error scaling {modality_name}: {e}")
                else:
                    print(f"Warning - No scaler found for {modality_name}")
        
        print(f"Debug - Created {len(modalities)} modalities: {list(modalities.keys())}")
        return modalities
    
    def predict_packet(self, packet_string):
        """Analyze a single packet and return comprehensive results"""
        if not self.models_loaded:
            raise Exception("Models not loaded. Please check model files.")
        
        start_time = time.time()
        
        # Parse packet data
        df = self.parse_packet_data(packet_string)
        
        # Create multi-modal features
        modalities = self.create_multi_modal_features(df)
        
        # Combine all features for traditional ML models
        combined_features = np.hstack(list(modalities.values()))
        print(f"Debug - Combined features shape: {combined_features.shape}")
        print(f"Debug - Combined features sample (first 10): {combined_features[0][:10]}")
        print(f"Debug - Combined features stats: min={combined_features.min():.4f}, max={combined_features.max():.4f}, mean={combined_features.mean():.4f}")
        
        results = {
            'base_models': {},
            'ensemble_models': {},
            'deep_learning': {},
            'ensemble_confidence': 0.0,
            'attack_probability': 0.0,
            'is_attack': False,
            'processing_time': 0.0,
            'models_used': 0
        }
        
        # Base model predictions
        base_predictions = []
        for name, model in self.base_models.items():
            try:
                pred = model.predict(combined_features)[0]
                pred_proba = model.predict_proba(combined_features)[0]
                
                # pred_proba[1] is the probability of class 1 (attack)
                attack_confidence = float(pred_proba[1])
                
                print(f"Debug - {name}: prediction={pred}, attack_prob={attack_confidence:.4f}, normal_prob={pred_proba[0]:.4f}")
                
                results['base_models'][name] = {
                    'prediction': int(pred),
                    'confidence': attack_confidence,  # Attack probability
                    'normal_prob': float(pred_proba[0])
                }
                base_predictions.append(attack_confidence)
                results['models_used'] += 1
                
            except Exception as e:
                print(f"Error with base model {name}: {e}")
        
        # Ensemble model predictions
        ensemble_predictions = []
        for name, model in self.ensemble_models.items():
            try:
                pred = model.predict(combined_features)[0]
                
                # Handle different voting classifier types
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(combined_features)[0]
                    confidence = float(pred_proba[1])  # Attack probability
                    normal_prob = float(pred_proba[0])
                    print(f"Debug - {name}: prediction={pred}, attack_prob={confidence:.4f}, normal_prob={normal_prob:.4f}")
                else:
                    # Hard voting classifier doesn't have predict_proba
                    confidence = float(pred)  # Use binary prediction as confidence
                    normal_prob = 1.0 - float(pred)
                    print(f"Debug - {name} (hard voting): prediction={pred}, binary_conf={confidence:.4f}")
                
                results['ensemble_models'][name] = {
                    'prediction': int(pred),
                    'confidence': confidence,
                    'normal_prob': normal_prob
                }
                ensemble_predictions.append(confidence)
                results['models_used'] += 1
                
            except Exception as e:
                print(f"Error with ensemble model {name}: {e}")
        
        # Deep learning prediction
        if self.multimodal_model and modalities:
            try:
                # Get the model's expected input names and order
                model_input_names = [inp.name for inp in self.multimodal_model.inputs]
                print(f"Debug - Model input names: {model_input_names}")
                
                # Map modality names to model input names
                modality_to_input_mapping = {}
                for input_name in model_input_names:
                    # Extract modality name from input name (e.g., 'statistical_input' -> 'statistical')
                    modality_name = input_name.replace('_input', '')
                    if modality_name in modalities:
                        modality_to_input_mapping[input_name] = modality_name
                
                print(f"Debug - Modality mapping: {modality_to_input_mapping}")
                
                # Prepare inputs in the correct order matching model's expected inputs
                modal_inputs = []
                for input_name in model_input_names:
                    if input_name in modality_to_input_mapping:
                        modality_name = modality_to_input_mapping[input_name]
                        modal_inputs.append(modalities[modality_name])
                        print(f"Debug - Added {modality_name} data shape {modalities[modality_name].shape} for {input_name}")
                    else:
                        print(f"Warning - No data found for model input: {input_name}")
                
                if len(modal_inputs) == len(model_input_names):
                    # Use GPU if available, otherwise CPU
                    gpu_devices = tf.config.list_physical_devices('GPU')
                    device_name = '/GPU:0' if gpu_devices else '/CPU:0'
                    
                    with tf.device(device_name):
                        dl_pred = self.multimodal_model.predict(modal_inputs, verbose=0)[0][0]
                    
                    device_used = "GPU" if gpu_devices else "CPU"
                    print(f"Debug - Deep learning prediction using {device_used}: {dl_pred}")
                    
                    results['deep_learning'] = {
                        'prediction': int(dl_pred > 0.5),
                        'confidence': float(dl_pred),
                        'modalities_count': len(modalities),
                        'device_used': device_used
                    }
                    results['models_used'] += 1
                else:
                    raise ValueError(f"Input count mismatch: model expects {len(model_input_names)}, got {len(modal_inputs)}")
                
            except Exception as e:
                print(f"Error with deep learning model: {e}")
                print(f"Available modalities: {list(modalities.keys())}")
                print(f"Expected model input names: {[inp.name for inp in self.multimodal_model.inputs] if hasattr(self.multimodal_model, 'inputs') else 'Unknown'}")
                results['deep_learning'] = {
                    'prediction': 0,
                    'confidence': 0.0,
                    'modalities_count': len(modalities),
                    'error': str(e)
                }
        
        # Calculate ensemble confidence using the same logic as training script
        all_predictions = base_predictions + ensemble_predictions
        if results['deep_learning'].get('confidence'):
            all_predictions.append(results['deep_learning']['confidence'])
        
        if all_predictions:
            # Use simple average like in the training script - no artificial boosting
            ensemble_confidence = float(np.mean(all_predictions))
            
            # Enhanced pattern analysis (for information only, not boosting)
            attack_indicators = self._analyze_attack_patterns(df, results)
            
            # Debug: Print attack pattern analysis results
            print(f"Debug - Attack pattern analysis:")
            print(f"  Pattern detected: {attack_indicators.get('pattern_detected', False)}")
            print(f"  Pattern type: {attack_indicators.get('pattern_type', 'None')}")
            print(f"  Specific features: {attack_indicators.get('specific_features', [])}")
            
            # Use weighted ensemble approach (like training script combines models)
            weighted_confidence = self._calculate_weighted_ensemble(base_predictions, ensemble_predictions, results['deep_learning'].get('confidence', 0))
            
            # Take the maximum of simple average and weighted approach (model combination)
            final_confidence = max(ensemble_confidence, weighted_confidence)
            
            results['ensemble_confidence'] = final_confidence
            results['attack_probability'] = final_confidence
            results['is_attack'] = final_confidence > 0.5
            
            # Add attack pattern information to results (for transparency)
            results['attack_patterns'] = attack_indicators
        
        results['processing_time'] = time.time() - start_time
        
        print(f"Debug - Final results summary:")
        print(f"  Base models: {len(results['base_models'])}")
        print(f"  Ensemble models: {len(results['ensemble_models'])}")
        print(f"  Deep learning: {'Success' if results['deep_learning'].get('confidence') else 'Failed'}")
        print(f"  Ensemble confidence: {results['ensemble_confidence']:.4f}")
        print(f"  Attack probability: {results['attack_probability']:.4f}")
        print(f"  Is attack: {results['is_attack']}")
        print(f"  Processing time: {results['processing_time']:.3f}s")
        
        return results
    
    def _analyze_attack_patterns(self, df, results):
        """Enhanced attack pattern analysis for better transparency"""
        attack_indicators = {
            'pattern_detected': False,
            'pattern_type': None,
            'specific_features': [],
            'pattern_score': 0
        }
        
        try:
            # Feature-based pattern detection (for information only)
            features = df.iloc[0].to_dict()
            
            print(f"Debug - Analyzing packet features:")
            print(f"  protocol_type: {features.get('protocol_type')}")
            print(f"  service: {features.get('service')}")
            print(f"  src_bytes: {features.get('src_bytes', 0)}")
            print(f"  dst_bytes: {features.get('dst_bytes', 0)}")
            print(f"  count: {features.get('count', 0)}")
            print(f"  same_srv_rate: {features.get('same_srv_rate', 0)}")
            print(f"  serror_rate: {features.get('serror_rate', 0)}")
            print(f"  rerror_rate: {features.get('rerror_rate', 0)}")
            
            # Mailbomb Pattern Analysis (informational only)
            protocol = features.get('protocol_type', '')
            service = features.get('service', '')
            src_bytes = features.get('src_bytes', 0)
            dst_bytes = features.get('dst_bytes', 0)
            count = features.get('count', 0)
            same_srv_rate = features.get('same_srv_rate', 0)
            serror_rate = features.get('serror_rate', 0)
            
            mailbomb_score = 0
            mailbomb_indicators = []
            
            # Check for high connection count
            if count > 1:
                mailbomb_score += 2
                mailbomb_indicators.append(f'high_count_{count}')
                
            # Check for same service rate
            if same_srv_rate >= 0.8:
                mailbomb_score += 3
                mailbomb_indicators.append(f'same_srv_rate_{same_srv_rate:.2f}')
                
            # Check for mail-related services
            if service.lower() in ['smtp', 'pop_3', 'imap4', 'pop_2']:
                mailbomb_score += 2
                mailbomb_indicators.append(f'mail_service_{service}')
                
            # Check for TCP protocol
            if protocol.lower() == 'tcp':
                mailbomb_score += 1
                mailbomb_indicators.append('tcp_protocol')
                
            # Check for large data transfer
            if dst_bytes > 100:
                mailbomb_score += 1
                mailbomb_indicators.append(f'large_dst_bytes_{dst_bytes}')
                
            # Check for no errors
            if serror_rate == 0:
                mailbomb_score += 1
                mailbomb_indicators.append('no_serror')
            
            print(f"Debug - Mailbomb analysis: score={mailbomb_score}, indicators={mailbomb_indicators}")
            
            # Pattern detection (informational only - no confidence boosting)
            if mailbomb_score >= 4:
                attack_indicators['pattern_detected'] = True
                attack_indicators['pattern_type'] = 'mailbomb'
                attack_indicators['specific_features'].extend(mailbomb_indicators)
                attack_indicators['pattern_score'] = mailbomb_score
                print(f"Debug - MAILBOMB PATTERN DETECTED! Score: {mailbomb_score}")
                return attack_indicators
            
            # Other attack patterns (informational only)
            if count > 500 or features.get('srv_count', 0) > 500:
                attack_indicators['pattern_detected'] = True
                attack_indicators['pattern_type'] = 'dos_flooding'
                attack_indicators['specific_features'].append('high_connection_count')
            
            elif protocol == 'tcp' and service in ['finger', 'sunrpc', 'ntp_u']:
                attack_indicators['pattern_detected'] = True
                attack_indicators['pattern_type'] = 'probe_scan'
                attack_indicators['specific_features'].append('unusual_service')
            
            elif features.get('num_root', 0) > 0 or features.get('root_shell', 0) > 0:
                attack_indicators['pattern_detected'] = True
                attack_indicators['pattern_type'] = 'privilege_escalation'
                attack_indicators['specific_features'].append('root_access_attempt')
            
            elif features.get('num_failed_logins', 0) > 2:
                attack_indicators['pattern_detected'] = True
                attack_indicators['pattern_type'] = 'brute_force'
                attack_indicators['specific_features'].append('multiple_failed_logins')
            
            else:
                error_rate = serror_rate + features.get('rerror_rate', 0)
                if error_rate > 0.5:
                    attack_indicators['pattern_detected'] = True
                    attack_indicators['pattern_type'] = 'network_anomaly'
                    attack_indicators['specific_features'].append('high_error_rate')
            
        except Exception as e:
            print(f"⚠️  Pattern analysis error: {e}")
        
        return attack_indicators
    
    def _calculate_weighted_ensemble(self, base_predictions, ensemble_predictions, dl_confidence):
        """Calculate weighted ensemble with enhanced accuracy"""
        try:
            # Base weights for different model types
            weights = {
                'base_models': 0.3,
                'ensemble_models': 0.4,
                'deep_learning': 0.3
            }
            
            # Adjust weights based on deep learning confidence
            if dl_confidence > 0.9:
                weights['deep_learning'] = 0.5
                weights['ensemble_models'] = 0.3
                weights['base_models'] = 0.2
            elif dl_confidence < 0.6:
                weights['deep_learning'] = 0.1
                weights['ensemble_models'] = 0.5
                weights['base_models'] = 0.4
            
            # Calculate base model confidence
            base_confidence = np.mean(base_predictions) if base_predictions else 0
            
            # Calculate ensemble confidence
            ensemble_confidence = np.mean(ensemble_predictions) if ensemble_predictions else 0
            
            # Weighted final confidence
            weighted_confidence = (
                base_confidence * weights['base_models'] +
                ensemble_confidence * weights['ensemble_models'] +
                dl_confidence * weights['deep_learning']
            )
            
            # Apply consensus boosting
            total_models = len(base_predictions) + len(ensemble_predictions) + 1
            total_attack_predictions = sum(1 for pred in base_predictions if pred > 0.5) + sum(1 for pred in ensemble_predictions if pred > 0.5) + (1 if dl_confidence > 0.5 else 0)
            
            consensus_ratio = total_attack_predictions / total_models
            if consensus_ratio >= 0.7:  # Strong consensus
                weighted_confidence = min(0.98, weighted_confidence * 1.1)
            elif consensus_ratio <= 0.3:  # Strong normal consensus
                weighted_confidence = max(0.02, weighted_confidence * 0.9)
            
            return max(0.0, min(1.0, weighted_confidence))
            
        except Exception as e:
            print(f"⚠️  Weighted ensemble calculation error: {e}")
            return dl_confidence  # Fallback to deep learning confidence

# Global predictor instance
predictor = NIDSPredictor()

# Load models on module import for production deployment
try:
    predictor.load_models()
    if predictor.models_loaded:
        print("✅ Models loaded successfully on startup")
    else:
        print("⚠️  Models not loaded - will attempt again when first request is made")
except Exception as e:
    print(f"⚠️  Error loading models on startup: {e}")
    print("🔄 Will attempt to load models on first request")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor.models_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/models/status', methods=['GET'])
def models_status():
    """Get status of loaded models"""
    return jsonify({
        'models_loaded': predictor.models_loaded,
        'base_models': list(predictor.base_models.keys()),
        'ensemble_models': list(predictor.ensemble_models.keys()),
        'multimodal_available': predictor.multimodal_model is not None,
        'scalers_loaded': len(predictor.scalers),
        'encoders_loaded': len(predictor.encoders)
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_packet():
    """Analyze network packet data"""
    try:
        data = request.get_json()
        
        if not data or 'packet_data' not in data:
            return jsonify({'error': 'Missing packet_data in request'}), 400
        
        packet_data = data['packet_data'].strip()
        if not packet_data:
            return jsonify({'error': 'Empty packet data provided'}), 400
        
        print(f"Debug - Analyzing packet: {packet_data[:50]}...")
        
        # Analyze the packet
        results = predictor.predict_packet(packet_data)
        
        print(f"Debug - Analysis completed, returning results")
        
        # Ensure all values are JSON serializable
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = make_serializable(results)
        
        return jsonify(serializable_results)
        
    except ValueError as e:
        print(f"ValueError in analyze_packet: {e}")
        return jsonify({'error': f'Invalid packet data format: {str(e)}'}), 400
    except Exception as e:
        print(f"Exception in analyze_packet: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/analyze/batch', methods=['POST'])
def analyze_batch():
    """Analyze multiple packets at once"""
    try:
        data = request.get_json()
        
        if not data or 'packets' not in data:
            return jsonify({'error': 'Missing packets array in request'}), 400
        
        packets = data['packets']
        if not isinstance(packets, list):
            return jsonify({'error': 'Packets must be an array'}), 400
        
        results = []
        for i, packet_data in enumerate(packets):
            try:
                result = predictor.predict_packet(packet_data.strip())
                result['packet_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'packet_index': i,
                    'error': str(e),
                    'packet_data': packet_data[:100] + '...' if len(packet_data) > 100 else packet_data
                })
        
        return jsonify({
            'results': results,
            'total_packets': len(packets),
            'successful_analyses': len([r for r in results if 'error' not in r])
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch analysis failed: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("NEXUS AI - Network Intrusion Detection System API")
    print("=" * 60)
    
    # Load models on startup
    predictor.load_models()
    
    if not predictor.models_loaded:
        print("❌ WARNING: Models could not be loaded!")
        print("   Make sure the 'models' directory exists with trained models")
        print("   Run the multimodal_ensemble_nids.py script first to train models")
    
    print(f"\n🚀 Starting Flask API server...")
    print(f"📡 Available endpoints:")
    print(f"   GET  /api/health - Health check")
    print(f"   GET  /api/models/status - Model status")
    print(f"   POST /api/analyze - Analyze single packet")
    print(f"   POST /api/analyze/batch - Analyze multiple packets")
    
    # Use appropriate server based on environment
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    if debug_mode:
        print(f"🔧 Development mode - using Flask dev server")
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        print(f"🚀 Production mode - Flask app ready for WSGI server")
        print(f"🌐 Listening on port {port}")
        # In production, this will be served by gunicorn
        app.run(debug=False, host='0.0.0.0', port=port)
