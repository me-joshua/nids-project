
# Loading Instructions for Multimodal Ensemble NIDS Models

## Load TensorFlow Model:
```python
import tensorflow as tf
model = tf.keras.models.load_model('models\multimodal_deep_learning')
```

## Load Sklearn Models:
```python
import joblib
base_models = {}
for model_name in ['random_forest', 'extra_trees', 'gradient_boosting', 'svm', 'neural_network', 'logistic_regression']:
    base_models[model_name] = joblib.load(f'models\base_models/{model_name}.pkl')

ensemble_models = {}
for model_name in ['hard_voting', 'soft_voting']:
    ensemble_models[model_name] = joblib.load(f'models\ensemble_models/{model_name}.pkl')
```

## Load Preprocessing:
```python
scalers = joblib.load('models\preprocessing\scalers.pkl')
encoders = joblib.load('models\preprocessing\encoders.pkl')
```

Total Models Saved: 9
Total Size: ~22.4 MB
