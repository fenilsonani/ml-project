import os
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import numpy as np

# Classification and regression targets defined previously
classification_targets = ['shape', 'clarity',
                          'colour', 'cut', 'polish', 'symmetry']
regression_target = 'carat'

# Directory where models are stored
model_dir = 'sklearn_models'
base_dir = ''  # Adjust if necessary


def load_models_and_encoders():
    """
    Load all saved classifiers, label encoders, and the regressor.
    """
    classifiers = {}
    label_encoders = {}

    for target in classification_targets:
        clf_path = os.path.join(model_dir, f'classifier_{target}.joblib')
        le_path = os.path.join(model_dir, f'label_encoder_{target}.joblib')
        if os.path.exists(clf_path) and os.path.exists(le_path):
            classifiers[target] = joblib.load(clf_path)
            label_encoders[target] = joblib.load(le_path)
        else:
            raise FileNotFoundError(f"Missing model or encoder for {target}")

    regressor_path = os.path.join(model_dir, 'regressor_carat.joblib')
    if os.path.exists(regressor_path):
        regressor = joblib.load(regressor_path)
    else:
        raise FileNotFoundError("Missing regressor model for carat")

    return classifiers, label_encoders, regressor


def initialize_feature_extractor():
    """
    Initialize the ResNet18 model for feature extraction.
    This should match exactly how you extracted features during training.
    """
    device = torch.device('cpu')
    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet.eval().to(device)

    # Transform should match the one used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return resnet, transform, device


def extract_features_for_image(image_path, resnet, transform, device):
    """
    Extract feature vector from a single image using the given resnet and transform.
    """
    img = Image.open(os.path.join(base_dir, image_path)).convert('RGB')
    img = transform(img).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        features = resnet(img.to(device))
    features = features.squeeze().numpy()  # shape [512]
    return features


def predict_image(image_path):
    """
    Given an image path relative to base_dir, perform the full inference:
    1. Extract features.
    2. Predict classification targets.
    3. Predict regression target.
    4. Decode classification results and print all predictions.
    """
    # Load models and encoders
    classifiers, label_encoders, regressor = load_models_and_encoders()

    # Initialize feature extractor
    resnet, transform, device = initialize_feature_extractor()

    # Extract features
    features = extract_features_for_image(
        image_path, resnet, transform, device)
    features = features.reshape(1, -1)  # [1, feature_dim]

    # Predict classification targets
    class_predictions = {}
    for target in classification_targets:
        pred_label = classifiers[target].predict(features)[0]
        # Decode to original string label
        decoded_label = label_encoders[target].inverse_transform([pred_label])[
            0]
        class_predictions[target] = decoded_label

    # Predict regression target (carat)
    carat_pred = regressor.predict(features)[0]

    # Print results
    print(f"Predictions for image: {image_path}")
    for target in classification_targets:
        print(f"{target.capitalize()}: {class_predictions[target]}")
    print(f"Carat (predicted): {carat_pred:.4f}")


# Example usage:
# Suppose you have an image "example_image.jpg" inside data/ directory
# Just call the function:
predict_image("example.jpeg")  # Replace with an actual image filename
