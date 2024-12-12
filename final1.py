# =============================================================================
# Section 1: Imports and Configuration
# =============================================================================

import os
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Machine Learning Imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Handling Class Imbalance
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE

# Explainability
import shap

# Serialization
import joblib
import json

# Suppress warnings for clean output
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Section 2: Configuration
# =============================================================================

# Paths and Directories
base_dir = 'data/'
csv_path = os.path.join(base_dir, 'diamond_data.csv')
output_dir = 'processed_data/'
feature_dir = os.path.join(output_dir, 'features/')
model_dir = os.path.join(output_dir, 'models/')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(feature_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Parameters
target_column = 'shape'  # Change to 'colour' if needed
img_height, img_width = 200, 200  # For feature extraction
batch_size = 32
frac_dataset = 0.1  # 30% of data for development

# =============================================================================
# Section 3: Data Loading and Cleaning
# =============================================================================

print("Loading data...")
df = pd.read_csv(csv_path)
df['path_to_img'] = df['path_to_img'].str.replace('web_scraped/', '', regex=False)

print("Checking image existence...")
df['image_exists'] = df['path_to_img'].apply(lambda x: os.path.exists(os.path.join(base_dir, x)))
df = df[df['image_exists']].drop(columns=['image_exists']).reset_index(drop=True)
print(f"Total valid images: {len(df)}")

# Handle Missing Values in Target Column
df[target_column] = df[target_column].fillna('Unknown')

# Optional: Use a fraction of data for faster development
df = df.sample(frac=frac_dataset, random_state=42).reset_index(drop=True)
print(f"Data reduced to {len(df)} samples for development.")

# =============================================================================
# Section 4: Exploratory Data Analysis (EDA)
# =============================================================================

print("Performing Exploratory Data Analysis...")

# Distribution of Target Variable
plt.figure(figsize=(8,6))
sns.countplot(data=df, x=target_column)
plt.title(f'Distribution of {target_column}')
plt.xticks(rotation=45)
plt.show()

# Target Variable Percentage
plt.figure(figsize=(8,6))
df[target_column].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
plt.ylabel('')
plt.title(f'{target_column} Distribution')
plt.show()

# If there are additional features (metadata), you can plot their distributions
# Example:
# if 'carat' in df.columns:
#     plt.figure(figsize=(8,6))
#     sns.histplot(df['carat'], bins=30, kde=True)
#     plt.title('Distribution of Carat')
#     plt.show()

# Correlation Matrix (if applicable)
# For image data, correlations between pixel values are not meaningful.
# If you have metadata features, you can compute correlations.
# Example:
# metadata_cols = ['carat', 'depth', 'table']
# if all(col in df.columns for col in metadata_cols):
#     plt.figure(figsize=(10,8))
#     sns.heatmap(df[metadata_cols].corr(), annot=True, cmap='coolwarm')
#     plt.title('Correlation Matrix of Metadata Features')
#     plt.show()

# =============================================================================
# Section 5: Handling Class Imbalance
# =============================================================================

print("Checking for class imbalance...")
class_counts = df[target_column].value_counts()
print(class_counts)

# Visualize Class Imbalance
plt.figure(figsize=(8,6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Class Distribution Before Balancing')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Compute Class Weights
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df[target_column])
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['label_encoded']),
    y=df['label_encoded']
)
class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}
print(f"Class Weights: {class_weights_dict}")

# =============================================================================
# Section 6: Feature Extraction from Images
# =============================================================================

print("Extracting features from images...")

def extract_hog_features(image_path, img_size=(128, 128)):
    """
    Extract Histogram of Oriented Gradients (HOG) features from an image.
    """
    image = cv2.imread(os.path.join(base_dir, image_path))
    if image is None:
        print(f"Warning: Unable to read image {image_path}.")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, img_size)
    hog = cv2.HOGDescriptor()
    h = hog.compute(image)
    return h.flatten()

# Apply feature extraction
features = []
valid_indices = []
for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting HOG features"):
    feat = extract_hog_features(row['path_to_img'], img_size=(128,128))
    if feat is not None:
        features.append(feat)
        valid_indices.append(idx)

# Update dataframe to only include successfully processed images
df = df.loc[valid_indices].reset_index(drop=True)
X = np.array(features)
y = df['label_encoded'].values
print(f"Extracted features shape: {X.shape}")

# Save extracted features for future use
feature_path = os.path.join(feature_dir, 'hog_features.npy')
np.save(feature_path, X)
print(f"HOG features saved to {feature_path}")

# =============================================================================
# Section 7: Feature Scaling
# =============================================================================

print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
scaler_path = os.path.join(feature_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# =============================================================================
# Section 8: Splitting Data
# =============================================================================

print("Splitting data into train/validation/test sets...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

print(f"Train size: {X_train.shape[0]}")
print(f"Validation size: {X_valid.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# =============================================================================
# Section 9: Machine Learning Models Development and Hyperparameter Tuning
# =============================================================================

print("Developing Machine Learning models with Hyperparameter Tuning...")

# Define ML Models and their Hyperparameter Grids
ml_models = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }
    },
    'SVC': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    }
}

# Store Best Models
best_models = {}

# Perform Grid Search with Cross-Validation
for model_name, mp in ml_models.items():
    print(f"Training {model_name}...")
    grid = GridSearchCV(
        estimator=mp['model'],
        param_grid=mp['params'],
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    best_models[model_name] = grid.best_estimator_
    print(f"Best parameters for {model_name}: {grid.best_params_}")
    print(f"Best CV F1-Score for {model_name}: {grid.best_score_:.4f}\n")

# =============================================================================
# Section 10: Evaluation of Machine Learning Models
# =============================================================================

print("Evaluating Machine Learning models on Test Set...")

for model_name, model in best_models.items():
    print(f"--- {model_name} ---")
    y_pred = model.predict(X_test)
    
    # For ROC AUC, handle multi-class
    if len(le.classes_) == 2:
        y_proba = model.predict_proba(X_test)[:,1]
    elif len(le.classes_) > 2:
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()
    
    # ROC AUC Score
    if y_proba is not None:
        if len(le.classes_) == 2:
            auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(figsize=(8,6))
            plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
            plt.plot([0,1], [0,1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc='lower right')
            plt.show()
            print(f"AUC-ROC: {auc:.4f}")
        else:
            # For multi-class, compute AUC for each class
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            print(f"Macro AUC-ROC: {auc:.4f}")
            # Plotting ROC curves for multi-class can be complex and is omitted here
    else:
        print("ROC AUC Score is not applicable.")
    
    # F1-Score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1-Score: {f1:.4f}\n")

# =============================================================================
# Section 11: Model Interpretation and Explainability
# =============================================================================

print("Generating SHAP values for Machine Learning models...")

# Initialize SHAP explainer
explainer = shap.Explainer(best_models['RandomForest'], X_train)

# Select a subset for SHAP to reduce computation
shap_values = explainer(X_test[:100])  # Adjust the number as needed

# Summary Plot
shap.summary_plot(shap_values, X_test[:100], feature_names=[f'feature_{i}' for i in range(X.shape[1])])

# Feature Importance for Random Forest
importances = best_models['RandomForest'].feature_importances_
indices = np.argsort(importances)[-10:]  # Top 10 features

plt.figure(figsize=(10,6))
plt.title("Top 10 Feature Importances - Random Forest")
sns.barplot(x=importances[indices], y=[f'Feature {i}' for i in indices])
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# =============================================================================
# Section 12: Saving Models and Results
# =============================================================================

print("Saving the models and results...")

# Save ML Models
for model_name, model in best_models.items():
    model_save_path = os.path.join(model_dir, f'ml_model_{model_name}.joblib')
    joblib.dump(model, model_save_path)
    print(f"{model_name} saved to {model_save_path}")

# Save Label Encoder
le_save_path = os.path.join(output_dir, 'label_encoder.pkl')
joblib.dump(le, le_save_path)
print(f"Label Encoder saved to {le_save_path}")

# Save Class Weights
class_weights_save_path = os.path.join(output_dir, 'class_weights.json')
with open(class_weights_save_path, 'w') as f:
    json.dump(class_weights_dict, f)
print(f"Class Weights saved to {class_weights_save_path}")

print("All models and artifacts have been saved successfully.")

# =============================================================================
# End of Script
# =============================================================================
