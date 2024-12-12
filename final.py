import os
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
import albumentations as A
from albumentations.core.composition import OneOf

# Configuration
base_dir = 'data/'
csv_path = os.path.join(base_dir, 'diamond_data.csv')
output_dir = 'organized_data/'
target_column = 'shape'  # Change to 'colour' if needed
img_height, img_width = 200, 200
batch_size = 32
epochs = 2
learning_rate = 0.001

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)

# Step 1: Read CSV and Verify Images
print("Loading data...")
df = pd.read_csv(csv_path)
df['path_to_img'] = df['path_to_img'].str.replace('web_scraped/', '', regex=False)

print("Checking image existence...")
df['image_exists'] = df['path_to_img'].apply(lambda x: os.path.exists(os.path.join(base_dir, x)))
df = df[df['image_exists']].drop(columns=['image_exists']).reset_index(drop=True)
print(f"Total valid images: {len(df)}")

# Optional: Use a fraction of data for faster development
frac_dataset = 0.3  # 30%
df = df.sample(frac=frac_dataset, random_state=42).reset_index(drop=True)
print(f"Data reduced to {len(df)} samples for development.")

# Step 2: Encode Labels
print("Encoding labels...")
le = LabelEncoder()
df[target_column] = le.fit_transform(df[target_column].fillna('Unknown'))
num_classes = len(le.classes_)
print(f"Number of classes: {num_classes}")

# Step 3: Split Data
print("Splitting data into train/validation/test sets...")
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,  # 70% train, 30% temp
    stratify=df[target_column],
    random_state=42
)

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,  # 15% validation, 15% test
    stratify=temp_df[target_column],
    random_state=42
)

print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(valid_df)}")
print(f"Test size: {len(test_df)}")

# Step 4: Organize Data into Directories
def organize_data(df, split_name, output_dir, base_dir, target_column):
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Organizing {split_name} data"):
        img_path = os.path.join(base_dir, row['path_to_img'])
        label = le.inverse_transform([row[target_column]])[0]
        dest_dir = os.path.join(output_dir, split_name, label)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(row['path_to_img']))
        try:
            # Use symlinks if possible
            if not os.path.exists(dest_path):
                os.symlink(os.path.abspath(img_path), dest_path)
        except OSError:
            # If symlink fails, copy the file
            shutil.copy(img_path, dest_path)

print("Organizing data into directories...")
organize_data(train_df, 'train', output_dir, base_dir, target_column)
organize_data(valid_df, 'validation', output_dir, base_dir, target_column)
organize_data(test_df, 'test', output_dir, base_dir, target_column)
print("Data organization complete.")

# Step 5: Data Augmentation (Optional)
# If classes are imbalanced, consider augmenting the training data.
# This step can be skipped if data is already balanced.

# Step 6: Create ImageDataGenerators
print("Creating ImageDataGenerators...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(output_dir, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    os.path.join(output_dir, 'validation'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    os.path.join(output_dir, 'test'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Step 7: Build the Model using Transfer Learning
print("Building the model...")
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
base_model.trainable = True  # Enable fine-tuning

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Step 8: Define Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

# Step 9: Train the Model
print("Starting training...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# Step 10: Evaluate the Model
print("Evaluating the model on the test set...")
test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Classification Report
print("Classification Report:")
report = classification_report(y_true, y_pred, target_names=le.classes_)
print(report)

# Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Step 11: Save the Model
model_save_path = os.path.join(output_dir, f'model_{le.classes_[0]}_to_{le.classes_[-1]}.h5')
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
