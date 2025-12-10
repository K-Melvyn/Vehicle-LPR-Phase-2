"""
train_model.py
Train a plate-type classifier (Phase 1)
Usage: python train_model.py
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# ----------------------------
# CONFIG - modify these paths
TRAIN_DIR = "dataset/training"
VAL_DIR = "dataset/validation"
MODEL_OUTPUT = "ghana_plate_classifier.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 8  # ajuster si tu as plus/moins de classes
# ----------------------------

if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
    raise FileNotFoundError(
        f"Dataset directories not found. Expecting:\n - {TRAIN_DIR}\n - {VAL_DIR}\nPlease create them and add images organized by class folders.")

# Data augmentation & generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Base model (transfer learning)
base_model = MobileNetV2(weights='imagenet', include_top=False,
                         input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(train_gen.num_classes,
                    activation='softmax')(x)  # dynamique

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Starting training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# Save model
model.save(MODEL_OUTPUT)
print(f"Model saved to {MODEL_OUTPUT}")

# Print baseline metrics
train_acc = history.history.get(
    'accuracy', [])[-1] if history.history.get('accuracy') else None
val_acc = history.history.get(
    'val_accuracy', [])[-1] if history.history.get('val_accuracy') else None
if train_acc is not None and val_acc is not None:
    print(f"Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%")
else:
    print("Training finished — no accuracy values found in history.")
