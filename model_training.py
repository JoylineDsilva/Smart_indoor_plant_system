import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths for saving models (changed to .keras extension)
health_model_path = "C:/Users/savin/SMARTPLANTCAREAPPS/plant_health_model.keras"
disease_model_path = "C:/Users/savin/SMARTPLANTCAREAPPS/plant_disease_model.keras"

# Paths for training and testing datasets
train_dir = "C:/Users/savin/SMARTPLANTCAREAPPS/data/processed_train"
test_dir = "C:/Users/savin/SMARTPLANTCAREAPPS/data/processed_test"

# Define ImageDataGenerators for training and validation data
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # Assuming 3 classes: Healthy, Diseased, Pest
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Define a simple CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # Output layer with 3 classes: Healthy, Diseased, Pest
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the plant health model
plant_health_model = create_model()
health_checkpoint = ModelCheckpoint(health_model_path, save_best_only=True, verbose=1)

# Train the plant health model
print("Training Plant Health Model...")
plant_health_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[health_checkpoint]
)

# Create and train the plant disease model
plant_disease_model = create_model()
disease_checkpoint = ModelCheckpoint(disease_model_path, save_best_only=True, verbose=1)

# Train the plant disease model
print("Training Plant Disease Model...")
plant_disease_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[disease_checkpoint]
)

print("Training completed for both models!")
