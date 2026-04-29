import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os

def build_model(num_classes):
    """
    Builds a high-accuracy model using Transfer Learning with MobileNetV2.
    """
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def save_initial_model():
    """
    Saves an initial model so the web application can start using the TensorFlow engine immediately.
    Note: This model needs to be trained on real data to be accurate!
    """
    print("Creating initial model for crop detection...")
    num_classes = 5
    model = build_model(num_classes)
    
    model_path = 'crop_model.h5'
    model.save(model_path)
    print(f"Initial model saved to {model_path}")
    print("Next steps: Train this model on the PlantVillage dataset for high accuracy.")

if __name__ == "__main__":
    save_initial_model()
