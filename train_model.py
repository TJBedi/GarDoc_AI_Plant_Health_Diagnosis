import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Force GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# GPU configuration
print("TensorFlow version:", tf.__version__)
print("Checking GPU availability...")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
        tf.config.experimental.set_memory_growth(device, True)
        print(f"GPU name: {device.name}")
        print(f"GPU device details: {tf.config.experimental.get_device_details(device)}")
else:
    print("No GPU found. Running on CPU")
    print("Available devices:", tf.config.list_physical_devices())

def diagnose_gpu():
    print("\n=== GPU Diagnostic Information ===")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # Check if CUDA is available through TensorFlow
    print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"TensorFlow GPU available: {tf.test.is_gpu_available()}")
    
    # Try to execute a simple operation on GPU
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print(f"Matrix multiplication result: {c}")
            print("Successfully executed operation on GPU")
    except Exception as e:
        print(f"Error executing operation on GPU: {e}")
    
    print("=== End of Diagnostic Information ===\n")

# Call the diagnostic function
diagnose_gpu()

# Rest of the imports
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = 224  # ResNet50 input size
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001
DATASET_PATH = 'dataset'
MODEL_PATH = 'plant_classification_model.h5'

# Create data generators with augmentation
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

# Build the model
def build_model(num_classes):
    # Load the ResNet50 model with pre-trained ImageNet weights
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

# Fine-tune the model
def fine_tune_model(model, base_model, train_generator, validation_generator):
    # First train only the top layers
    print("Training top layers...")
    history1 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=10,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        ]
    )
    
    # Unfreeze some layers of the base model for fine-tuning
    print("Fine-tuning ResNet layers...")
    for layer in base_model.layers[-30:]:  # Unfreeze the last 30 layers
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    history2 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
            ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        ]
    )
    
    return model, history1, history2

# Plot training history
def plot_training_history(history1, history2):
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig('training_history.png')
    plt.show()

# Evaluate the model
def evaluate_model(model, validation_generator):
    # Get the class indices
    class_indices = validation_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    
    # Get predictions
    validation_generator.reset()
    y_pred = model.predict(validation_generator, steps=validation_generator.samples // BATCH_SIZE + 1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true labels
    y_true = validation_generator.classes[:len(y_pred_classes)]
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred_classes, target_names=[class_names[i] for i in range(len(class_names))])
    print(report)
    
    # Save the report to a file
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    
    # Create and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(16, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, [class_names[i] for i in range(len(class_names))], rotation=90)
    plt.yticks(tick_marks, [class_names[i] for i in range(len(class_names))])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    
    return class_names

def main():
    # Create data generators
    train_generator, validation_generator = create_data_generators()
    
    # Get number of classes
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {train_generator.class_indices}")
    
    # Build the model
    model, base_model = build_model(num_classes)
    
    # Print model summary
    model.summary()
    
    # Train and fine-tune the model
    model, history1, history2 = fine_tune_model(model, base_model, train_generator, validation_generator)
    
    # Plot training history
    plot_training_history(history1, history2)
    
    # Evaluate the model
    class_names = evaluate_model(model, validation_generator)
    
    # Save the class names mapping
    class_mapping = {i: name for i, name in enumerate(class_names.values())}
    with open('class_mapping.txt', 'w') as f:
        for idx, name in class_mapping.items():
            f.write(f"{idx}: {name}\n")
    
    print(f"Model saved to {MODEL_PATH}")
    print("Training complete!")

if __name__ == "__main__":
    main()