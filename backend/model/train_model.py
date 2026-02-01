import tensorflow as tf
from model_architecture import ImprovedChangeDetectionModel
from utils.data_loader import TiffDataLoader
import os
import matplotlib.pyplot as plt

def train_model():
    print("Initializing TIFF Change Detection Model...")
    
    # Initialize model
    model_wrapper = ImprovedChangeDetectionModel(input_shape=(256, 256, 3))
    model = model_wrapper.model
    model_wrapper.compile_model()
    
    print("Model architecture:")
    model.summary()
    
    # Load TIFF data
    print("Loading TIFF datasets...")
    data_loader = TiffDataLoader(data_path='../datasets/', batch_size=8)
    
    # Split data (you might want a more sophisticated split)
    all_datasets = data_loader.discover_datasets()
    print(f"Found {len(all_datasets)} image pairs")
    
    split_idx = int(0.8 * len(all_datasets))
    train_datasets = all_datasets[:split_idx]
    val_datasets = all_datasets[split_idx:]
    
    def train_generator():
        return data_loader.create_data_generator_from_list(train_datasets)
    
    def val_generator():
        return data_loader.create_data_generator_from_list(val_datasets)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'weights/best_model.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator(),
        steps_per_epoch=len(train_datasets) // data_loader.batch_size,
        epochs=100,
        validation_data=val_generator(),
        validation_steps=len(val_datasets) // data_loader.batch_size,
        callbacks=callbacks
    )
    
    # Save final model
    model.save('weights/final_model.h5')
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    model, history = train_model()