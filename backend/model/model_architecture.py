import tensorflow as tf
from tensorflow.keras import layers, Model

class ImprovedChangeDetectionModel:
    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        self.model = self.build_improved_model()
    
    def build_improved_model(self):
        """Improved U-Net architecture for satellite change detection"""
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Encoder with more capacity
        # Block 1
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        # Block 2
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        # Block 3
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Bottleneck
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
        
        # Decoder with skip connections
        # Up Block 1
        up1 = layers.UpSampling2D(size=(2, 2))(conv4)
        concat1 = layers.concatenate([conv3, up1], axis=-1)
        conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(concat1)
        conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
        
        # Up Block 2
        up2 = layers.UpSampling2D(size=(2, 2))(conv5)
        concat2 = layers.concatenate([conv2, up2], axis=-1)
        conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat2)
        conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)
        
        # Up Block 3
        up3 = layers.UpSampling2D(size=(2, 2))(conv6)
        concat3 = layers.concatenate([conv1, up3], axis=-1)
        conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat3)
        conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
        
        # Output layer
        outputs = layers.Conv2D(1, 1, activation='sigmoid', name='change_mask')(conv7)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.MeanIoU(num_classes=2, name='iou')
            ]
        )