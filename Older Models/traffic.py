import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, ReLU, Add, GlobalAveragePooling2D


base_path = "c:/Users/Sean/Desktop/TrafficLights/train_dataset/"
json_path = os.path.join(base_path, "train.json")
images_folder = os.path.join(base_path, "train_images/")

# Load dataset function to handle multiple annotations per image
def load_data(json_path, images_folder):
    with open(json_path, 'r') as f:
        data = json.load(f)

    images, colors, bndboxes = [], [], []

    for annotation in data["annotations"]:
        filename = annotation["filename"].split("\\")[-1]  # Handle Windows-style paths
        file_path = os.path.join(images_folder, filename)

        # Load and preprocess image
        image = tf.keras.preprocessing.image.load_img(file_path, target_size=(128, 128))
        image = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Normalize

        # For each traffic light (inbox) in the image, treat it as a separate sample
        for inbox in annotation["inbox"]:
            color = inbox["color"]
            if color == "red":
                colors.append(0)
            elif color == "green":
                colors.append(1)
            elif color == "yellow":
                colors.append(2)

            bndbox = annotation["bndbox"]
            box = [bndbox["xmin"], bndbox["ymin"], bndbox["xmax"], bndbox["ymax"]]
            bndboxes.append(box)

            images.append(image)

    return np.array(images), np.array(colors), np.array(bndboxes)

# Load the data
images, color_labels, bndbox_labels = load_data(json_path, images_folder)

# Convert color labels to categorical (3 classes: red, green, yellow)
color_labels = tf.keras.utils.to_categorical(color_labels, num_classes=3)

# Residual block function with 1x1 convolution for skip connection
def residual_block(x, filters):
    skip = x  # Save the input for the skip connection

    # First convolution layer
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second convolution layer
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    # Adjust the shape of the input tensor to match the output shape if needed
    if skip.shape[-1] != filters:
        skip = Conv2D(filters, (1, 1), padding='same')(skip)

    # Add skip connection
    x = Add()([x, skip])
    x = ReLU()(x)

    return x

# Define the model
input_layer = Input(shape=(128, 128, 3))

x = Conv2D(32, (3, 3), padding='same')(input_layer)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D((2, 2))(x)

x = residual_block(x, 64)
x = MaxPooling2D((2, 2))(x)

x = residual_block(x, 128)
x = MaxPooling2D((2, 2))(x)

x = residual_block(x, 256)
x = MaxPooling2D((2, 2))(x)

x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layers for classification and bounding box regression
color_output = Dense(3, activation='softmax', name='color_output')(x)
bbox_output = Dense(4, activation='linear', name='bbox_output')(x)

# Define the complete model
model = Model(inputs=input_layer, outputs=[color_output, bbox_output])

# Compile the model with smooth L1 loss for bounding box regression
def smooth_l1_loss(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = less_than_one * 0.5 * diff**2 + (1 - less_than_one) * (diff - 0.5)
    return tf.reduce_mean(loss)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={'color_output': 'categorical_crossentropy', 'bbox_output': smooth_l1_loss},
    metrics={'color_output': 'accuracy', 'bbox_output': 'mse'}
)

# Train the model
history = model.fit(
    images, 
    {'color_output': color_labels, 'bbox_output': bndbox_labels}, 
    epochs=20, 
    batch_size=32, 
    verbose=2
)

# Extract and print metrics from the training history
color_accuracy = history.history['color_output_accuracy']
bbox_mse = history.history['bbox_output_mse']

print("\nTraining Results:")
for epoch in range(len(color_accuracy)):
    print(f"Epoch {epoch + 1}: Color Accuracy = {color_accuracy[epoch]:.4f}, "
          f"BBox MSE = {bbox_mse[epoch]:.4f}")

# Save the trained model to Google Drive
model.save("/content/drive/MyDrive/TrafficLights/traffic_light_model_with_residuals.h5")
print("Model training complete and saved.")
