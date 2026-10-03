import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Dataset
dataset_path = "dataset"

# Images prepare karna
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.5
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=2,
    class_mode="binary",
    subset="training"
)

validation_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=2,
    class_mode="binary",
    subset="validation"
)

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Training
model.fit(
    train_data,
    validation_data=validation_data,
    epochs=5
)

# Save model
model.save("cat_dog_model.keras")

print("Model training complete!")