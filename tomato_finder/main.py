import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load CSV file
df = pd.read_csv('data\\train.csv')

# Path to the directory where images are stored
image_directory = 'data\\Three Classes'

# Preprocess images: normalize pixel values, resize images, etc.
datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of tensor image data with real-time data augmentation
train_generator = datagen.flow_from_dataframe(
    dataframe=df, 
    x_col="name",
    y_col="condition",
    class_mode="categorical",  # for multi-class classification
    target_size=(64, 64),  # resize images
    batch_size=32
)

# Define your CNN model
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # Assuming 10 classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=100)  # Number of epochs can be adjusted

model.save('tomato_model.h5')