import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = load_model('model_name.h5')

# Directory containing test images
test_image_directory = 'testing_toy'  # Replace with your test image directory

# Preprocess images: normalize pixel values, resize images, etc.
test_datagen = ImageDataGenerator(rescale=1./255)

# Prepare a list of filenames and their predicted classes
filenames = []
predictions = []

# Iterate over each image in the test directory
for filename in os.listdir(test_image_directory):
    if filename.endswith('.png'):  # or '.jpg', '.jpeg' depending on your file type
        # Load and preprocess the image
        img_path = os.path.join(test_image_directory, filename)
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make a prediction
        predicted_class = np.argmax(model.predict(img_array), axis=-1)[0]

        # Append results
        filenames.append(filename)
        predictions.append(predicted_class)

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'Filename': filenames,
    'PredictedClass': predictions
})

# Save the results to a CSV file
results_df.to_csv('output.csv', index=False)
