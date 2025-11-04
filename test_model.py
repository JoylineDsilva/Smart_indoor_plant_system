import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('plant_health_model.h5')

# Define class labels
class_labels = ["Healthy", "Pest Infested", "Disease Affected", "Nutrient Deficient"]

# Function to predict the class of a new image
def predict_plant_health(image_path):
    img = image.load_img(image_path, target_size=(150, 150))  # Resize image to match model's expected input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize image

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions, axis=1)[0]]
    confidence = float(predictions[0][np.argmax(predictions)]) * 100  # Convert to percentage

    return predicted_class, confidence

# Test the model with an example image
image_path = 'C:/Users/savin/SMARTPLANTCAREAPPS/data/processed_test'  # Replace with the path to a test image
predicted_class, confidence = predict_plant_health(image_path)

# Print the result
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence}%")


