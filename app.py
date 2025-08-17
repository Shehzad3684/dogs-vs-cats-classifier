import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# 1. Load the trained model
try:
    loaded_model = load_model('/tmp/dogs_vs_cats_model.h5') # Assuming the model is saved in /tmp
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")
    loaded_model = None # Set to None if loading fails

# 3. Implement image preprocessing
def preprocess_image(image_array):
    """
    Preprocesses an image to match the input requirements of the model.

    Args:
        image_array: A NumPy array representing the image.

    Returns:
        A preprocessed NumPy array ready for model prediction.
    """
    # Resize the image
    img = Image.fromarray(image_array.astype(np.uint8)) # Ensure correct data type for Image.fromarray
    img = img.resize((256, 256))
    img_array = np.array(img)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Expand dimensions to add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# 4. Integrate the model for prediction
def classify_image(image):
    """
    Preprocesses the image and predicts whether it is a cat or a dog.

    Args:
        image: A NumPy array representing the input image.

    Returns:
        A string indicating the predicted class ("Dog" or "Cat") or an error message.
    """
    if loaded_model is None:
        return "Error: Model not loaded."

    try:
        preprocessed_img = preprocess_image(image)
        prediction = loaded_model.predict(preprocessed_img)

        # Assuming a threshold of 0.5, where >= 0.5 is Dog and < 0.5 is Cat
        if prediction[0][0] >= 0.5:
            return "Dog"
        else:
            return "Cat"
    except Exception as e:
        return f"Error during prediction: {e}"


# 2. Build the user interface and 5. Display the results
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Dog vs Cat Classifier",
    description="Upload an image and the model will predict if it is a dog or a cat."
)

if __name__ == "__main__":
    iface.launch(share=True)