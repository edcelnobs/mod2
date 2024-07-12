import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

st.header('Potato Image Disease Classification Model')

# Load the pre-trained Keras model
model = load_model('modelp.keras')

# Define the categories
name = text.Label['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
img_height, img_width = 180, 180

# Upload an image using Streamlit file uploader
image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image is not None:
    # Load and preprocess the image for prediction
    image_load = tf.keras.preprocessing.image.load_img(image, target_size=(img_height, img_width))
    img_arr = tf.keras.preprocessing.image.img_to_array(image_load)
    img_bat = np.expand_dims(img_arr, axis=0)

    # Make predictions
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Display the prediction result
    st.write('Potato Disease Predicted: ' + mane[np.argmax(score)])
    st.write(f'With accuracy: {np.max(score) * 100:.2f}%')
