import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from train import train_model
import base64
import matplotlib.pyplot as plt
from tensorflow import keras

image_path = 'img/sign.jpeg'
#Convert the image to base64
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

st.set_page_config(page_title='Handwritten Digit Classifier', page_icon="🚀")

st.title('Handwritten Digit Recognizer')
st.subheader('This webapp predicts the number that you have drawn on the canvas from 0-9')

# Define the model directory and model file path
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model.keras')
MODEL_FILE = 'model.keras'


col3, col4 = st.columns(2)

# Check if the model file exists
if not os.path.isfile(MODEL_DIR):
    st.warning(f"Model file not found at {MODEL_FILE}. Please train the data first.")
    if st.button("Train Data", type="primary"):
        with st.spinner("Training the model..."):
            train_model(MODEL_FILE)
else:
    try:
        # Load the model
        model = load_model(MODEL_FILE)
        st.success(f"Model loaded successfully from {MODEL_FILE}.")        
        # Divide into two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Draw your digit here:")
            canvas_result = st_canvas(
                fill_color='#000000',
                stroke_width=30,
                stroke_color='#FFFFFF',
                background_color='#000000',
                width=300,
                height=300,
                drawing_mode="freedraw" if st.checkbox("Draw (or Delete)?", True) else "transform",
                key='canvas'
            )
        
        with col2:
            if canvas_result.image_data is not None:
                img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
                rescaled = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
                st.write("Your drawn image: ")
                st.write('Model Input')
                st.write(' ')
                st.image(rescaled)

                # Convert the image to grayscale and normalize
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_img = gray_img / 255.0

                # Add a batch dimension
                img_array = gray_img.reshape((1, 28, 28, 1))

                # Make a prediction
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction)

                # Display the prediction
                st.write("Predicted Class:", predicted_class)
                image_html = f'''
                <div style="text-align: right;  padding-right: 15px;">
                    <img src="data:image/jpeg;base64,{encoded_image}" alt="Image" width="150">
                </div>
                '''
                st.markdown(image_html, unsafe_allow_html=True)
            else:
                st.write("Please draw a digit on the canvas.")
        with st.expander("Select a Sample Image"):
            (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
            X_train = X_train / 255.0
            X_test = X_test / 255.0

            sample_image_index = st.selectbox("Select a sample image:", range(len(X_test)))
            st.write("Sample Image:")
            plt.imshow(X_test[sample_image_index], cmap='gray')
            st.pyplot(plt)

            input_image = X_test[sample_image_index].reshape(1, 28, 28, 1)
            prediction = model.predict(input_image)
            predicted_class = np.argmax(prediction)

            st.write("Predicted Class:", predicted_class)

    except Exception as e:
        st.error(f"Error loading the model: {e}")

