import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

menu = ['Capture From Webcam']

choice = st.sidebar.selectbox('Side Menu?', menu)


#Load your model and check create the class_names list
Model_Path = '____________'
class_names = [_________________________]
model = tf.keras.load_model(Model_Path)


if choice == 'Capture From Webcam':
    cap = cv2.VideoCapture(0)  # device 0
    run = st.checkbox('Show Webcam')
    capture_button = st.checkbox('Capture')

    captured_image = np.array(None)


    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    FRAME_WINDOW = st.image([])
    while run:
        ret, frame = cap.read()        
        # Display Webcam
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB ) #Convert color
        FRAME_WINDOW.image(frame)

        if capture_button:      
            captured_image = frame
            break

    cap.release()

    if  captured_image.all() != None:
        st.image(captured_image)
        st.write('Image is capture:')

        #Resize the Image according with your model
        captured_image = cv2.resize(_____________)
        #Expand dim to make sure your img_array is (1, Height, Width , Channel ) before plugging into the model
        img_array  = np.expand_dims(captured_image, axis=0)
        #Check the img_array here
        st.write(img_array)

        prediction = model.predict(img_array)

        # Preprocess your prediction , How are we going to get the label name out from the prediction
        # Now it's your turn to solve the rest of the code