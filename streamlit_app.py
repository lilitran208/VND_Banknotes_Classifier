import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

menu = ['Home','Predict Banknotes']
choice = st.sidebar.selectbox('Menu',menu)

if choice == 'Home':
    st.header('Weekly Project 8 - Banknotes Classifier')
    st.write('Hello guys, this is my model to predict banknotes of Vietnam')
    st.image('.\media\donald duck.gif')
    
elif choice == 'Predict Banknotes':

# Load your model and check create the class_names list
    Model_Path = 'model_checkpoint4_DenseNet.h5'
    class_names = ['1000','10000','100000','2000','20000','200000','5000','50000','500000']
    model = tf.keras.models.load_model(Model_Path)
    
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

    if captured_image.all() != None:
        st.image(captured_image)
        st.write('Image is capture')

        #Resize the Image according with your model
        captured_image = cv2.resize(captured_image,(224,224))
        #Expand dim to make sure your img_array is (1, Height, Width , Channel ) before plugging into the model
        img_array  = np.expand_dims(captured_image, axis=0)
        #Check the img_array here
        #st.write(img_array)

        prediction = model.predict(img_array)

        # Preprocess your prediction , How are we going to get the label name out from the prediction
        # Now it's your turn to solve the rest of the code
        index = np.argmax(prediction.flatten())
        st.write('The banknote is ',class_names[index],'dong')
    
    uploaded_file = st.file_uploader('Upload photo',type=['png','jpg','jpeg'])
    

    if uploaded_file != None:
        # uploaded_image = Image.open(uploaded_file) 
        # uploaded_image = np.array(uploaded_image)
        # st.image(uploaded_file, channels='BGR')
        uploaded_image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        uploaded_image = cv2.imdecode(uploaded_image, 1)
        st.image(uploaded_file, channels='BGR')

        st.write('Image is uploaded')
        #Resize the Image according with your model
        uploaded_image = cv2.resize(uploaded_image,(224,224),interpolation = cv2.INTER_AREA)
        #Expand dim to make sure your img_array is (1, Height, Width , Channel ) before plugging into the model
        img_array  = np.expand_dims(uploaded_image, axis=0)
        #Check the img_array here
        #st.write(img_array)

        prediction = model.predict(img_array)

        # Preprocess your prediction , How are we going to get the label name out from the prediction
        # Now it's your turn to solve the rest of the code
        index = np.argmax(prediction.flatten())
        st.write('The banknote is ',class_names[index],'dong')
