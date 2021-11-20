import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

st.write('Lối nhỏ - Đen Vâu')
st.audio('media\Loi_nho.mp3')

menu = ['Home','About me', 'Read Data','Capture From Webcam']
choice = st.sidebar.selectbox('Menu',menu)

if choice == 'Home':
    st.header('This is a Dog blog')
    st.write('Hello guys')
    st.image('media\dog-beach-lifesaver.png')
    st.video('media\dogs.mp4')

    col1,col2 = st.columns(2)

    with col1:
        dog_name = st.text_input('What is your dog name')
        st.write('Your dog name: ',dog_name)
    with col2:
        age= st.slider('Dog age',min_value =1 , max_value = 20)
        st.write('Age: ', age)
elif choice == 'Read Data':
    df = pd.read_csv('media\AB_NYC_2019.csv')
    st.dataframe(df)
elif choice == 'About me':
    fileUp = st.file_uploader('Upload file',type = ['jpeg','png','jpg'])
    st.image(fileUp)

elif choice == 'Capture From Webcam':

    #Load your model and check create the class_names list
    # Model_Path = '____________'
    # class_names = [_________________________]
    # model = tf.keras.load_model(Model_Path)

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

        