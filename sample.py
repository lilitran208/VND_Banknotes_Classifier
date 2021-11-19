import streamlit as st
import pandas as pd

st.write('Lối nhỏ - Đen Vâu')
st.audio('media\Loi_nho.mp3')

menu = ['Home','About me', 'Read Data','Camera']
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

elif choice == 'Camera':
    