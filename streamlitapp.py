
import streamlit as st
import os
import tempfile
import imageio 
import numpy as np

import tensorflow as tf 
from utils import load_video, load_data, num_to_char
from modelutil import load_model


st.set_page_config(layout='wide')

with st.sidebar:
    st.image('C:/Users/gayat/OneDrive/Desktop/mini/logo.png')
    st.title('AUNAR')
    st.info('Aunar is a web app that transcribes speech by analyzing lip movements. Our aim is to unite accessibility and innovation to make this a valuable tool for diverse users.')

video_file = st.file_uploader("Upload a video file", type=["mpg","mp4","mov"])

col1, col2 = st.columns(2)

if video_file is not None:
    bytes_data = video_file.read()
    original_filename = video_file.name
    output_filename = f"converted_{original_filename}.mp4"

    with col1:
        st.text('Preview')
       
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(bytes_data)
            temp_filename = temp_file.name

            exit_code = os.system(f'ffmpeg -i "{temp_filename}" -vcodec libx264 "{output_filename}" -y')
            if exit_code != 0:
                st.error(f"FFmpeg conversion failed with exit code: {exit_code}")
            else:
                st.video(output_filename)
        

    with col2:
        st.text('Translation')
       
        video= load_video(output_filename)
        
       
        imageio.mimsave('animation.gif', video, fps=10)
        

       
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    

        # Convert prediction to text
        
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        


