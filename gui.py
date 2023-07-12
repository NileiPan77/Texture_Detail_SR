import streamlit as st
import numpy as np
import cv2
import pyexr
from fourier_filtering import *
import tempfile
import os

st.set_page_config(layout='wide')

@st.cache_data
def load_image(uploaded_file):
    # save the file to local temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".exr") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_file_name = tmp.name

    # load the image
    exr = pyexr.open(tmp_file_name).get('B')[:, :, 0]

    # delete the temporary file
    os.remove(tmp_file_name)

    return exr

st.title("Texture Transfer GUI")

with st.sidebar:
    # Set up sliders for parameters
    r = st.slider("Low-pass filter cutoff frequency (r_low)", min_value=1, max_value=1000, value=180)
    r_high = st.slider("High-pass filter cutoff frequency (r_high)", min_value=1, max_value=1000, value=120)
    degree = st.slider("Butterworth filter order (degree)", min_value=1, max_value=20, value=1)
    degree_high = st.slider("Butterworth filter order (degree for high pass)", min_value=1, max_value=20, value=1)

if 'r' not in st.session_state:
    st.session_state['r'] = r
if 'r_high' not in st.session_state:
    st.session_state['r_high'] = r_high
if 'degree' not in st.session_state:
    st.session_state['degree'] = degree
if 'degree_high' not in st.session_state:
    st.session_state['degree_high'] = degree_high

if 'high_freq' not in st.session_state:
    st.session_state['high_freq'] = None
if 'low_freq' not in st.session_state:
    st.session_state['low_freq'] = None
if 'result' not in st.session_state:
    st.session_state['result'] = None

# Set up file loaders
if st.button('Re-upload files') or st.session_state['result'] is None:
    
    target_file = st.file_uploader("Choose target file", type=['exr'])
    base_file = st.file_uploader("Choose base file", type=['exr'])

    if target_file is not None:
        st.session_state['target_file'] = target_file
    if base_file is not None:
        st.session_state['base_file'] = base_file


# Perform Fourier filtering if files have been uploaded and high_freq, low_freq, and 
if  'base_file' in st.session_state and st.session_state['target_file'] and st.session_state['base_file'] and \
    (st.session_state['r'] != r or st.session_state['r_high'] != r_high \
        or st.session_state['degree'] != degree or st.session_state['degree_high'] != degree_high):

    target_exr = load_image(st.session_state['target_file'])
    base_exr = load_image(st.session_state['base_file'])
    hi_freq, low_freq, result = cufourier_filtering_3out(target_exr, base_exr, r, r_high, degree, degree_high)

    # save the images
    st.session_state['high_freq'] = hi_freq
    st.session_state['low_freq'] = low_freq
    st.session_state['result'] = result

    # save the parameters
    st.session_state['r'] = r
    st.session_state['r_high'] = r_high
    st.session_state['degree'] = degree

tab1, tab2, tab3 = st.tabs(["High Frequency", "Low Frequency", "Result"])

with tab1:
    if st.session_state['high_freq'] is not None:

        exposure_high_freq = st.slider("Exposure_high_freq", min_value=0.0, max_value=10.0, value=1.0)
        offset_high_freq = st.slider("Offset_high_freq", min_value=-1.0, max_value=1.0, value=0.0)
        gamma_high_freq = st.slider("Gamma_high_freq", min_value=0.1, max_value=10.0, value=2.2)

        new_hi_freq = exposure_high_freq * st.session_state['high_freq'] + offset_high_freq
        new_hi_freq = np.power(new_hi_freq, 1.0 / gamma_high_freq)

        if st.checkbox('normalize_high'):
            new_hi_freq = cv2.normalize(new_hi_freq, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        st.image(new_hi_freq, caption='high freq', width=1024, clamp=True)
    
with tab2:
    if st.session_state['low_freq'] is not None:

        exposure_low_freq = st.slider("Exposure_low_freq", min_value=0.0, max_value=10.0, value=1.0)
        offset_low_freq = st.slider("Offset_low_freq", min_value=-1.0, max_value=1.0, value=0.0)
        gamma_low_freq = st.slider("Gamma_low_freq", min_value=0.1, max_value=10.0, value=2.2)

        new_low_freq = exposure_low_freq * st.session_state['low_freq'] + offset_low_freq
        new_low_freq = np.power(new_low_freq, 1.0 / gamma_low_freq)
        if st.checkbox('normalize_low'):
            new_low_freq = cv2.normalize(new_low_freq, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        st.image(new_low_freq, caption='low freq', width=1024, clamp=True)

with tab3:
    if st.session_state['result'] is not None:

        exposure = st.slider("Exposure", min_value=0.0, max_value=10.0, value=1.0)
        offset = st.slider("Offset", min_value=-1.0, max_value=1.0, value=0.0)
        gamma = st.slider("Gamma", min_value=0.1, max_value=10.0, value=2.2)

        new_result = exposure * st.session_state['result'] + offset
        new_result = np.power(new_result, 1.0 / gamma)

        if st.checkbox('normalize_result'):
            new_result = cv2.normalize(new_result, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        st.image(new_result, caption='result', use_column_width='auto', clamp=True)