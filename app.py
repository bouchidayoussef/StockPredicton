import tensorflow as tf
import streamlit as st
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import io
import time
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import os
from page import welcome, about, our_data, realtime, predict_prophet, predict_lstm


def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
                st.session_state["username"] in st.secrets["passwords"]
                and st.session_state["password"]
                == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True


if check_password():
    st.markdown("""
        <style>
        .reportview-container {
            background: url("https://www.desktopbackground.org/download/2560x1600/2010/12/16/127022_light-grey-backgrounds-hd_2560x1600_h.jpg")
        }
       .sidebar .sidebar-content {
            background: url("")
        }
        </style>
        """,
                unsafe_allow_html=True
                )

    add_selectbox = st.sidebar.selectbox(
        'Choose a page',
        ('Welcome', 'About Neural Networks', 'Our Data', 'Real-Time Stock', 'Prediction Using a RNN (lstm)', 'Prediction '
                                                                                                             'Using '
                                                                                                             'fbProphet')
    )

    if add_selectbox == 'Welcome':
        welcome()
    if add_selectbox == 'About Neural Networks':
        about()
    if add_selectbox == 'Our Data':
        our_data()
    if add_selectbox == 'Real-Time Stock':
        realtime()
    if add_selectbox == 'Prediction Using a RNN (lstm)':
        predict_lstm()
    if add_selectbox == 'Prediction Using fbProphet':
        predict_prophet()

