import time

import pandas as pd
import numpy as np
import sklearn as scikit_learn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import csv
#import magic 
import pickle
import random 
from random import randint
from random import uniform
from pathlib import Path
import json
from scipy import stats
import os
import joblib as joblib
from joblib import dump, load
import pymongo
#np.random.seed(1000)
rstate = 12

# import module
import streamlit as st

# set page configuation
#st.set_page_config(page_title="ML Interface", layout="wide")

# Title
#st.title("ML Interface")
st.markdown("<h1 style='text-align: center;'>ML Interface</h1>", unsafe_allow_html=True)

# description
st.markdown("""
    <div style="text-align: center;">
        <p>This webpage is an interface where you can store and upload results from a Machine Learning experiment.</p>
        <p>Here, you upload and store models in a database and can load them to test and generate results.</p>
        <p>You can then load the results and visualize them in a simplistic manner.</p>
        <p>This can include an ROC Curve, a results table, a confusion matrix, graphs, etc.</p>
    </div>
""", unsafe_allow_html=True)

st.write("")  # Add for more space
st.write("")
st.write("")

# two options to decide what to do
left_column, right_column = st.columns(2)

# Upload and test model button
if left_column.button('Upload and Test ML Model'):
    st.switch_page("pages/Testing_Models.py")  # Redirect to upload_test.py

# Visualize ML Results button
if right_column.button('Visualize ML Results'):
    st.switch_page("pages/Visualize_Options.py")  # Redirect to visualize_results.py
