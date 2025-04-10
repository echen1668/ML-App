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
#np.random.seed(1000)
rstate = 12

# import module
import streamlit as st

from Common_Tools import wrap_text_excel, expand_cell_excel, grid_excel
from roctools import full_roc_curve, plot_roc_curve

# back button to return to main page
if st.button('Back'):
    st.switch_page("ML Interface.py")  # Redirect to the main back

# Title
st.title("Visualize ML Results")

st.write("This page helps you visualize the results of your ML model(s).")

# two options to decide what to do
left_column, right_column = st.columns(2)

# Upload and test model button
if left_column.button('Visualize the results of a single ML experiment'):
    st.switch_page("pages/Visualize_Results.py")  # Redirect to Testing_Single_Models.py

# Visualize ML Results button
if right_column.button('Compare the results of mutiple ML experiment'):
    st.switch_page("pages/Visualize_Multi_Results.py")  # Redirect to Testing_Multi_Models.py