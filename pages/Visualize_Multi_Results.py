import time
import psutil
import pandas as pd
import numpy as np
import sklearn as scikit_learn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
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
import plotly.express as px
#np.random.seed(1000)
rstate = 12

# import module
import streamlit as st

from Common_Tools import wrap_text_excel, expand_cell_excel, grid_excel
from roctools import full_roc_curve, plot_roc_curve

def plot_roc(data, options):
    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(12, 8))

    # go throught each outcome to plot its ROC curve
    for option in options:
        name, outcome = option.split('-', 1)
        if "evaluation" in list(data[name][outcome].keys()):
            values = data[name][outcome]["evaluation"] # get the ground truths and probs for the specified outcome
        else: 
            values = data[name][outcome]

        y_true = values["Ground Truths"]
        y_prob = values["Probability Scores"]

        # calcaute the AUROC
        fpr, tpr, _ = roc_curve(y_true, y_prob[1])
        roc_auc = auc(fpr, tpr)

        res, res_array = full_roc_curve(y_true.to_numpy(), y_prob[1].to_numpy())

        # Extract Confidence Interval values
        try:
            auc_ci_low = values["AUROC CI Low"]
            auc_ci_high = values["AUROC CI High"]
        except: # calcuate them if not aviavble in data
            auc_ci_low = res['auc_cilow']
            auc_ci_high = res['auc_cihigh']

        # get other needed metrics
        specificity = res_array['tnr']
        sensitivity = res_array['tpr']

        # plot the ROC Curve
        ax.plot(fpr, tpr, label=f'{option} (AUC = {roc_auc:.4f} [{auc_ci_low:.4f}, {auc_ci_high:.4f}])', linewidth=2)

        # get the CI
        ax.fill_between(1-specificity, res_array['tpr_low'], res_array['tpr_high'], alpha=.2)

    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title("ROC Curves", fontsize=16)
    ax.legend(loc="lower right", fontsize=11)

    st.pyplot(fig)  # Use Streamlit's function to display the plot

# back button to return to main page
if st.button('Back'):
    st.switch_page("pages/Visualize_Options.py")  # Redirect to the main back

# Title
st.title("Visualize ML Results")

st.write("This page helps you visualize the results of your ML model(s).")
st.write("")  # Add for more space
st.write("")

# Upload and look at a results table
st.markdown("<h2 style='text-align: center;'>Visualize the Results Table</h2>", unsafe_allow_html=True)

# File uploader for ML Results table
uploaded_file = st.file_uploader("Upload a results table. (Must be either an Excel or CSV file)")

# Initialize session state for df_total
if "df_total" not in st.session_state:
    st.session_state.df_total = pd.DataFrame()

if uploaded_file is not None:
    try:
        # Determine file type and read accordingly
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            file_name = uploaded_file.name.replace(".csv", "")
        else:
            df = pd.read_excel(uploaded_file)
            file_name = uploaded_file.name.replace(".xlsx", "")

        # add a column to dataframe with the name of the experiment
        df.insert(0, "Exp_Name", file_name)

        # add error margins for AUROC Scores
        df["Upper_CI_Gap"] = df["AUROC CI Upper"] - df["AUROC Score"]
        df["Lower_CI_Gap"] = df["AUROC Score"] - df["AUROC CI Lower"]

        df["Upper_CI_Gap (Train)"] = df["AUROC CI Upper (Train)"] - df["AUROC Score (Train)"]
        df["Lower_CI_Gap (Train)"] = df["AUROC Score (Train)"] - df["AUROC CI Lower (Train)"]

        # Prevent adding the same file multiple times after rerun
        if st.session_state.df_total.empty or file_name not in st.session_state.df_total['Exp_Name'].values:
             # add to dataframe
            st.session_state.df_total = pd.concat([st.session_state.df_total, df])

    except Exception as e:
        st.error(f"Error loading file: {e}")

# button to reset the table
if st.button('Clear Table'):
    st.session_state.df_total = pd.DataFrame()
    uploaded_file = None

if len(st.session_state.df_total) != 0:
    # Display the DataFrame
    st.write("### ML Results Table:")
    st.dataframe(st.session_state.df_total)

      # List of all column names excluding the specified ones
    columns_to_exclude = ['Exp_Name', 'Outcome', 'AUROC CI Lower', 'AUROC CI Upper', 'AUROC CI Lower (Train)', 'AUROC CI Upper (Train)', 
                          'Cutoff value', 'Best Model', 'TN', 'TP', 'FN', 'FP', 'P', 'N', 'P (Train)', 'N (Train)',
                          'Upper_CI_Gap', 'Lower_CI_Gap', "Upper_CI_Gap (Train)", "Lower_CI_Gap (Train)", "AUROC CI Upper (Train)", "AUROC CI Lower (Train)"]
    options = [col for col in st.session_state.df_total.columns if col not in columns_to_exclude]
        
    # Dropdown to select the metric to disply in a barchart
    metric = st.selectbox("Select a Metric", options)

    # error bars
    error_y = None # default
    error_y_minus = None

    if metric == "AUROC Score":
        error_y = "Upper_CI_Gap"
        error_y_minus = "Lower_CI_Gap"

    elif metric == "AUROC Score (Train)":
        error_y = "Upper_CI_Gap (Train)"
        error_y_minus = "Lower_CI_Gap (Train)"

    fig = px.bar(
        st.session_state.df_total,
        x="Outcome",
        y=metric,
        color="Exp_Name",
        barmode="group",
        title=f"{metric} for Each Outcome",
        text_auto=True,  # Adds values on top of bars
        error_y=error_y,
        error_y_minus=error_y_minus
    )

    # Set y-axis range from 0 to 1
    fig.update_layout(yaxis=dict(range=[0, 1]))

    # Rotate text labels to be horizontal and limit decimals to 2 places
    fig.update_traces(
        textangle=0,  # Horizontal text
        texttemplate="%{y:.3f}",  # Format values to 2 decimal places
        textposition="inside"

    )


    # Add hover tooltip to show confidence intervals

    # Show the plot
    st.plotly_chart(fig)

st.write("")  # Add for more space
st.write("")

# Upload and look at a results table
st.markdown("<h2 style='text-align: center;'>Visualize the ROC Curve</h2>", unsafe_allow_html=True)

# File uploader for Groud Truths and Probablites
uploaded_values = st.file_uploader("Upload the Groud Truths and Probablites.")

# Initialize session state for df_total
if "outcome_dic_total" not in st.session_state:
    st.session_state.outcome_dic_total = {}

# button to reset the dictonary
if st.button('Clear Values'):
    st.session_state.outcome_dic_total = {}
    list_of_outcomes = []
    outcome_options= []
    uploaded_values = None

if uploaded_values is not None:
    try:
        # Determine file type and read accordingly
        outcome_dic = joblib.load(uploaded_values)

        name = uploaded_values.name.replace(".joblib", "")

        # add outcome_dic to st.session_state.outcome_dic_total
        if len(list(st.session_state.outcome_dic_total.keys())) == 0 or name not in list(st.session_state.outcome_dic_total.keys()):
            st.session_state.outcome_dic_total[name] = outcome_dic
        
        list_of_outcomes = [
            f"{exp}-{outcome}"
            for exp, outcome in st.session_state.outcome_dic_total.items()
            for outcome in st.session_state.outcome_dic_total[name].keys()
        ]

        # Initialize session state variable
        if "show_values_outcome_dic" not in st.session_state:
            st.session_state.show_values_outcome_dic = False

        # Button to display values of outcome_dic
        if st.button('Display the Values'):
            st.session_state.show_values_outcome_dic = True  # Set state to show values

        # Button to hide values (appears only when values are shown)
        if st.session_state.show_values_outcome_dic:
            st.write(st.session_state.outcome_dic_total)
            if st.button('Hide the Values'):
                st.session_state.show_values_outcome_dic = False  # Reset state to hide values
                st.rerun()  # Refresh the page to update UI


        st.title("ROC Curve Analysis")

        # Select multiple outcomes for the ROC curve plot
        outcome_options = st.multiselect("Select Outcomes to Plot", list_of_outcomes)

        if outcome_options:
            plot_roc(st.session_state.outcome_dic_total, outcome_options)
    except Exception as e:
        st.error(f"Error loading file: {e}")