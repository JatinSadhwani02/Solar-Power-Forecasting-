import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('Agg')
		

def visualize():
    uploaded_file_report = st.file_uploader("Upload Forecast Report file", type="csv")
    if uploaded_file_report is not None:
        file_details = {"FileName":uploaded_file_report.name,"FileType":uploaded_file_report.type}
        df_all = pd.read_csv(uploaded_file_report)  
    features = ("Actual Gen (MW)","Predicted Gen (MW)","Deviation %")
    selected_feat = st.multiselect("Features",features,default=["Predicted Gen (MW)","Actual Gen (MW)"])

    if st.button('Plot'):
        st.dataframe(df_all)
        st.line_chart(df_all[["Predicted Gen (MW)","Actual Gen (MW)"]],use_container_width=True)

      

	





