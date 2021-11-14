import streamlit as st
import streamlit.components.v1 as stc

import pandas as pandas

import numpy as np
from pathlib import Path
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import matplotlib
matplotlib.use('Agg') # TkAgg
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import base64 
import pandas as pd
import altair as alt
import plotly.express as px
timestr = time.strftime("%Y%m%d-%H%M%S")		
from solar_ml import *
# from solar_dsm import *

html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Solar Power Forecasting </h1>
		
		</div>
		"""

# Setting Page icon
img = Image.open("24.jpg")
PAGE_CONFIG = {"page_title":"Solar Fx","page_icon":img,"layout":"centered"}
st.set_page_config(**PAGE_CONFIG)	


##For DSM Calculation
state_code_dict =  {'Andhra Pradesh':1,'Uttar Pradesh':1,'Madhya Pradesh':1,'Gujarat':2}
col7, col8, col9, cola, colb = st.columns(5)
with col9:
	st.image(img,width = 120,use_column_width=True)






def save_uploaded_file(uploadedfile):
	with open(os.path.join("C:/Users/Hp/Desktop/Applied AI/Kaggle/Solar Power Forecasting/Saving_files_here/Predicted_files",uploadedfile.name),"wb") as f:
		f.write(uploadedfile.getbuffer())
	return st.success("Saved file :{} in Saving_files_here/Model_input_files folder".format(uploadedfile.name))


def save_predicted_file(uploadedfile):
	with open(os.path.join("Saving_files_here/Predicted_files",uploadedfile.name),"wb") as f:
		f.write(uploadedfile.getbuffer())
	return st.success("Saved file :{} in Saving_files_here/Predicted_files folder".format(uploadedfile.name))

def csv_downloader(data):
	csvfile = data.to_csv()
	b64 = base64.b64encode(csvfile.encode()).decode()
	new_filename = "Forecast_file_{}_.csv".format(timestr)
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
	st.markdown(href,unsafe_allow_html=True)

def main():
	
	stc.html(html_temp)
	# menu = ['Generate Forecast','Forecast Analysis','Load Data','Comment Section']
	menu = ['Generate Forecast','Load Data','Comment Section']
	choice = st.sidebar.selectbox('Menu',menu)

	if choice == 'Generate Forecast':
		
		
		st.subheader('Single Block Prediction')
		col12,colp,colq,colr = st.columns(4)
		with col12:
			Plant = ['Plant 1','Plant 2','Plant 3','Plant 4']
			choice_plant = st.selectbox('Plant',Plant)
		with colp:
			AMBIENT_TEMPERATURE = st.number_input("Ambient Temperature(°C)")
		with colq:
			MODULE_TEMPERATURE = st.number_input("Module Temperature(°C)")
		with colr:
			IRRADIATION = st.number_input("Irradiation(kW/m²)",0.00,1.50)
		st.write('')
		st.write('')		
		if st.button('Predict'):
			
			single_value = np.array([AMBIENT_TEMPERATURE,MODULE_TEMPERATURE,IRRADIATION]).reshape(1,-1)
			model_single = load_model("Solar_forecast_model_final.pkl")
			prediction_single = model_single.predict(single_value)[0]

			st.write('Model Prediction: {} MW'.format(round((prediction_single/1000),2)))
		
		# Working with File Upload
		st.write('')
		st.write('')
		st.write('')
		st.subheader('Multiple Blocks Prediction')
		st.write('')
		
		collac, colla, collab = st.columns(3)
		with collac:
			Plant_ = ['Plant 1','Plant 2','Plant 3','Plant 4']
			choice_plan_ = st.selectbox('Plant',Plant_,key='001')

		with colla:
			fr_date = st.date_input('From')

		with collab:
			t_date = st.date_input('To')	
		uploaded_file = st.file_uploader("Choose a csv file", type="csv")
		
		if uploaded_file is not None:
			file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
			df = pd.read_csv(uploaded_file)
			df_n = df[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION']]
			st.write('Uploaded file')
			st.dataframe(df.iloc[:,1:6])
			#save_uploaded_file(uploaded_file)
		
		st.write('')
		st.write('')
		
		if st.button('Predict',key=2):
			prediction_df = run_ml_app(df_n)	
			prediction_df = pd.concat([df[['DATE_TIME','BLOCK']],pd.Series(prediction_df)/1000],axis=1)
			prediction_df.columns = ['DATE_TIME','BLOCK','PREDICTED GEN(MW)']
			

			with st.spinner('Model is working!...'):
				time.sleep(0.2)
				st.success('Forecast ready!')	
			st.dataframe(prediction_df)
			           
			st.write('')
			st.write('')
			csv_downloader(prediction_df)
			st.write('')
			st.write('Forecast')
			st.line_chart(prediction_df[['PREDICTED GEN(MW)']],use_container_width=True) 
			
		

	elif choice == 'Load Data':
		colc, cold, cole= st.columns([2.8,4,1])
		with cold:
			st.title('Load Data')

		Date = st.date_input('Select Date')
		st.write('')
		st.write('')
		if st.button('Load'):
			#Data = Date.dt.date.astype(str)
			df_data = pd.read_csv('final_dataset.csv',index_col=0)
			df_data['Date'] = pd.to_datetime(df_data['DATE_TIME']).dt.date.astype(str)
			#df_data['Date'] = df_data['Date'].apply(lambda x: str(x).replace('-','/'))
			Date = str(Date)
			
			data_display = df_data[df_data['Date']== Date].iloc[:,:-1].copy()
			st.dataframe(data_display)
			csv_downloader(data_display)

		
	else :
		colx, coly = st.columns([2,5])
		with coly:
			st.title('Comment Section')
		colg, colh = st.columns(2)
		with colg:
			Date2 = st.date_input('Select Date')
		with colh:
			menuh = ['Person 1','Person 2','Person 3']
			choiceh = st.selectbox('Shift Incharge',menuh)

		
		message = st.text_area("Enter Comment",height=100)
		st.write('')
		st.write('')
		
		if st.button('Submit'):
			comment_section = pd.read_csv('comment_df.csv',index_col=0)
			
			comment_section = comment_section.append({'Date':str(Date2),'Shift Incharge':choiceh,'Comment':message},ignore_index=True)
			comment_section.to_csv('comment_df.csv')
			st.success('Comment saved successfully!')
			st.write('')
			st.write('')
	
			csv_downloader(comment_section)
			

					

if __name__=='__main__':
	main()	
