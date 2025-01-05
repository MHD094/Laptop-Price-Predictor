import streamlit as st
import pickle
import numpy as np
import pandas as pd

# import the model
mp = pickle.load(open('model.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))
label_encoder = pickle.load(open('label_encoder.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# product
product = st.selectbox('Product',df['Product'].unique())

# type of laptop
type_laptop = st.selectbox('Type',df['TypeName'].unique())

# Inches
Inches = float(st.selectbox('Inches',df['Inches'].unique()))

# ScreenResolution
ScreenResolution = st.selectbox('ScreenResolution',df['ScreenResolution'].unique())

# CPU_Company
CPU_Company = st.selectbox('CPU_Company',df['CPU_Company'].unique())

# CPU_Type
CPU_Type = st.selectbox('CPU_Type',df['CPU_Type'].unique())

# CPU_Frequency (GHz)
CPU_Frequency = float(st.selectbox('CPU_Frequency',df['CPU_Frequency (GHz)'].unique()))

# RAM (GB)
RAM = int(st.selectbox("RAM (GB)", df['RAM (GB)'].unique()))

# Memory
Memory = st.selectbox('Memory',df['Memory'].unique())

# GPU_Company
GPU_Company = st.selectbox('GPU_Company',df['GPU_Company'].unique())

# GPU_Type
GPU_Type = st.selectbox('GPU_Type',df['GPU_Type'].unique())

# OpSys	
OpSys = st.selectbox('OpSys',df['OpSys'].unique())

# Weight (kg)
Weight = float(st.selectbox('Weight',df['Weight (kg)'].unique()))

if st.button('Prediction'):
    # transform the data
    data = pd.DataFrame({
        'Company': [company],
        'Product': [product],
        'TypeName': [type_laptop],
        'Inches': [Inches],
        'ScreenResolution': [ScreenResolution],
        'CPU_Company': [CPU_Company],
        'CPU_Type': [CPU_Type],
        'CPU_Frequency (GHz)': [CPU_Frequency],
        'RAM (GB)': [RAM],
        'Memory': [Memory],
        'GPU_Company': [GPU_Company],
        'GPU_Type': [GPU_Type],
        'OpSys': [OpSys],
        'Weight (kg)': [Weight]
    })

    data = pd.DataFrame(data)
    for column in ['Company', 'Product', 'TypeName','ScreenResolution', 'CPU_Company','CPU_Type' ,'Memory','GPU_Company','GPU_Type', 'OpSys']:
        data[column] = label_encoder[column].transform(data[column])
    predicted_price = mp.predict(data)
    st.subheader(f"Predicted price: **(€){predicted_price[0]:.2f}**")
