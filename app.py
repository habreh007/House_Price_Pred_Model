# ---- imports and setup ----
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
import os

# ---- page configuration ----
st.set_page_config(
	page_title="House Price Predictor",
	page_icon="🏠",
	layout="centered",
	initial_sidebar_state="expanded"
)

# ---- custom style ----
st.markdown("""
	<style>
		.stApp {
			background-color: #f8faff;
			color: #1e3a8a;
		}
		h1, h2, h3, h4 {
			color: #1d4ed8;
		}
		.stTextInput > div > div > input, .stNumberInput input {
			border: 1px solid #1d4ed8;
			border-radius: 8px;
		}
		div.stButton > button:first-child {
			background-color: #1e3a8a;
			color: white;
			border-radius: 8px;
			font-weight: bold;
			padding: 0.5em 1.5em;
			transition: all 0.3s ease;
		}
		div.stButton > button:first-child:hover {
			background-color: #3b82f6;
			transform: scale(1.05);
		}
	</style>
""", unsafe_allow_html=True)

# ---- load dataset ----
data_path = os.path.join("data_folder", "california_dataset.csv")
if os.path.exists(data_path):
	df = pd.read_csv(data_path)
	st.sidebar.success("✅ Dataset loaded successfully from 'data_folder'")
else:
	st.sidebar.error("⚠️ Dataset file not found in 'data_folder'")
	df = None

# ---- load model ----
model_path = "house_price_model.pkl"
with open(model_path, 'rb') as file:
	model = pickle.load(file)

# ---- app title ----
st.title("🏡 House Price Prediction App")
st.write("Predict California house prices using a trained Random Forest model.")
st.markdown("---")

# ---- input form ----
st.header("Enter House Features")

col1, col2 = st.columns(2)

with col1:
	MedInc = st.number_input("Median Income", min_value=0.0, value=3.5)
	HouseAge = st.number_input("House Age", min_value=0.0, value=20.0)
	AveRooms = st.number_input("Average Rooms", min_value=0.0, value=5.0)
	AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, value=1.0)

with col2:
	Population = st.number_input("Population", min_value=0.0, value=1000.0)
	AveOccup = st.number_input("Average Occupancy", min_value=0.0, value=3.0)
	Latitude = st.number_input("Latitude", min_value=30.0, value=34.0)
	Longitude = st.number_input("Longitude", min_value=-130.0, value=-118.0)

st.markdown("---")

# ---- prediction ----
if st.button("🔍 Predict House Price"):
	with st.spinner("Predicting... Please wait ⏳"):
		time.sleep(1)
		input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
		prediction = model.predict(input_data)
		st.success(f"🏠 **Predicted House Price:** ${prediction[0]*100000:.2f}")

# ---- footer ----
st.markdown("---")
st.markdown(
	"""
	<div style='text-align: center; color: #1d4ed8; font-weight: 600; font-size: 16px;'>
	Developed by 💙 <b>Habib-ur-Rehman</b> 
	</div>
	""",
	unsafe_allow_html=True
)
