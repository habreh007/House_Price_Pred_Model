# ==============================
# House Price Prediction App
# ==============================

import streamlit as st
import numpy as np
import pandas as pd
import os
import time

from sklearn.ensemble import RandomForestRegressor

# ---------- Page Config ----------
st.set_page_config(
	page_title="House Price Predictor",
	page_icon="🏠",
	layout="centered"
)

# ---------- Styling ----------
st.markdown("""
<style>
.stApp { background-color: #f8faff; }
h1, h2 { color: #1d4ed8; }
div.stButton > button {
	background-color: #1e3a8a;
	color: white;
	font-weight: bold;
	border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load Dataset ----------
DATA_PATH = "data_folder/california_dataset.csv"

if not os.path.exists(DATA_PATH):
	st.error("❌ Dataset not found in data_folder")
	st.stop()

@st.cache_data
def load_data(path):
	return pd.read_csv(path)

df = load_data(DATA_PATH)

# ---------- SET TARGET COLUMN MANUALLY ----------
# 🔴 CHANGE THIS IF YOUR COLUMN NAME IS DIFFERENT
TARGET_COLUMN = "Target"   # <-- MOST COMMON
# TARGET_COLUMN = "price"
# TARGET_COLUMN = "HousePrice"

if TARGET_COLUMN not in df.columns:
	st.error(f"❌ Target column '{TARGET_COLUMN}' not found in dataset")
	st.stop()

# ---------- Train Model ----------
@st.cache_resource
def train_model(data):
	X = data.drop(TARGET_COLUMN, axis=1)
	y = data[TARGET_COLUMN]

	model = RandomForestRegressor(
		n_estimators=150,
		random_state=42,
		n_jobs=-1
	)
	model.fit(X, y)
	return model, X.columns

model, feature_cols = train_model(df)

# ---------- UI ----------
st.title("🏡 House Price Prediction")
st.write("Predict house prices using a Random Forest model.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
	MedInc = st.number_input("Median Income", 0.0, 20.0, 3.5)
	HouseAge = st.number_input("House Age", 0.0, 100.0, 20.0)
	AveRooms = st.number_input("Average Rooms", 0.0, 20.0, 5.0)
	AveBedrms = st.number_input("Average Bedrooms", 0.0, 10.0, 1.0)

with col2:
	Population = st.number_input("Population", 0.0, 50000.0, 1000.0)
	AveOccup = st.number_input("Average Occupancy", 0.0, 10.0, 3.0)
	Latitude = st.number_input("Latitude", 30.0, 45.0, 34.0)
	Longitude = st.number_input("Longitude", -130.0, -110.0, -118.0)

# ---------- Prediction ----------
if st.button("🔍 Predict House Price"):
	with st.spinner("Predicting..."):
		time.sleep(1)

		input_data = pd.DataFrame([[
			MedInc, HouseAge, AveRooms, AveBedrms,
			Population, AveOccup, Latitude, Longitude
		]], columns=feature_cols)

		pred = model.predict(input_data)[0]
		st.success(f"🏠 Estimated Price: **${pred * 100000:,.2f}**")

# ---------- Footer ----------
st.markdown("---")
st.markdown(
	"<div style='text-align:center; font-weight:600;'>Developed by 💙 Habib-ur-Rehman</div>",
	unsafe_allow_html=True
)
