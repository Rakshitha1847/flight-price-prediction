import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

st.set_page_config(page_title="Flight Price Prediction", layout="centered")
st.title("✈️ Flight Price Prediction (XGBoost)")

uploaded_file = st.file_uploader(
    "Upload Flight Price Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info(" Please upload the flight price CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

stops_map = {
    "non-stop": 0, "non stop": 0, "0": 0,
    "one": 1, "1": 1, "1 stop": 1,
    "two": 2, "2": 2, "2 stops": 2,
    "three": 3, "3": 3, "3 stops": 3
}

df["stops"] = (
    df["stops"]
    .astype(str)
    .str.lower()
    .map(stops_map)
)

df = df.dropna(subset=["stops"])
df["stops"] = df["stops"].astype(int)

categorical_cols = [
    "airline",
    "source_city",
    "destination_city",
    "departure_time",
    "arrival_time",
    "class"
]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

for col in df.columns:
    if col != "price":
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

X = df.drop("price", axis=1)
y = df["price"]

@st.cache_resource
def train_model(X, y):
    model = XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X, y)
    return model

model = train_model(X, y)

st.success(" Dataset loaded and model trained successfully")

st.subheader(" Select Flight Details")

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("Airline", encoders["airline"].classes_)
    source_city = st.selectbox("From (Source City)", encoders["source_city"].classes_)
    departure_time = st.selectbox("Departure Time", encoders["departure_time"].classes_)
    stops = st.selectbox("Stops", [0, 1, 2, 3])

with col2:
    flight_class = st.selectbox("Class", encoders["class"].classes_)
    destination_city = st.selectbox("To (Destination City)", encoders["destination_city"].classes_)
    arrival_time = st.selectbox("Arrival Time", encoders["arrival_time"].classes_)
    days_left = st.number_input("Days Left for Departure", min_value=1, step=1)

duration = st.slider("Duration (Hours)", 0.5, 24.0, 2.5)

if st.button(" Predict Flight Price"):

    input_data = pd.DataFrame([{
        "airline": encoders["airline"].transform([airline])[0],
        "source_city": encoders["source_city"].transform([source_city])[0],
        "destination_city": encoders["destination_city"].transform([destination_city])[0],
        "departure_time": encoders["departure_time"].transform([departure_time])[0],
        "arrival_time": encoders["arrival_time"].transform([arrival_time])[0],
        "stops": int(stops),
        "class": encoders["class"].transform([flight_class])[0],
        "duration": float(duration),
        "days_left": int(days_left)
    }])

    input_data = input_data[X.columns]

    prediction = model.predict(input_data)[0]

    st.markdown(
        f"""
        <div style="
            padding: 25px;
            border-radius: 12px;
            background-color: #e8f0fe;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            color: #0d47a1;
        ">
         Value Predicted<br><br>
        ₹ {int(prediction)}
        </div>
        """,
        unsafe_allow_html=True
    )
