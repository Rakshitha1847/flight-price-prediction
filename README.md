# flight-price-prediction

Flight Price Prediction (XGBoost)
A machine learning web application built with Streamlit that predicts flight ticket prices using XGBoost regression. Users can upload their dataset, train the model, and make real-time predictions through an intuitive interface.

Features:

1.CSV Upload: Upload your flight price dataset directly through the web interface

2.Data Preprocessing: Automatic cleaning and encoding of categorical variables

3.XGBoost Model: High-performance gradient boosting for accurate price predictions

4.Interactive UI: User-friendly input controls for flight details

5.Real-time Prediction: Instant price estimation based on selected parameters

6.Visual Feedback: Clean, styled output display for prediction.

--Usage--

Run the Streamlit app

Open your browser and navigate to http://localhost:8501

Upload Dataset: Upload a CSV file containing flight price data with the following columns (example):

airline: Airline company name

source_city: Departure city

destination_city: Arrival city

departure_time: Time of departure

arrival_time: Time of arrival

stops: Number of stops (0, 1, 2, 3)

class: Economy/Business class

duration: Flight duration in hours

days_left: Days until departure

price: Target variable (ticket price)

Train Model: The app automatically processes the data and trains an XGBoost model

Make Predictions: Fill in the flight details using the input controls and click "Predict Flight Price"



