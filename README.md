# 🚲 Intelligent Bike Rental Predictor

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)


A modern, interactive Machine Learning dashboard built with Streamlit to forecast bike rental demand in real-time. This project uses a Random Forest Regressor trained on historical weather and seasonal data to predict how many bikes will be rented under varying conditions.

 ### demo
  
     https://naveen-khan-bike-rental-prediction-app-tr5fau.streamlit.app/

## ✨ Features

*   **Real-time Predictions:** Adjust sliders for Temperature, Humidity, Wind Speed, and Season to see instant rental forecasts.
  
*   **Scenario Analysis:** Compares the predicted outcome against preset extreme weather scenarios (e.g., "Cold & Windy" vs. "Hot & Dry").
*   **Interactive Data Explorer:** Dive into historical data with Plotly-powered scatter plots and box plots. Filter data by season to discover underlying trends.
*   **Model Insights (Explainable AI):** 
    *   View **Feature Importances** to understand what drives demand (Spoiler: Temperature is key!).
    *   Evaluate model performance using Actual vs. Predicted scatter plots and Residual Distribution histograms.
*   **Dynamic Metrics:** Tracks test-set $R^2$ Score and Mean Absolute Error (MAE) evaluated on an 80/20 train-test split.

## 🛠️ Tech Stack

*   **Frontend/App Framework:** [Streamlit](https://streamlit.io/)
*   **Machine Learning:** [scikit-learn](https://scikit-learn.org/) (RandomForestRegressor)
*   **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Data Visualization:** [Plotly Express & Graph Objects](https://plotly.com/python/)

## 🚀 How to Run Locally

Follow these steps to get the app running on your local machine.

### 1. Clone the repository
```bash
git clone https://github.com/your-username/bike-rental-predictions.git
cd bike-rental-predictions
```

### 2. Install dependencies
Make sure you have Python installed. Then, install the required packages:
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```
The application will automatically open in your default web browser at `http://localhost:8501`.

## 📁 Project Structure

```text
bike_rental_predictions/
├── app.py                      # Main Streamlit application file
├── bike_rental_100_rows.csv    # Dataset containing historical rental records
├── Bike-rent-peditions.ipynb   # Jupyter Notebook for initial EDA and model prototyping
└── README.md                   # This documentation file
```

## 🧠 About the Model

The prediction engine is powered by a **Random Forest Regressor** with 200 estimators.
The model is trained on four normalized features:
1.  **Temperature** ($0$ = freezing cold, $1$ = hottest recorded)
2.  **Humidity** ($0$ = bone dry, $1$ = fully saturated)
3.  **Wind Speed** ($0$ = still, $1$ = storm-force wind)
4.  **Season** (Categorical: Spring, Summer, Fall, Winter)

During startup, the app dynamically trains the model and calculates validation metrics ($R^2$ and MAE) on a 20% hold-out test set to ensure the predictions reflect real-world accuracy without overfitting.

---
*Built with ❤️ for Data Science and Machine Learning enthusiasts.*
