import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(layout="wide")

def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("data.csv")  # fallback default
    return data

def calculate_price(data):
    # Example heuristic to create a 'PRICE' column if missing
    # Use columns you expect to be present in the data
    # Adjust weights as you like or based on domain knowledge
    if all(col in data.columns for col in ['RM', 'LSTAT', 'CRIM']):
        data['PRICE'] = (
            data['RM'] * 50000
            - data['LSTAT'] * 1000
            - data['CRIM'] * 200
            + 100000
        )
    else:
        # Fallback: use sum of all numeric columns as dummy price
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        data['PRICE'] = data[numeric_cols].sum(axis=1)
    return data

def train_model(data):
    # Remove rows where PRICE is NaN or infinite
    data = data[data['PRICE'].notnull() & np.isfinite(data['PRICE'])]
    
    # Remove rows with NaNs in features as well
    data = data.dropna()
    
    X = data.drop('PRICE', axis=1)
    y = data['PRICE']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

def main():
    st.title("🏡 Boston Housing Price Predictor (XGBoost with Scaler)")

    # 📂 File Upload
    st.sidebar.header("Upload Dataset (CSV)")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    data = load_data(uploaded_file)

    if 'PRICE' not in data.columns:
        st.warning("'PRICE' column not found. Calculating 'PRICE' using heuristic.")
        data = calculate_price(data)

    model, scaler = train_model(data)

    # 👇 Sidebar input for prediction
    st.sidebar.header("Input Features for Prediction")
    input_features = {
        col: st.sidebar.number_input(
            col, 
            float(data[col].min()), 
            float(data[col].max()), 
            float(data[col].mean())
        )
        for col in data.columns if col != 'PRICE'
    }
    input_df = pd.DataFrame([input_features])

    st.write("### 🔍 Input Features")
    st.write(input_df)

    if st.button("📈 Predict Price"):
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        st.success(f"Predicted House Price: ${prediction:.2f}")

    # 📊 Correlation Heatmap
    st.subheader("🔗 Correlation Heatmap")
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax1)
    st.pyplot(fig1)

    # 💰 Price Distribution
    st.subheader("💰 Price Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(data['PRICE'], bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)

    # 🛏 Rooms vs Price
    if 'RM' in data.columns:
        st.subheader("🛏 Rooms vs. Price")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x=data['RM'], y=data['PRICE'], ax=ax3)
        ax3.set_xlabel("Average Number of Rooms (RM)")
        ax3.set_ylabel("House Price")
        st.pyplot(fig3)

    # 📉 LSTAT vs Price
    if 'LSTAT' in data.columns:
        st.subheader("📉 LSTAT (% Lower Status) vs. Price")
        fig5, ax5 = plt.subplots()
        sns.scatterplot(x=data['LSTAT'], y=data['PRICE'], ax=ax5)
        ax5.set_xlabel("% Lower Status Population (LSTAT)")
        ax5.set_ylabel("House Price")
        st.pyplot(fig5)

    # 📉 CRIM vs Price
    if 'CRIM' in data.columns:
        st.subheader("📉 CRIM (Crime Rate) vs. Price")
        fig6, ax6 = plt.subplots()
        sns.scatterplot(x=data['CRIM'], y=data['PRICE'], ax=ax6)
        ax6.set_xlabel("Per Capita Crime Rate (CRIM)")
        ax6.set_ylabel("House Price")
        st.pyplot(fig6)

    # 📊 Feature Importance
    st.subheader("📊 Feature Importance (XGBoost)")
    importances = model.feature_importances_
    feature_df = pd.DataFrame({
        'Feature': data.columns.drop('PRICE'),
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig4, ax4 = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_df, ax=ax4)
    st.pyplot(fig4)

if __name__ == '__main__':
    main()
