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
    if all(col in data.columns for col in ['RM', 'LSTAT', 'CRIM']):
        data['PRICE'] = (
            data['RM'] * 50000
            - data['LSTAT'] * 1000
            - data['CRIM'] * 200
            + 100000
        )
    else:
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        data['PRICE'] = data[numeric_cols].sum(axis=1)
    return data

def train_model(data):
    data = data[data['PRICE'].notnull() & np.isfinite(data['PRICE'])]
    data = data.dropna()

    X = data.drop('PRICE', axis=1)
    y = data['PRICE']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

def generate_dynamic_insights(data, target='PRICE', threshold=0.5):
    insights = []
    if target not in data.columns:
        return ["Target column not found."]
    
    corr_matrix = data.corr()[target].drop(target).sort_values(key=lambda x: abs(x), ascending=False)
    
    for feature, corr in corr_matrix.items():
        if abs(corr) >= threshold:
            direction = "increases" if corr > 0 else "decreases"
            insights.append(
                f"- **{feature}** has a correlation of `{corr:.2f}` with **{target}**. "
                f"This likely **{direction}** the housing price."
            )
    
    if not insights:
        insights.append("No strong correlations (>|0.5|) found between features and the target price.")
    
    return insights


def main():
    st.title("🏡 Boston Housing Price Predictor (XGBoost with Scaler)")

    st.markdown("""
        This interactive dashboard allows you to explore and predict housing prices using 
        an XGBoost Regressor trained on a dataset of Boston homes. You can:
        - Upload your own dataset
        - View data distributions and relationships
        - Predict house prices using adjustable input sliders
    """)

    # 📂 File Upload
    st.sidebar.header("Upload Dataset (CSV)")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    data = load_data(uploaded_file)

    if 'PRICE' not in data.columns:
        st.warning("'PRICE' column not found. Calculating 'PRICE' using heuristic.")
        data = calculate_price(data)

    # 📄 Show data
    st.subheader("📄 Dataset Preview")
    st.dataframe(data.head())

    st.markdown("### 🧾 Dataset Summary Statistics")
    st.dataframe(data.describe())

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

    # 📊 Enhanced Correlation Heatmap
    st.subheader("🔗 Feature Correlation Heatmap")
    fig1 = plt.figure(figsize=(14, 10))
    mask_matrix = np.triu(data.corr())
    sns.heatmap(data.corr(), annot=True, mask=mask_matrix, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap", fontsize=18)
    plt.tight_layout()
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
        fig4, ax4 = plt.subplots()
        sns.scatterplot(x=data['LSTAT'], y=data['PRICE'], ax=ax4)
        ax4.set_xlabel("% Lower Status Population (LSTAT)")
        ax4.set_ylabel("House Price")
        st.pyplot(fig4)

    # 📉 CRIM vs Price
    if 'CRIM' in data.columns:
        st.subheader("📉 CRIM (Crime Rate) vs. Price")
        fig5, ax5 = plt.subplots()
        sns.scatterplot(x=data['CRIM'], y=data['PRICE'], ax=ax5)
        ax5.set_xlabel("Per Capita Crime Rate (CRIM)")
        ax5.set_ylabel("House Price")
        st.pyplot(fig5)

    # 📊 Feature Importance
    st.subheader("📊 Feature Importance (XGBoost)")
    importances = model.feature_importances_
    feature_df = pd.DataFrame({
        'Feature': data.columns.drop('PRICE'),
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig6, ax6 = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_df, ax=ax6)
    st.pyplot(fig6)

    # 📌 Dynamic Key Insights
    st.subheader("📌 Key Insights")
    insights = generate_dynamic_insights(data)
    for insight in insights:
        st.markdown(insight)


if __name__ == '__main__':
    main()
