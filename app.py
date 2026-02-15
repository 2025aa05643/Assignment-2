import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Mobile Price ML Dashboard",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Mobile Price Range Prediction Dashboard")
st.markdown("This application compares six machine learning models for multi-class classification.")

st.sidebar.header("⚙ Model Configuration")

# Sidebar model selection
model_name = st.sidebar.selectbox(
    "Choose a Classification Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

uploaded_file = st.file_uploader(
    "Upload CSV File (Must include 'price_range' column)",
    type=["csv"]
)


if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📂 Dataset Overview")
    st.write("Shape of dataset:", data.shape)
    st.write(data.head())

    if "price_range" not in data.columns:
        st.error("The dataset must contain a 'price_range' column.")
    else:
        X = data.drop("price_range", axis=1)
        y = data["price_range"]

        scaler = joblib.load("model/scaler.pkl")
        model = joblib.load(f"model/{model_name}.pkl")

        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        
        st.subheader("📊 Model Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy_score(y, predictions):.4f}")
        col2.metric("AUC (OvR)", f"{roc_auc_score(y, probabilities, multi_class='ovr'):.4f}")
        col3.metric("MCC", f"{matthews_corrcoef(y, predictions):.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Precision", f"{precision_score(y, predictions, average='weighted'):.4f}")
        col5.metric("Recall", f"{recall_score(y, predictions, average='weighted'):.4f}")
        col6.metric("F1 Score", f"{f1_score(y, predictions, average='weighted'):.4f}")

        
        st.subheader("📈 Classification Report")
        st.text(classification_report(y, predictions))


        st.subheader("📉 Confusion Matrix")

        cm = confusion_matrix(y, predictions)

        fig, ax = plt.subplots()
        cax = ax.matshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        fig.colorbar(cax)
        st.pyplot(fig)

else:
    st.info("Upload a dataset to begin evaluation.")
