import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="E-Commerce Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Explainable Customer Churn Prediction Dashboard")
st.write("Using XGBoost, Customer Segmentation and Machine Learning")

# ============================================================
# LOAD FILES
# ============================================================

@st.cache_data
def load_data():
    cluster_summary = pd.read_csv("customer_segment_summary.csv")
    importance_df = pd.read_csv("xgboost_feature_importance.csv")
    results = pd.read_csv("model_comparison_results.csv")
    return cluster_summary, importance_df, results

cluster_summary, importance_df, results = load_data()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Project Overview",
        "Model Performance",
        "Feature Importance",
        "Customer Segmentation",
        "Business Insights",
        "Live Prediction"
    ]
)

# ============================================================
# PROJECT OVERVIEW
# ============================================================

if page == "Project Overview":
    st.header("Project Overview")

    st.write("""
    This project predicts customer churn for an e-commerce business and segments customers based on their purchasing and behavioral patterns.

    The main goal is to help e-commerce companies:
    - Identify customers who are likely to churn
    - Understand important churn-driving factors
    - Segment customers into meaningful business groups
    - Improve customer retention and marketing strategies
    """)

    st.subheader("Project Workflow")

    st.write("""
    1. Load E-Commerce Dataset  
    2. Clean and preprocess data  
    3. Convert order-level data into customer-level data  
    4. Create derived churn label  
    5. Train Logistic Regression, Random Forest and XGBoost models  
    6. Evaluate model performance  
    7. Explain churn drivers using feature importance and SHAP  
    8. Segment customers using K-Means clustering  
    9. Generate business insights for customer retention  
    """)

    st.info("""
    Note: Large output files such as full customer-level predictions and high-risk customer lists are not included in this GitHub repository due to file size limitations. 
    This dashboard uses summarized model outputs for deployment.
    """)

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":
    st.header("Model Performance")

    st.write("This section compares the performance of Logistic Regression, Random Forest and XGBoost models.")

    st.dataframe(results, use_container_width=True)

    metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    available_metrics = [col for col in metric_columns if col in results.columns]

    fig = px.bar(
        results,
        x="Model",
        y=available_metrics,
        barmode="group",
        title="Model Performance Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    if "ROC AUC" in results.columns:
        best_model = results.sort_values(by="ROC AUC", ascending=False).iloc[0]
        st.success(
            f"Best Model: {best_model['Model']} with ROC AUC = {best_model['ROC AUC']:.4f}"
        )

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "Feature Importance":
    st.header("Feature Importance")

    st.write("These features influenced the XGBoost churn prediction model the most.")

    top_n = st.slider("Select number of features", 5, 30, 15)

    top_features = importance_df.head(top_n)

    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Important Features"
    )

    fig.update_layout(yaxis=dict(autorange="reversed"))

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(top_features, use_container_width=True)

    st.warning("""
    Important Academic Note: If `days_since_last_order` appears as the top feature, it may indicate target leakage because the churn label was created using this feature.
    For research-grade improvement, this feature should be removed and the model should be retrained.
    """)

# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif page == "Customer Segmentation":
    st.header("Customer Segmentation")

    st.write("Customers were segmented using K-Means clustering based on behavioral and revenue-related features.")

    st.dataframe(cluster_summary, use_container_width=True)

    if "segment_name" in cluster_summary.columns and "total_customers" in cluster_summary.columns:
        fig = px.bar(
            cluster_summary,
            x="segment_name",
            y="total_customers",
            title="Customer Count by Segment"
        )

        st.plotly_chart(fig, use_container_width=True)

    if "avg_revenue" in cluster_summary.columns and "churn_rate" in cluster_summary.columns:
        fig2 = px.scatter(
            cluster_summary,
            x="avg_revenue",
            y="churn_rate",
            size="total_customers" if "total_customers" in cluster_summary.columns else None,
            color="segment_name" if "segment_name" in cluster_summary.columns else None,
            hover_name="segment_name" if "segment_name" in cluster_summary.columns else None,
            title="Segment Revenue vs Churn Rate"
        )

        st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# BUSINESS INSIGHTS
# ============================================================

elif page == "Business Insights":
    st.header("Business Insights")

    st.subheader("Key Insights from the Project")

    st.write("""
    Based on the model output and customer segmentation, the following insights can be derived:
    
    1. Customer inactivity is the strongest churn indicator.
    2. Customers with high churn probability should be targeted with retention campaigns.
    3. Customer segmentation helps identify groups such as loyal customers and high-risk customers.
    4. XGBoost performed strongly compared to baseline models.
    5. Explainable AI helps understand why customers are predicted to churn.
    """)

    st.subheader("Recommended Retention Strategies")

    st.write("""
    - Send personalized discount offers to high-risk customers.
    - Provide loyalty rewards to valuable customers.
    - Use email or campaign-based re-engagement for inactive customers.
    - Monitor customers with low session duration and low order frequency.
    - Improve customer satisfaction for users with negative review sentiment.
    """)

    st.info("""
    Full customer-level prediction and high-risk customer list can be generated from the Kaggle notebook and stored externally using Google Drive, Kaggle Dataset, or Git LFS.
    """)

# ============================================================
# LIVE PREDICTION
# ============================================================

elif page == "Live Prediction":
    st.header("Live Customer Churn Prediction")

    st.write("""
    Upload a new customer CSV file using the same format as `new_customer_template.csv`.
    The model will predict churn probability and risk level.
    """)

    try:
        model = joblib.load("xgboost_churn_model.pkl")
        model_features = joblib.load("model_features.pkl")
    except FileNotFoundError:
        st.error("""
        Model files are missing. Please upload these files to GitHub:
        - xgboost_churn_model.pkl
        - model_features.pkl
        - new_customer_template.csv
        """)
        st.stop()

    try:
        template_df = pd.read_csv("new_customer_template.csv")

        st.download_button(
            label="Download Prediction Template",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="new_customer_template.csv",
            mime="text/csv"
        )

        st.subheader("Prediction Template Preview")
        st.dataframe(template_df.head(), use_container_width=True)

    except FileNotFoundError:
        st.warning("new_customer_template.csv is missing. Please upload it to GitHub.")

    uploaded_file = st.file_uploader("Upload New Customer CSV", type=["csv"])

    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(new_data.head(), use_container_width=True)

        for col in model_features:
            if col not in new_data.columns:
                new_data[col] = 0

        new_data = new_data[model_features]

        churn_prob = model.predict_proba(new_data)[:, 1]
        churn_pred = model.predict(new_data)

        result_df = pd.DataFrame()
        result_df["Customer_No"] = range(1, len(new_data) + 1)
        result_df["Churn_Prediction"] = churn_pred
        result_df["Churn_Probability"] = churn_prob

        def risk_level(prob):
            if prob >= 0.70:
                return "High Risk"
            elif prob >= 0.40:
                return "Medium Risk"
            else:
                return "Low Risk"

        result_df["Risk_Level"] = result_df["Churn_Probability"].apply(risk_level)

        st.subheader("Prediction Results")
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="live_churn_prediction_results.csv",
            mime="text/csv"
        )