
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# Locate the project folder
project_path = Path(__file__).resolve().parent

# Load the trained Random Forest model
model_path = project_path / "models" / "tuned_random_forest.pkl"
model = joblib.load(model_path)

# Load default values for the remaining model features
defaults_path = project_path / "models" / "input_defaults.json"

with open(defaults_path, "r") as file:
    default_values = json.load(file)


# Customize sidebar appearance
st.markdown("""
<style>
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    font-size: 18px;
    padding: 12px 8px;
    margin-bottom: 12px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)


# Sidebar title
st.sidebar.markdown(
    "<h1 style='font-size: 32px; font-weight: 800;'>🏨 Smart Hotel ML</h1>",
    unsafe_allow_html=True
)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔍 Prediction", "📊 Model Performance", "📈 Feature Insights"]
)


# ---------------- HOME PAGE ----------------

if page == "🏠 Home":

    st.title("🏨 Smart Hotel Booking Cancellation Predictor")

    st.write(
        "Welcome to the Smart Hotel Booking Cancellation Prediction "
        "and Analytics application!"
    )

    st.info(
        "This application uses a trained Random Forest model "
        "to predict whether a hotel booking is likely to be cancelled."
    )

    st.subheader("✨ What Can You Explore?")

    st.write("🔍 **Prediction:** Predict booking cancellation using hotel details.")

    st.write("📊 **Model Performance:** View accuracy, precision, recall, and F1 score.")

    st.write("📈 **Feature Insights:** Discover important cancellation factors.")

    st.success("👈 Select a section from the sidebar to get started!")
    
    # Display project overview
    st.subheader("📌 Project Overview")

    
    # Display project overview cards with custom font sizes
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="text-align:center; padding:15px 5px;">
            <div style="font-size:19px; font-weight:600;">🤖 ML Model</div>
            <div style="font-size:16px; margin-top:10px;">Random Forest</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align:center; padding:15px 5px;">
            <div style="font-size:19px; font-weight:600;">🎯 Accuracy</div>
            <div style="font-size:16px; margin-top:10px;">85.22%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="text-align:center; padding:15px 5px;">
            <div style="font-size:19px; font-weight:600;">🏨 Dataset</div>
            <div style="font-size:16px; margin-top:10px;">119K Bookings</div>
        </div>
        """, unsafe_allow_html=True)




# ---------------- PREDICTION PAGE ----------------

elif page == "🔍 Prediction":

    st.title("🔍 Booking Cancellation Prediction")

    st.subheader("📝 Enter Booking Details")

    # Select hotel type
    hotel = st.selectbox(
        "Select Hotel Type",
        ["City Hotel", "Resort Hotel"]
    )

    # Enter booking lead time
    lead_time = st.number_input(
        "Lead Time (Number of Days Before Arrival)",
        min_value=0,
        max_value=737,
        value=30
    )

    # Select deposit type
    deposit_type = st.selectbox(
        "Select Deposit Type",
        ["No Deposit", "Non Refund", "Refundable"]
    )

    # Enter special requests
    special_requests = st.number_input(
        "Number of Special Requests",
        min_value=0,
        max_value=5,
        value=0
    )

    # Enter room price
    adr = st.number_input(
        "Average Daily Rate (Room Price)",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    # Show model input features
    with st.expander("View Model Input Features"):
        st.write(list(model.feature_names_in_))

    # Create booking using saved default values
    booking_data = default_values.copy()

    # Replace defaults with user inputs
    booking_data["hotel"] = hotel
    booking_data["lead_time"] = lead_time
    booking_data["deposit_type"] = deposit_type
    booking_data["total_of_special_requests"] = special_requests
    booking_data["adr"] = adr

    # Convert booking into a DataFrame
    input_df = pd.DataFrame([booking_data])

    # Arrange columns in the order expected by the model
    input_df = input_df[list(model.feature_names_in_)]

    # Display booking details
    with st.expander("View Booking Data for Prediction"):
        st.dataframe(input_df)

    # Predict cancellation when the button is clicked
    if st.button("🔍 Predict Cancellation"):

        # Predict cancellation class
        prediction = model.predict(input_df)[0]

        # Calculate cancellation probability
        probability = model.predict_proba(input_df)[0][1]

        # Display prediction
        if prediction == 1:
            st.error("⚠️ This booking is likely to be cancelled.")
        else:
            st.success("✅ This booking is unlikely to be cancelled.")

        # Display cancellation probability
        st.metric(
            "Estimated Cancellation Probability",
            f"{probability * 100:.2f}%"
        )

        # Display probability bar
        st.progress(float(probability))

        # Explain prediction uncertainty
        if 0.40 <= probability <= 0.60:
            st.info(
                "ℹ️ This prediction is uncertain because "
                "the probability is close to 50%."
            )
        else:
            st.info(
                "ℹ️ The prediction is further from "
                "the 50% decision threshold."
            )


# ---------------- MODEL PERFORMANCE PAGE ----------------

elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Evaluation results of the tuned Random Forest model "
        "on the held-out test dataset."
    )

    # Display evaluation metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", "85.22%")
        st.metric("Precision", "85.56%")

    with col2:
        st.metric("Recall", "72.82%")
        st.metric("F1 Score", "78.68%")

    st.caption(
        "These metrics were calculated using unseen test data."
    )


# ---------------- FEATURE INSIGHTS PAGE ----------------

elif page == "📈 Feature Insights":

    st.title("📈 Important Cancellation Factors")

    st.write(
        "Explore the features that contributed most "
        "to the Random Forest model's predictions."
    )

    # Feature importance values from Day 8
    factors = {
        "Deposit Type": 21.54,
        "Country": 13.42,
        "Lead Time": 10.03,
        "Market Segment": 8.33,
        "Special Requests": 6.64
    }

    # Convert feature importance into a DataFrame
    importance_df = pd.DataFrame(
        list(factors.items()),
        columns=["Feature", "Importance"]
    )

    # Display horizontal feature importance chart
    st.bar_chart(
        importance_df.set_index("Feature"),
        horizontal=True
    )

    st.caption(
        "Top five features based on Random Forest feature importance. "
        "These values show model importance, not causal effects."
    )
