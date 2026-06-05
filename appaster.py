import streamlit as st
import pickle
import pandas as pd
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Asteroid Threat Monitor",
    layout="centered"
)

st.title("☄️ Asteroid Threat Monitor")
st.write("Evaluate asteroid data using our live prediction model and log metrics for future analysis.")

# 2. Load the Model safely
MODEL_PATH = "asteroid_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.error(f"🚨 **Model file missing:** Please make sure `{MODEL_PATH}` is in the same folder as this script.")
else:
    # 3. User Input Section
    
    col1, col2 = st.columns(2)
    with col1:
        abs_mag = st.number_input(
            "Absolute Magnitude", 
            value=20.0, 
            step=0.5, 
            help="Used directly by the prediction model."
        )
    with col2:
        diameter = st.number_input(
            "Diameter (km)", 
            value=0.5, 
            step=0.05, 
            format="%.4f",
            help="Used directly by the prediction model."
        )
        
    st.markdown("---")
    
    
    col3, col4 = st.columns(2)
    with col3:
        velocity = st.number_input(
            "Velocity (km/h)", 
            value=45000.0, 
            step=1000.0
        )
    with col4:
        closest_dist = st.number_input(
            "Closest Recorded Distance (km)", 
            value=5000000.0, 
            step=100000.0
        )

    # 4. Trigger Prediction Action
    if st.button("Predict Hazard Status", type="primary"):
        
        # We strictly pass ONLY the 2 required features into the model
        asteroid_df = pd.DataFrame({
            'Absolute Magnitude': [abs_mag],
            'Max Diameter (km)': [diameter]
        })
        
        try:
            # Predict probability using your model's logic
            probability = model.predict_proba(asteroid_df)[0][1]
            is_hazardous = bool(probability >= 0.30)
            
            st.markdown("---")
            st.subheader("📊 Live Prediction Analysis")
            
            # Displays classification status and metrics
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if is_hazardous:
                    st.error("⚠️ **Classification:** POTENTIALLY HAZARDOUS")
                else:
                    st.success("✅ **Classification:** SAFE / NOT HAZARDOUS")
                    
            with res_col2:
                st.metric(label="Hazard Probability", value=f"{probability:.2%}")
            
            st.progress(probability)
            
            # 5. Display All Provided Data Summary Table
            st.markdown("### 📋 Captured Metrics Summary")
            summary_data = {
                "Metric Parameter": ["Absolute Magnitude", "Diameter", "Velocity", "Closest Recorded Distance"],
                "Value": [f"{abs_mag}", f"{diameter} km", f"{velocity:,.2f} km/h", f"{closest_dist:,.2f} km"]
            }
            st.table(pd.DataFrame(summary_data))
            
            # --- UPDATED CAPTION ---
            st.caption(
                "Current model predictions are based on Absolute Magnitude and Maximum Diameter. "
                "Other parameters are displayed for future expansion."
            )
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
