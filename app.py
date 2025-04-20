# Set the page configuration as the first Streamlit command
import streamlit as st
st.set_page_config(page_title="Disease Prediction System", layout="wide")

import gdown
import os
import joblib
import pandas as pd
import pandas as pd
import joblib
from PIL import Image
import base64
import time
import os
import sys
import logging
from datetime import datetime

sys.path.append('.')  # Ensure modules can be imported
from buffer_system import PredictionBuffer
from online_learning import OnlineLearning

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger('app')


try:
    model, columns, label_encoder = joblib.load("disease_model.pkl")
    description = pd.read_csv("description.csv")
    precautions = pd.read_csv("precautions_df.csv")
    medications = pd.read_csv("medications.csv")
    diets = pd.read_csv("diets.csv")
    diets.columns = diets.columns.str.strip().str.lower()
    
    # Initialize buffer and online learning system
    buffer = PredictionBuffer(buffer_size=50)  # Retrain after 50 new examples
    online_learner = OnlineLearning()
    
except Exception as e:
    st.error(f"Error loading model or data files: {e}")
    logger.error(f"Error loading model or data files: {e}")
    st.stop()

# Custom CSS styling
st.markdown("""
<style>
    /* Base styles */
    html, body, .stApp { 
        background-color: #f4f6fa; 
        font-family: 'Segoe UI', sans-serif; 
        color: #222; 
    }

    .title { 
        font-size: 36px; 
        font-weight: 700; 
        color: #0077b6; 
        text-align: center; 
        margin-bottom: 30px; 
    }

    .section { 
        background-color: #ffffff;
        border-radius: 12px; 
        padding: 25px; 
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); 
    }

    .green-header { 
        color: #1f9d55; 
        font-size: 20px; 
        font-weight: 600; 
        border-bottom: 2px solid #e6e6e6;
        margin-bottom: 15px; 
        padding-bottom: 8px; 
    }

    .result-header { 
        color: #0077b6; 
        font-size: 26px; 
        font-weight: 700; 
        text-align: center; 
        margin: 20px 0;
        text-transform: uppercase; 
    }

    .container { 
        display: flex; 
        flex-wrap: wrap; 
        justify-content: space-between; 
        gap: 20px; 
        margin-bottom: 20px; 
        width: 100%; 
    }

    .col { 
        flex: 1 1 45%; 
        min-width: 45%;
        background-color: #ffffff !important; 
        border-radius: 12px !important; 
        padding: 20px !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04) !important; 
        color: #222 !important; 
    }

    .col ul { 
        padding-left: 20px; 
        margin-top: 0; 
    }

    .col li { 
        color: #333 !important; 
        font-weight: 500; 
        margin-bottom: 8px; 
        line-height: 1.6; 
    }

    .stButton > button { 
        background-color: #2dbe60 !important; 
        color: white !important; 
        font-weight: 600 !important;
        border: none !important; 
        border-radius: 6px !important; 
        padding: 10px 20px !important;
        margin-top: 15px !important; 
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        transition: background-color 0.2s ease-in-out, transform 0.2s ease-in-out; 
        width: 100%; /* Make buttons full width */
    }

    .stButton > button:hover { 
        background-color: #259b4e !important; 
        transform: translateY(-1px); 
    }

    .warning { 
        background-color: #fff8db; 
        color: #856404; 
        padding: 12px 15px; 
        border-left: 5px solid #ffecb5;
        border-radius: 6px; 
        font-weight: 500; 
        margin-top: 10px; 
    }

    .dashboard-container {
        background-color: #fff;
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    .doctor-image {
        display: block;
        margin: 0 auto 20px auto;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        max-width: 100%;
    }
    
    .welcome-header {
        color: #0077b6;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 15px;
    }
    
    .welcome-text {
        color: #333;
        text-align: center;
        margin-bottom: 20px;
        line-height: 1.6;
    }
    
    .stat-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        text-align: center;
    }
    
    .stat-label {
        color: #555;
        font-size: 14px;
        font-weight: 600;
    }
    
    .stat-value {
        color: #1f9d55;
        font-size: 24px;
        font-weight: 700;
    }
    
    .divider {
        height: 1px;
        background-color: #e6e6e6;
        margin: 25px 0;
    }
    
    .disclaimer {
        background-color: #f0f4f8;
        border-left: 4px solid #0077b6;
        padding: 10px 15px;
        font-size: 14px;
        color: #555;
        margin-top: 20px;
        border-radius: 4px;
    }

    /* For the form section */
    .white-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #eaeaea;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
    }
    
    .symptom-row {
        margin-bottom: 10px;
    }
    
    /* Button container for centered buttons in a row */
    .button-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
    }
    
    .button-container .stButton {
        flex: 0 0 auto;
        width: auto;
    }
    
    /* Fix for button width in the container */
    .button-container .stButton > button {
        width: auto;
    }
    .result-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
    margin-bottom: 20px;
    }

    .result-table th, .result-table td {
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid #e0e0e0;
    }

    .result-table th {
        background-color: #f5f7fa;
        color: #1f9d55;
        font-weight: 600;
    }

    .result-table tr:hover {
        background-color: #f9f9f9;
    }

    .result-table tr:last-child td {
        border-bottom: none;
    }

    /* Feedback confirmation styling */
    .feedback-confirmed {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: 500;
        color: #1f9d55;
        margin: 20px 0;
        border-left: 5px solid #1f9d55;
    }

    .feedback-negative {
        background-color: #fff8e1;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: 500;
        color: #ff8f00;
        margin: 20px 0;
        border-left: 5px solid #ff8f00;
    }

    /* Custom styling for the feedback section */
    .feedback-section {
        background-color: #f0f8ff;
        padding: 18px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e1ebf5;
    }

    .feedback-section h3 {
        color: #0077b6;
        font-size: 22px;
        margin-bottom: 8px;
    }

    .feedback-section p {
        color: #555;
        margin-bottom: 15px;
    }

    .feedback-buttons {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: 15px;
    }

    .feedback-buttons .stButton > button {
        min-width: 120px;
    }
    
    /* Error message styling */
    .error-text {
        color: #ff0000 !important;
        font-weight: 500;
    }

    @media (max-width: 768px) {
        .container { 
            flex-direction: column !important; 
        }
        .col { 
            width: 100% !important; 
            flex: 1 1 100%;
        }
        .title { 
            font-size: 28px; 
        }
        
        .button-container {
            flex-direction: column;
        }
            
        .feedback-buttons {
            flex-direction: column;
            gap: 15px;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Utility functions ---
def predict_disease(symptoms):
    input_df = pd.DataFrame([[0]*len(columns)], columns=columns)
    for symptom in symptoms:
        if symptom in input_df.columns:
            input_df.at[0, symptom] = 1
    prediction = model.predict(input_df)[0]
    return label_encoder.inverse_transform([prediction])[0]

def predict_top_n(symptoms, top_n=3):
    input_df = pd.DataFrame([[0]*len(columns)], columns=columns)
    for symptom in symptoms:
        if symptom in input_df.columns:
            input_df.at[0, symptom] = 1
    probs = model.predict_proba(input_df)[0]
    top_indices = probs.argsort()[::-1][:top_n]
    return [(label_encoder.inverse_transform([i])[0], probs[i]*100) for i in top_indices]


def fetch_description(disease):
    row = description[description['Disease'].str.lower() == disease.lower()]
    return row['Description'].values[0] if not row.empty else "No description available."

def fetch_precautions(disease):
    row = precautions[precautions['Disease'].str.lower() == disease.lower()]
    if not row.empty:
        return row.iloc[0][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].dropna().tolist()
    return ["No precautions found."]

def fetch_medications(disease):
    row = medications[medications['Disease'].str.lower() == disease.lower()]
    return eval(row['Medication'].values[0]) if not row.empty else ["No medications found."]

def fetch_diet(disease):
    row = diets[diets['disease'].str.lower() == disease.lower()]
    return eval(row['diet'].values[0]) if not row.empty else ["No diet found."]

# --- Dynamic Symptoms Input ---
if "symptoms" not in st.session_state:
    st.session_state.symptoms = [""] * 5  # Start with 5 input boxes
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None
    
if "current_symptoms" not in st.session_state:
    st.session_state.current_symptoms = []
    
# Always show admin panel
if "buffer_size" not in st.session_state:
    st.session_state.buffer_size = buffer.get_buffer_size()

# Add feedback state in session if not exists
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "feedback_message" not in st.session_state:
    st.session_state.feedback_message = ""


def add_symptom():
    st.session_state.symptoms.append("")

def submit_feedback(feedback):
    """Process user feedback about prediction accuracy"""
    if st.session_state.current_prediction and st.session_state.current_symptoms:
        # Create symptom dictionary from current symptoms
        symptom_dict = {symptom: 1 for symptom in st.session_state.current_symptoms}
        print(f"Symptom Dict: {symptom_dict}")
        # Add to buffer
        needs_retraining = buffer.add_entry(
            symptom_dict, 
            st.session_state.current_prediction,
            feedback_correct=feedback
        )
        
        # Update buffer size in session state
        st.session_state.buffer_size = buffer.get_buffer_size()
        
        # Check if we need to retrain
        if needs_retraining:
            with st.spinner("Retraining model with new data..."):
                success = online_learner.update_model()
                if success:
                    # Reload model and clear buffer
                    global model, columns, label_encoder
                    model, columns, label_encoder = joblib.load("disease_model.pkl")
                    buffer.clear_buffer()
                    st.session_state.buffer_size = 0
                    st.success("Model successfully updated with new data!")
                    # Log the retraining event
                    logger.info("Model retrained with buffer data")
                else:
                    st.markdown('<p class="error-text">Failed to update model.</p>', unsafe_allow_html=True)
                    
        return True
    return False

def add_new_symptom(symptom_name):
    """Add a new symptom to the dataset and model"""
    if not symptom_name or not isinstance(symptom_name, str):
        st.markdown('<p class="error-text">Please enter a valid symptom name</p>', unsafe_allow_html=True)
        return False
    
    with st.spinner(f"Adding new symptom '{symptom_name}' to the system..."):
        success = online_learner.add_new_symptom(symptom_name)
        if success:
            # Reload model 
            global model, columns, label_encoder
            model, columns, label_encoder = joblib.load("disease_model.pkl")
            st.success(f"New symptom '{symptom_name}' added successfully!")
            # Log the new symptom addition
            logger.info(f"New symptom '{symptom_name}' added to the system")
            return True
        else:
            st.markdown(f'<p class="error-text">Failed to add symptom \'{symptom_name}\'. It may already exist.</p>', unsafe_allow_html=True)
            return False

def make_new_prediction():
    # Reset feedback state for new prediction
    st.session_state.feedback_submitted = False
    st.session_state.feedback_message = ""

# --- Page Layout with Dashboard ---
# Create two columns: left for dashboard, right for the main content with more space between them
left_col, spacer, right_col = st.columns([1, 0.1, 2])  # Added spacer column for gap

# Left Column - Dashboard
with left_col:
    # Doctor image - Using a placeholder, you should replace with your actual image
    st.markdown("""
                
        <h2 class="welcome-header">Welcome to HealthAssist by Medisync</h2>
        <img src="https://th.bing.com/th/id/OIP.yrksxzN_8mATcMiOmn2QWAHaE8?rs=1&pid=ImgDetMain" alt="Doctor" class="doctor-image">
        <p class="welcome-text">
            Your personal health assistant powered by machine learning.
            Select your symptoms on the right panel to get an instant health analysis.
        </p>
        <div class="divider"></div>
    """, unsafe_allow_html=True)
    
    # Statistics cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-label">Diseases Database</div>
                <div class="stat-value">40+</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-label">Symptoms Tracked</div>
                <div class="stat-value">80+</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Add another stat row
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-label">Prediction Accuracy</div>
                <div class="stat-value">96.7%</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Learning Buffer</div>
                <div class="stat-value">{st.session_state.buffer_size}/50</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Add disclaimer
    st.markdown("""
        <div class="divider"></div>
        <div class="disclaimer">
            <strong>Important:</strong> This tool provides initial insights based on symptoms, but always consult a medical professional for a proper diagnosis and treatment plan.
        </div>
    """, unsafe_allow_html=True)
    
    # Admin panel section - always visible
    st.markdown("### Healthcare Professional Panel")
    
    # Add new symptom section
    st.subheader("Add New Symptom")
    new_symptom = st.text_input("Enter new symptom name (use underscores instead of spaces):")
    if st.button("Add Symptom to System"):
        if new_symptom:
            # Create a placeholder for the notification
            symptom_notification = st.empty()
            
            # Show a prominent notification
            symptom_notification.warning(f"⏳ Please wait! Adding new symptom '{new_symptom}' to the system...")
            
            success = add_new_symptom(new_symptom)
            if success:
                # Replace the notification with success message
                symptom_notification.success(f"✅ New symptom '{new_symptom}' added successfully!")
            else:
                # Replace the notification with error message
                symptom_notification.error(f"❌ Failed to add symptom '{new_symptom}'. It may already exist.")
        else:
            st.markdown('<p class="error-text">Please enter a symptom name</p>', unsafe_allow_html=True)
    
    # Force retrain option
    st.subheader("Model Management")
    if st.button("Force Retrain Model Now"):
        # Create a placeholder for the notification
        retrain_notification = st.empty()
        
        # Show a prominent notification
        retrain_notification.warning("⏳ Please wait! We are retraining our model. This may take a few moments...")
        
        success = online_learner.update_model()
        if success:
            buffer.clear_buffer()
            st.session_state.buffer_size = 0
            
            # Replace the notification with success message
            retrain_notification.success("✅ Model successfully retrained!")
        else:
            # Replace the notification with error message
            retrain_notification.error("❌ No new data available for retraining.")
    
    # View buffer status
    st.subheader("Learning Buffer Status")
    st.write(f"Current buffer size: {st.session_state.buffer_size}/50 examples")
    st.progress(st.session_state.buffer_size / 50)
    
    if st.button("View Buffer Contents"):
        if os.path.exists("buffer.csv"):
            buffer_df = pd.read_csv("buffer.csv")
            if len(buffer_df) > 0:
                st.dataframe(buffer_df)
            else:
                st.info("Buffer is currently empty.")
        else:
            st.markdown('<p class="error-text">Buffer file not found.</p>', unsafe_allow_html=True)

# Right Column - Main App Content
with right_col:
    st.markdown('<div class="title">🩺 Disease Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<h3>Select Your Symptoms</h3>', unsafe_allow_html=True)
    st.markdown('<p>Choose at least 5 symptoms from the dropdowns below for an accurate prediction.</p>', unsafe_allow_html=True)
    
    # Create dropdown inputs in groups of 3
    for i in range(0, len(st.session_state.symptoms), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(st.session_state.symptoms):
                with col:
                    # Add a placeholder option as the first option in the dropdown
                    options = ["Select a symptom"] + columns.tolist()
                    
                    # Set the index based on current selection or default to the placeholder (index 0)
                    current_index = 0  # Default to placeholder
                    if st.session_state.symptoms[idx] in columns:
                        # If there's a valid selection, find its index (+1 because we added a placeholder)
                        current_index = options.index(st.session_state.symptoms[idx])
                    
                    st.session_state.symptoms[idx] = st.selectbox(
                        f"Symptom {idx+1}",
                        options=options,
                        index=current_index,
                        key=f"symptom_{idx}",
                        help="Select a symptom from the dropdown"
                    )
    
    # Create a container for the buttons to be centered and in one line
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    
    # Add columns for buttons
    button_col1, button_col2 = st.columns(2)
    
    # Add Symptom button
    with button_col1:
        add_button = st.button("➕ Add Another Symptom", on_click=add_symptom)
    
    # Submit button
    with button_col2:
        submit_button = st.button("Get Prediction", key="submit", on_click=make_new_prediction)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction Results
    if submit_button or st.session_state.current_prediction:
        if submit_button:  # Only recalculate when the button is clicked
            symptoms_cleaned = [s for s in st.session_state.symptoms if s and s != "Select a symptom"]
            if len(symptoms_cleaned) != len(set(symptoms_cleaned)):
                st.markdown('<div class="warning"><span class="error-text">Please don\'t select the same symptom multiple times. Each symptom should be unique.</span></div>', unsafe_allow_html=True)
            elif len(symptoms_cleaned) < 4:
                st.markdown('<div class="warning"><span class="error-text">Please select at least 5 symptoms for an accurate prediction.</span></div>', unsafe_allow_html=True)
            else:
                # Make prediction
                st.session_state.current_symptoms = symptoms_cleaned
                disease = predict_disease(symptoms_cleaned)
                diseases = predict_top_n(symptoms_cleaned)
                # Save prediction results to session state
                st.session_state.current_prediction = disease
                st.session_state.prediction_results = diseases
                # Initialize selected disease to the top prediction
                if "selected_disease" not in st.session_state:
                    st.session_state.selected_disease = disease
        
        # Display the results if we have a current prediction
        if st.session_state.current_prediction:
            # If we have stored prediction results, use them
            if hasattr(st.session_state, 'prediction_results'):
                diseases = st.session_state.prediction_results
            else:
                # Fallback if somehow we lost the results
                diseases = predict_top_n(st.session_state.current_symptoms)
            
            st.markdown(f"""
            <div class="section">
                <div class="result-header">Prediction Results</div>
                <h2 style="text-align: center; color: #333; margin-top: 5px;">Top Predictions</h2>
                <table class="result-table">
                    <thead>
                        <tr>
                            <th>Disease</th>
                            <th>Probability</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>{diseases[0][0]}</td>
                            <td>{diseases[0][1]:.2f}%</td>
                        </tr>
                        <tr>
                            <td>{diseases[1][0]}</td>
                            <td>{diseases[1][1]:.2f}%</td>
                        </tr>
                        <tr>
                            <td>{diseases[2][0]}</td>
                            <td>{diseases[2][1]:.2f}%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            # Create buttons for each disease
            st.markdown("<h3 style='text-align: center; margin-top: 20px;'>Select a disease to view details:</h3>", unsafe_allow_html=True)
            
            # Create a row of buttons for disease selection
            disease_cols = st.columns(3)
            
            # Add a button for each disease
            for i, (disease_name, prob) in enumerate(diseases):
                with disease_cols[i]:
                    if st.button(f"{disease_name}", key=f"disease_btn_{i}"):
                        st.session_state.selected_disease = disease_name
            
            # Display details for the selected disease
            if "selected_disease" in st.session_state:
                selected_disease = st.session_state.selected_disease
                
                st.markdown(f"""
                <div style="
                    background-color: #f0f8ff;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 20px 0;
                    text-align: center;
                ">
                    <h2 style="color: #0077b6;">Details for {selected_disease}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Display disease information in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    # Description
                    st.markdown(f"""
                        <div style="
                            background-color: white;
                            padding: 20px;
                            border-radius: 10px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            height: 100%;
                        ">
                            <h3 style="color: #1f9d55;">📝 Description</h3>
                            <p style="color: #333;">{fetch_description(selected_disease)}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Medications
                    st.markdown(f"""
                        <div style="
                            background-color: white;
                            padding: 20px;
                            border-radius: 10px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            height: 100%;
                        ">
                            <h3 style="color: #1f9d55;">💊 Medications</h3>
                            <ul style="color: #333; padding-left: 20px;">
                                {"".join([f"<li>{med}</li>" for med in fetch_medications(selected_disease)])}
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Second row of information
                col3, col4 = st.columns(2)
                
                with col3:
                    # Precautions
                    st.markdown(f"""
                        <div style="
                            background-color: white;
                            padding: 20px;
                            border-radius: 10px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            margin-top: 20px;
                        ">
                            <h3 style="color: #1f9d55;">🛡️ Precautions</h3>
                            <ul style="color: #333; padding-left: 20px;">
                                {"".join([f"<li>{p}</li>" for p in fetch_precautions(selected_disease)])}
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    # Diet
                    st.markdown(f"""
                        <div style="
                            background-color: white;
                            padding: 20px;
                            border-radius: 10px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            margin-top: 20px;
                        ">
                            <h3 style="color: #1f9d55;">🥗 Recommended Diet</h3>
                            <ul style="color: #333; padding-left: 20px;">
                                {"".join([f"<li>{d}</li>" for d in fetch_diet(selected_disease)])}
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Feedback section
            st.markdown("""
            <div style="
                background-color: #f0f8ff;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                text-align: center;
            ">
                <h3 style="color: #0077b6;">Was this prediction helpful?</h3>
                <p>Your feedback helps improve our system</p>
            </div>
            """, unsafe_allow_html=True)
            
                        # If feedback was already submitted, show the message
            if st.session_state.feedback_submitted:
                if "correct" in st.session_state.feedback_message:
                    st.success(st.session_state.feedback_message)
                else:
                    st.warning(st.session_state.feedback_message)
            else:
                # Otherwise show the feedback buttons
                feedback_col1, feedback_col2 = st.columns(2)
                
                with feedback_col1:
                    if st.button("👍 Yes, The prediction is Correct"):
                        if submit_feedback(True):
                            st.session_state.feedback_submitted = True
                            st.session_state.feedback_message = "Thank you for your feedback! It helps our system learn."
                            st.rerun()
                
                with feedback_col2:
                    if st.button("👎 No, The prediction is Incorrect"):
                        if submit_feedback(False):
                            st.session_state.feedback_submitted = True
                            st.session_state.feedback_message = "Thank you for your feedback. We'll work to improve our predictions."
                            st.rerun()
            
            # Medical disclaimer
            st.markdown("""
                <div style="
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 20px;
                    text-align: center;
                ">
                    <p style="color: #666; font-size: 14px; margin: 0;">
                        <strong>Medical Disclaimer:</strong> This prediction is based on machine learning and should not replace professional medical advice.
                        Please consult with a healthcare professional for proper diagnosis and treatment.
                    </p>
                </div>
            """, unsafe_allow_html=True)

