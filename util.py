import os
import base64
import streamlit as st
import cv2
import preprocess
import numpy as np
import pandas as pd
from PIL import Image
from keras.layers import *
from keras.models import *
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import joblib
import matplotlib.pyplot as plt
from scipy.stats import norm
def raise_placeholder_error(filename: str = ""):
    raise NotImplementedError(
        f"This function requires a private file ({filename}) that is not included in the public repository. "
        f"Please use the private deployment environment to run this."
    )
# Disable GPU usage (run on CPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def set_background(image_file):
    """
    Set background image for Streamlit app.
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def survival(df, file, model):
    """
    Run survival prediction for a single patient based on multimodal input.

    Args:
        df (pd.DataFrame): Input structured record data for a patient.
        file (UploadedFile): Uploaded cystoscopic image.
        model (tf.keras.Model): Trained multimodal survival prediction model.

    Returns:
        summary_lines (list): HTML strings summarizing the prediction results.
        fig (matplotlib.figure.Figure): Plot showing recurrence probability.
        back_remove_image (str): Path to background-removed image file.
        preprocess_image (str): Path to fully preprocessed image file.
    """
    
    # Step 1: Load image and save original version
    image = Image.open(file).convert("RGB")
    image_array = np.asarray(image)

    # Background removal
    image = preprocess.background_remove(file, image_array)
    image_back_removed = image
    
    # Step 2: Apply preprocessing pipeline
    img = preprocess.zero_padding(image)
    img = cv2.resize(img, (512, 512))
    img = preprocess.Min_Max_Normalization(img)
    img = cv2.medianBlur(img,5)
    img = preprocess.Histogram_Equalization_CLAHE_Color(img, limit=2,kernel_size=7)

    # Convert to RGB and uint8 before encoding
    img_back_removed_rgb = cv2.cvtColor(image_back_removed, cv2.COLOR_RGB2BGR)
    img_preprocessed_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Encode images to PNG in memory
    _, back_buf = cv2.imencode('.png', img_back_removed_rgb)
    _, pre_buf = cv2.imencode('.png', img_preprocessed_rgb)
    
    try:
        # Step 3: Load pre-fitted scaler and expected input column structure
        # NOTE: This file was generated during training and only includes scaler & column metadata (no patient data)
        
        # scaler_bundle = joblib.load("./model/scaler_EORTC.pkl")
        # scaler = scaler_bundle["scaler"]
        # X_train_columns = scaler_bundle["columns"]
        raise_placeholder_error("model/scaler_EORTC.pkl")
        scaler = None
        X_train_columns = []  # placeholder로 남김
    # Match column order and scale the input record
    except NotImplementedError as e:
        st.warning(str(e))
        return None, None, None, None
    
    df = df[X_train_columns]  
    X_test = scaler.transform(df)
    X_test = pd.DataFrame(X_test, columns=X_train_columns)

    # Step 4: Prepare inputs for model inference
    img_batch = np.expand_dims(img, axis=0)
    
    # Step 5: Predict Weibull parameters (α, β) from model output
    try:
        # pred = model.predict([X_test,img_batch])
        raise_placeholder_error("model weights")
        pred = np.zeros((1, 2))  # dummy
    except NotImplementedError as e:
        print(e)
        return None, None, None, None
    EPS = 1e-6
    z_alpha_ = pred[:, 0]
    z_beta_ = pred[:, 1]
    alpha_ = tf.nn.softplus(z_alpha_) + EPS  
    beta_ = tf.nn.softplus(z_beta_) + EPS   

    # Step 6: Compute survival probabilities across time points
    common_time_grid = np.linspace(0.3, 198.0, 32)
    patient_surv = np.exp(- (common_time_grid / beta_[0]) ** alpha_[0])


    # Step 7: Load pretrained clustering model and risk group order
    try:
        # with open("./model/kmeans_model.pkl", "rb") as f:
        #     kmeans = pickle.load(f)

        # with open("./datasets/risk_group_order.pkl", "rb") as f:
        #     sorted_idx = pickle.load(f) 
        raise_placeholder_error("model/kmeans_model.pkl and datasets/risk_group_order.pkl")
        kmeans = None
        sorted_idx = np.array([0, 1, 2])  # dummy structure
    except NotImplementedError as e:
        print(e)
        return None, None, None, None
    
    # Step 8: Assign patient to risk group based on predicted survival curve
    patient_cluster = kmeans.predict([patient_surv])[0]  # shape: (1,)
    patient_risk_group = np.where(sorted_idx[::-1] == patient_cluster)[0][0]
    risk_group_name = ["low", "Intermediate", "High"][patient_risk_group]


    # Step 9: Estimate recurrence probability and recurrence window
    recurrence_prob = 1.0 - patient_surv[-1]
    recurrence_percent = int(round(recurrence_prob * 100))
    
    
    range_min, range_max = None, None
    high_risk_threshold = 0.5603
    mid_risk_threshold = 0.1809
    low_risk_threshold = 0.0203   
    
    # Determine time window using threshold-based logic or fallback heuristic
    if (
    np.any(patient_surv <= high_risk_threshold)
    and np.any(patient_surv <= mid_risk_threshold)
    and np.any(patient_surv <= low_risk_threshold)
):
        min_idx = np.argmax(patient_surv <= high_risk_threshold)
        max_idx = np.argmax(patient_surv <= low_risk_threshold)
        range_min = common_time_grid[min_idx]
        range_max = common_time_grid[max_idx]
    else:
        # Heuristic fallback by recurrence probability range
        if recurrence_percent >= 70:
            range_min, range_max = 3, 9
        elif recurrence_percent >= 50:
            range_min, range_max = 6, 18
        elif recurrence_percent >= 30:
            range_min, range_max = 9, 24
        else:
            range_min, range_max = 12, 36
            
    # Step 10: Estimate recurrence probability density function (PDF) over time
    mu = (range_min + range_max) / 2  
    sigma = (range_max - range_min) / 4  
    pdf = norm.pdf(common_time_grid, loc=mu, scale=sigma)
    pdf = pdf / pdf.max()  
    
    recurrence_prob_curve = 1 - patient_surv

    # Step 11: Plot recurrence probability and distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    mask = (common_time_grid > range_min + 1) & (common_time_grid <= range_max)
    ax.scatter(common_time_grid[mask], recurrence_prob_curve[mask],
            label="Recurrence Probability (1 - S(t))", color='blue', s=25, alpha=0.9)
    ax.plot(common_time_grid, pdf, label="Estimated Recurrence Distribution", color='green', linestyle='--')
    ax.fill_between(common_time_grid, 0, pdf,
                where=((common_time_grid >= range_min) & (common_time_grid <= range_max)),
                color='gray', alpha=0.3, label='Estimated Recurrence Window')

    ax.set_xlim(left=0, right=min(range_max + 25, common_time_grid[-1]))
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Months after TUR-BT")
    ax.set_ylabel("Probability")
    ax.set_title("Recurrence Probability Distribution")
    ax.legend()
    fig.tight_layout()
    
    # Step 12: Format summary output for UI
    summary_lines = [
        f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 6px; background-color: #f9f9f9;'>",
        f"<h4>📊 Predicted Recurrence Probability: <b>{recurrence_percent}%</b></h4>",
        f"<h4>⏳ Estimated Recurrence Window: <b>{int(round(range_min))}–{int(round(range_max))} months</b></h4>",
        f"<h4>🧬 Predicted Risk Group: <b>{risk_group_name}</b></h4>",
        f"</div>"
    ]
    return summary_lines, fig,  back_buf.tobytes(), pre_buf.tobytes()


    
 

