import os
import cv2
import base64
import pickle
import joblib
import streamlit as st
from pathlib import Path

import preprocess
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import norm
import matplotlib.pyplot as plt

import tensorflow as tf 
from keras.layers import *
from keras.models import *

from sklearn.preprocessing import MinMaxScaler

def raise_placeholder_error(filename: str = ""):
    raise NotImplementedError(
        f"This function requires a private file ({filename}) that is not included in the public repository. "
        f"Please use the private deployment environment to run this."
    )
    
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

def survival(df,ori_df, file, model):
    """
    Run survival prediction for a single patient based on multimodal input.

    Args:
        df (pd.DataFrame): Input structured record data for a patient.
        csv_path (UploadedFile): Uploaded structured data file.
        file (UploadedFile): Uploaded cystoscopic image.
        model (tf.keras.Model): Trained multimodal survival prediction model.

    Returns:
        summary_lines (list): HTML strings summarizing the prediction results.
        fig (matplotlib.figure.Figure): Plot showing recurrence probability.
        back_image_bytes (bytes): Background removed image bytes.
        pre_image_bytes (bytes): Fully preprocessed image bytes.
    """
    orig_pil = Image.open(file).convert("RGB")
    orig = np.array(orig_pil)
    orig = cv2.resize(orig, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    
    image = Image.open(file).convert("RGB")
    image_array = np.asarray(image)
    image_back_removed =image_array
    image = image_array
    img = preprocess.zero_padding(image)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    img = preprocess.Min_Max_Normalization(img)
    img = preprocess.Histogram_Equalization_CLAHE_Color(img, limit=2,kernel_size=7)
    img_pre = img
    
    # encode to PNG buffers
    _, back_buf = cv2.imencode('.png', image_back_removed)
    _, pre_buf = cv2.imencode('.png', img_pre) 
    
    # Prepare prediction input      
    img_batch_item = img.astype(np.float32) / 255.0  
    img_batch_item = np.expand_dims(img_batch_item, axis=0)  
    
    # Load feature columns metadata (placeholder)
    try:
        # scaler_bundle = joblib.load("./model/scaler.pkl")
        # feature_columns = scaler_bundle["columns"]
        raise_placeholder_error("model/scaler.pkl")
    except NotImplementedError as e:
        st.warning(str(e))
        feature_columns = []  # downstream 에서 df_sub = df[feature_columns] 하면 빈 DataFrame
    
    # Build structured DataFrame for scaling (empty if no metadata)
    ori_df.index      = ori_df.index.astype(str).str.strip()
    ori_df.columns    = ori_df.columns.str.strip()
    df_sub = df.reindex(columns=feature_columns).astype(float) if feature_columns else pd.DataFrame()

    # Load min/max params (placeholder)
    try:
        # params = pickle.load(open("./datasets/minmax_params.pkl","rb"))
        raise_placeholder_error("datasets/minmax_params.pkl")
    except NotImplementedError as e:
        st.warning(str(e))
        data_min = np.zeros(df_sub.shape[1])
        data_max = np.ones(df_sub.shape[1])
    
    
    # Scale structured inputs
    if not df_sub.empty:
        raw_vals = df_sub.values
        denom = data_max - data_min
        denom[denom == 0] = 1
        scaled = (raw_vals - data_min) / denom
        X_test = pd.DataFrame(scaled, columns=feature_columns, index=df_sub.index)
    else:
        X_test = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
        
    # Model inference (placeholder)
    try:
        # pred = model.predict([X_test, img_batch_item])
        raise_placeholder_error("model weights")
        pred = np.zeros((1,2), dtype=float)
    except NotImplementedError as e:
        st.warning(str(e))
        return None, None, None, None
    
     # Unpack Weibull parameters and compute survival curve
    EPS = 1e-6
    z_alpha_ = pred[:, 0]
    z_beta_ = pred[:, 1]
    alpha_ = tf.nn.softplus(z_alpha_) + EPS  # shape parameter
    beta_ = tf.nn.softplus(z_beta_) + EPS    # scale parameter

    # --- Load common time grid for survival curve (placeholder) ---
    try:
        # common_time_grid = np.load("./model/common_time_grid.npy")
        raise_placeholder_error("model/common_time_grid.npy")
    except NotImplementedError as e:
        st.warning(str(e))
        common_time_grid = np.arange(0, 60, 12, dtype=float)
    patient_surv = np.exp(- (common_time_grid / beta_)**alpha_)  # shape (M,)

    
    # Load clustering metadata (placeholder)
    try:
        # with open("./model/kmeans_model.pkl","rb") as f: kmeans = pickle.load(f)
        # with open("./datasets/risk_group_order.pkl","rb") as f: sorted_idx = pickle.load(f)
        raise_placeholder_error("model/kmeans_model.pkl and datasets/risk_group_order.pkl")
    except NotImplementedError as e:
        st.warning(str(e))
        kmeans = None
        sorted_idx = np.arange(3)
  
    
    if kmeans is not None:
        patient_cluster = kmeans.predict([patient_surv])[0]
        patient_risk_group = np.where(sorted_idx[::-1] == patient_cluster)[0][0]
        risk_group_name = ["low","Intermediate","High"][patient_risk_group]
    else:
        risk_group_name = "unknown"
        
    # Compute recurrence window
    recurrence_prob = 1.0 - patient_surv[-1]
    recurrence_percent = int(round(recurrence_prob * 100))
    
    # Compute recurrence window thresholds (placeholder)
    range_min, range_max = None, None
    try:
        # with open("./model/risk_thresholds.pkl", "rb") as f:
        #     thr = pickle.load(f)
        raise_placeholder_error("model/risk_thresholds.pkl")
        high_risk_threshold = thr["high"]
        mid_risk_threshold  = thr["mid"]
        low_risk_threshold  = thr["low"]
    except NotImplementedError as e:
        st.warning(str(e))
        # 기본 임계값 설정 (예시)
        high_risk_threshold = 0.7
        mid_risk_threshold  = 0.5
        low_risk_threshold  = 0.3

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
        # Fallback estimation based on recurrence percent
        if recurrence_percent >= 70:
            range_min, range_max = 3, 9
        elif recurrence_percent >= 50:
            range_min, range_max = 6, 18
        elif recurrence_percent >= 30:
            range_min, range_max = 9, 24
        else:
            range_min, range_max = 12, 36
            
    mu = (range_min + range_max) / 2  
    sigma = (range_max - range_min) / 4  
    pdf = norm.pdf(common_time_grid, loc=mu, scale=sigma)
    pdf = pdf / pdf.max()  
    

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute recurrence probability curve
    recurrence_prob_curve = 1 - patient_surv  # fraction 0~1
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        common_time_grid,
        recurrence_prob_curve,
        label="Recurrence Probability (1 - S(t))",
        linestyle='-',
        marker='o',
        markersize=4,
        alpha=0.8
    )


    mask = (common_time_grid >= range_min) & (common_time_grid <= range_max)
    ax.scatter(
        common_time_grid[mask],
        recurrence_prob_curve[mask],
        color='blue',
        s=60,
        alpha=0.9,
        label=f"Focus Window ({int(range_min)}–{int(range_max)} mo)"
    )


    idx_focus_end = np.where(mask)[0][-1] 
    focus_t    = common_time_grid[idx_focus_end]
    focus_frac = recurrence_prob_curve[idx_focus_end]


    ax.scatter(
        [focus_t],
        [focus_frac],
        color='magenta',
        s=120,
        label=f"Focus-End Recurrence ≈ {focus_frac*100:.1f}%"
    )
    ax.annotate(
        f"{focus_frac*100:.1f}%",
        xy=(focus_t, focus_frac),
        xytext=(focus_t, focus_frac + 0.05),
        ha='center',
        arrowprops=dict(arrowstyle="->", color="magenta", lw=1.5),
        color="magenta",
        fontsize=12,
        fontweight="bold"
    )


    ax.plot(common_time_grid, pdf, linestyle='--', alpha=0.8,
            label="Estimated Recurrence Distribution")
    ax.fill_between(common_time_grid, 0, pdf,
                    where=mask,
                    color='gray', alpha=0.3,
                    label="Estimated Recurrence Window")

    ax.set_xlim(0, min(range_max + 5, common_time_grid[-1]))
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Months after TUR-BT")
    ax.set_ylabel("Probability")
    ax.set_title("Recurrence Probability Distribution")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()


    focus_percent = f"{focus_frac*100:.1f}%"
    # Format summary output for UI
    summary_lines = [
        f"<div style='border: 1px solid #ddd; padding: 10px; border-radius: 6px; background-color: #f9f9f9;'>",
        f"<h4>📊 Predicted Recurrence Probability: <b>{recurrence_percent}%</b></h4>",
        f"<h4>⏳ Estimated Recurrence Window: <b>{int(round(range_min))}–{int(round(range_max))} months </b></h4>",
        f"<h4>🔎 Focus-End Recurrence : <b>{focus_percent}</b></h4>",
        f"<h4>🧬 Predicted Risk Group: <b>{risk_group_name}</b></h4>",
        f"</div>"
    ]
    return summary_lines, fig,  back_buf.tobytes(), pre_buf.tobytes()



    
 

