import streamlit as st
import joblib
from PIL import Image, ImageFile
import pandas as pd
from util import *
import warnings
from tensorflow.keras.models import load_model
from preprocess import *
import io
from keras.utils import custom_object_scope
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

@tf.function
def weibull_nll(y_true, y_pred):
    """
    Weibull negative log-likelihood loss function for survival analysis.

    Args:
        y_true: Tensor of shape (batch_size, 2), where each row is [time, event]
                - time: observed survival time
                - event: 1 if event occurred (uncensored), 0 if censored
        y_pred: Tensor of shape (batch_size, 2), where each row is [z_alpha, z_beta]
                - z_alpha, z_beta are raw outputs from the model (before softplus)
    
    Returns:
        Negative log-likelihood loss (scalar)
    """
    EPS = 1e-6  # Small constant to prevent log(0) or division by zero
    
    times  = y_true[:, 0]  # observed survival times
    events = y_true[:, 1]  # event indicators (1 = event occurred, 0 = censored)

    z_alpha = y_pred[:, 0]
    z_beta  = y_pred[:, 1]
    # Convert raw model outputs to positive Weibull parameters using softplus
    alpha_ = tf.nn.softplus(z_alpha) + EPS  # shape parameter
    beta_  = tf.nn.softplus(z_beta)  + EPS  # scale parameter
    # Check for NaNs or Infs (for debugging)
    tf.debugging.check_numerics(alpha_, "alpha_ contains NaN or Inf")
    tf.debugging.check_numerics(beta_, "beta_ contains NaN or Inf")
    tf.debugging.check_numerics(times, "times contains NaN or Inf")
    tf.debugging.check_numerics(events, "events contains NaN or Inf")
    
    # Log-likelihood of event occurring at time t: log f(t)
    logf = ( tf.math.log(alpha_) 
            - alpha_ * tf.math.log(beta_)
            + (alpha_ - 1.0)*tf.math.log(times + EPS)
            - tf.pow((times+EPS)/beta_, alpha_) )
    
    # Log survival function (for censored data): log S(t)
    logS = - tf.pow((times+EPS)/beta_, alpha_)

    # Negative log-likelihood:
    #   - If event occurred: use logf
    #   - If censored: use logS
    #   Loss = -mean( event * logf + (1 - event) * logS )
    nll = - tf.reduce_mean( events*logf + (1.0 - events)*logS )
    
    return nll


st.set_page_config(layout="wide")


def initialize_session():
    defaults = {
        'page': 'input',
        'csv_data': None,
        'file_image': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session()


if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'file_image' not in st.session_state:
    st.session_state.file_image = None
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'input_saved' not in st.session_state:
    st.session_state.input_saved = False

# --------------------------- Page: Input ---------------------------
if st.session_state.page == 'input':
    
    if 'csv_data' not in st.session_state:
            st.session_state.csv_data = None
    
    if 'empty_df' not in st.session_state:
        st.session_state.empty_df = pd.read_csv("./datasets/bladder_example.csv", index_col=0)
    if "csv" not in st.session_state:
                st.session_state.csv = pd.DataFrame() 


    if "input_data" not in st.session_state:
        st.session_state.input_data = {}
    if "page_index" not in st.session_state:
        st.session_state.page_index = 0
    if "nav_clicked" not in st.session_state:
        st.session_state.nav_clicked = False
    if "target_page" not in st.session_state:
        st.session_state.target_page = 0
    st.title('🩺 Bladder Cancer Recur')

    st.write("")
    st.write("")

    with st.sidebar:
        st.write("### 📥 Download example csv file ")
        st.write("")
        with open("./datasets/bladder_example.csv") as f:
            st.download_button("Download example csv file",f,file_name = "bladder_example.csv")
        with open("./datasets/csv_guideline.xlsx","rb") as f:
            st.download_button(label = "Download csv guideline",
                        data = f,
                        file_name = "csv_guideline.xlsx",
                        mime="application/vnd.ms-excel")
        st.write("")
        st.write(":exclamation: To upload a csv file, you must upload it in a designated format.")
        st.write(":exclamation: If there is no CSV file in the designated format, please download the example file and input the values according to the guidelines.")
        st.write(":exclamation: Age, Sex, Op-duration and OptoBCG_dur columns are required ")
        st.write(":exclamation: If you do not follow the instructions, you might encounter some errors.")
    
    # upload file
    # No personal data is stored. All user inputs are kept in session memory only and discarded after use.
    image, record = st.columns(2,gap="large")

    with st.container(border=True):
        with image:
            st.write("### 📤 Upload the image file")
            file_image = st.file_uploader('',type=['jpeg','jpg','png'])
            if file_image is not None:
                try:
                    image = Image.open(file_image).convert("RGB")
                    image.verify()  
                    st.image(image, use_container_width =True)
                    st.session_state.file_image = file_image  
                except Exception:
                    st.error("❌ Invalid or corrupted image file. Please re-upload.")
            st.write("❌ Please avoid images with cystoscopy lines.")
            st.write("✂️ please crop the letters in your image")
            example_image_path = "./datasets/example_image.png"
            example_image = Image.open(example_image_path).convert("RGB")
            st.image(example_image, use_container_width =True)
            st.write("")
            st.write("")
            st.write("")
        
    with st.container(border=True):
        with record:
            st.write("### 📤 Upload the File or Input the values")
            st.write("")
            st.write("")
            csv = None
            vertical_alignment = st.selectbox("✅ Select Option", ["Upload File", "Enter Text"], index=1)
            if vertical_alignment == "Upload File":
                file_csv = st.file_uploader('',type=["csv", "xlsx", "xls"])
                st.write("👈 Sample of csv, excel file is in the sidebar")
                st.write("")
                # load classifier
                if file_csv:
                    try:
                        selected_data = file_csv
                        st.session_state.csv_data = selected_data
                        st.session_state.input_saved = True
                        st.success("✅ File uploaded successfully.")
                        st.write("")
                        st.write("")
                        st.write("")
                        csv = selected_data
                    except Exception:
                        st.error("❌ Invalid CSV file. Please check the format.")
                
            elif vertical_alignment == "Enter Text":
                st.warning("🛠 Text input form to be implemented here.")
                
            
                history_items = {
                    "Age": ("number", True),
                    "Sex": ("select", True),
                    "Height(cm)": ("number", False),
                    "Weight(kg)": ("number", False),
                    "Smoking": ("select", False),
                    "Hypertension": ("select", False),
                    "Diabetes": ("select", False),
                    "Dyslipidemia": ("select", False),
                    "Previous surgical history": ("select", False),
                    "past TUR-BT":("select", True),    
                }
                pre_op_lab_items = {                
                    "Pre-op TP (Total Potention)": ("number", False),
                    "Pre-op albumin": ("number", False),
                    "Pre-op BUN": ("number", False),
                    "Pre-op Cr (Creatinine)": ("number", False),
                    "Pre-op UA (Uric Acid)": ("number", False),
                    "Pre-op LDH (lactate dehydrogenas)": ("number", False),
                    "Pre-op GGT (Gamma glutamyl peptidase)": ("number", False),
                    "Pre-op ALP (alkaline phosphatase)": ("number", False),
                    "Pre-op WBC": ("number", False),
                    "Pre-op Hb (Hemoglobin)": ("number", False),
                    "Pre-op Platelet": ("number", False),
                    "Pre-op NEU (Neutrophil count %)": ("number", False),
                    "Pre-op LYM (lymphocyte %)": ("number", False),
                    "Pre-op MONO (monocyte %)": ("number", False),
                    "Pre-op EOS (eosinophil %)": ("number", False),
                    "Pre-op BASO (basophil %)": ("number", False),
                }
                UA_items = {
                    "UA_SG": ("select", False),
                    "UA_pH": ("select", False),
                    "UA_protein": ("select", False),
                    "UA_Glucose": ("select", False),
                    "UA_ketone": ("select", False),
                    "UA_urobilinogen": ("number", False),
                    "UA_Nitrite": ("select", False),
                    "UA_RBC": ("select", False),
                    "UA_WBC": ("select", False),
                    "UA_squamous": ("select", False),
                    "UA_Hyaline": ("select", False),
                    "UA_Bacteria": ("select", False) 
                }

                op_items = {  
                    "Biopsy": ("select", False), 
                    "Biopsy_malignancy": ("select", False),
                    "OP_duration" : ("select", True), 
                    "Size(cm)" : ("number", False),
                    "Multifocal": ("select", False),
                    "BCG number":("number", False),
                    "Additional BCG injections after the first 6 times" : ("select", False),
                    "OPtoBCG_dur(days)" : ("number", True),
                }
        
                post_op_lab_items = {                
                    "Post-op Glucose": ("number", False),
                    "Post-op BUN": ("number", False),
                    "Post-op Cr (Creatinine)": ("number", False),
                    "Post-op WBC": ("number", False),
                    "Post-op Hb (Hemoglobin)": ("number", False),
                    "Post-op Platelet": ("number", False),
                    "Post-op NEU (Neutrophil count %)": ("number", False),
                    "Post-op LYM (lymphocyte %)": ("number", False),
                    "Post-op MONO (monocyte %)": ("number", False),
                    "Post-op EOS (eosinophil %)": ("number", False),
                    "Post-op BASO (basophil %)": ("number", False),
                }
                Pathology_items = {
                    "urothelial carcinoma": ("select", False),
                    "grade": ("select", False),
                    "subepithelial connective tissue invasion (T1)": ("select", False),
                    "proper muscle (T2)": ("select", False),
                    "muscle invasion": ("select", False),
                    "histologic variant": ("number", False),
                    "lymphovasuclar invasion": ("select", False),
                    "CIS component": ("select", False),
                }
                all_items = {}
                for d in [history_items.items(), pre_op_lab_items.items(), UA_items.items(), 
                        op_items.items(), post_op_lab_items.items(),Pathology_items.items()
                        ]:
                    all_items.update(d)

                pages_data = [
                    list(history_items.keys()),        # Page 1: Patient history
                    list(pre_op_lab_items.keys()),     # Page 2: Pre-op labs
                    list(UA_items.keys()),             # Page 3: Urinalysis
                    list(op_items.keys()),             # Page 4: Operation details
                    list(post_op_lab_items.keys()),    # Page 5: Post-op labs
                    list(Pathology_items.keys())       # Page 6: Pathology
                ]
                custom_options = {
                    "Sex": ["", "Male", "Female"],
                    "Smoking": ["", "Non-smoker", "Former smoker", "Current smoker", "Uncertain"],
                    "Hypertension": ["", "Yes", "No", "Unknown"],
                    "Diabetes": ["", "Yes", "No", "Unknown"],
                    "Dyslipidemia":["", "Yes", "No", "Unknown"],
                    "Previous surgical history":["", "Yes","No"],
                    "past TUR-BT":["", "Yes","No"],
                    
                    
                    "UA_SG":["", ">=1.030", ">=1.00, <1.030"],
                    "UA_pH":["", "5.0", "5.5", "6.0","6.5","7.0","7.5","8.0","8.5",">=9.0"],
                    "UA_protein":["", "10(+/)", "10(+/-)", "10(1+)","100(2+)","1000(4+)","15(+/-)","30(+/-)","30(1+)","300(3+)"],
                    "UA_Glucose":["", "0.1 (+/-)", "0.25(1+)","0.5(2+)","1.0(3+)",">=1.0(3+)",">=1.3(3+)","2.0(4+)"],
                    "UA_ketone":["", "5(+/-)", "10(1+)", "15(1+)"],
                    "UA_Nitrite":["", "-", "+"],
                    "UA_RBC":["", "0-1/ <1", "1-4", "5-9","10-19","20-29","30-49","50-99","<1/2 of field of view  / >100"],
                    "UA_WBC":["", "0-1/ <1", "1-4", "5-9","10-19","20-29","30-49","50-99","<1/2 of field of view  / >100"],
                    "UA_squamous":["", "0-1/ <1", "1-4", "5-9","10-19","20-29","30-49","50-99","<1/2 of field of view  / >100"],
                    "UA_Hyaline":["", "<1", "1-2", "3-5","6-10","11-20",">20"],
                    "UA_Bacteria":["","Not Found", "A few", "Rare", "Many", "Moderate"],
                    
                    "Biopsy":["", "Implement", "Not implement"],
                    "Biopsy_malignancy":["", "Not finding", "Finding"],
                    "OP_duration":["", "< 30 minutes", "30 minutes ~ 1 hour", "> 1 hour"],
                    "Multifocal":["", "Yes","No"],
                    "Additional BCG injections after the first 6 times":["", "Yes","No"],
                    
                    "urothelial carcinoma":["", "Yes","No"],
                    "grade":["", "Low", "High"],
                    "subepithelial connective tissue invasion (T1)":["", "Absent", "Present"],
                    "proper muscle (T2)":["", "Not submitted", "Submitted"],
                    "muscle invasion":["", "Absent", "Present"],
                    "histologic variant":["", "Absent", "Present", "ocal glandular differentiation"],
                    "lymphovasuclar invasion":["", "Absent", "Present"],
                    "CIS component":["", "Absent", "Present"],
                        }
                
                
                
                st.text("")
                total_pages = len(pages_data)

                if "page_index" not in st.session_state or st.session_state.page_index >= total_pages:
                    st.session_state.page_index = 0
                page_labels = ["History*", "Pre-OP Lab", "UA", "OP*", "Post-OP Lab", "Pathology"]
                selected = st.radio(
                                "페이지 선택", 
                                page_labels,
                                index=st.session_state.page_index, 
                                horizontal=True, 
                                label_visibility="collapsed"
                                )
                st.markdown(
                    "<div style='text-align:right; color: gray; font-size: 0.9em;'>* :&nbsp; required</div>",
                    unsafe_allow_html=True
                )
                st.session_state.page_index = page_labels.index(selected)
                st.text("")
                current_items = pages_data[st.session_state.page_index]
                for item in current_items:
                    input_type, required = all_items[item]
                    label = f"{item} {'*' if required else '(optional)'}"
                    prev_value = st.session_state.input_data.get(item, "")
                    
                    if item in custom_options:
                        options = custom_options[item]
                        index = options.index(prev_value) if prev_value in options else 0
                        value = st.selectbox(label, options=options, index=index, key=item)
                        
                    elif input_type == "number":
                        value = st.number_input(label, min_value=0.0, max_value=300.0, value=prev_value if isinstance(prev_value, (int, float)) else 0.0, step=0.01, key=item)
                    else:
                        value = st.text_input(label, value=prev_value if prev_value else "", key=item)

                    st.session_state.input_data[item] = value


                save_clicked = st.button("✅ Save inputs")
                if save_clicked:
                    missing_required = []
                    for item, (typ, required) in all_items.items():
                        val = st.session_state.input_data.get(item, None)
                        if required:
                            if typ == "number":
                                if val is None or val == 0:
                                    missing_required.append(item)
                            else:
                                if val is None:
                                    missing_required.append(item)

                    if missing_required:
                        st.error(f"❌ Please fill in required fields: {', '.join(missing_required)}")
                    else:
                        st.success("✅ All required fields are filled!")
                        data = st.session_state.input_data
                        # The mapping of categorical values and column standardization
                        # is handled using a predefined dictionary (not shown here for confidentiality).

                        replace_map = {
                            "": None,
                            "Category_A": 0, "Category_B": 1, "Category_C": 2,
                            "Option_X": 1, "Option_Y": 0,
                            "Level_1": 0, "Level_2": 1, "Level_3": 2,
                            "Positive": 1, "Negative": 0,
                            "Short": 0, "Medium": 1, "Long": 2,
                            "Low": 0, "High": 1,
                            "Present": 1, "Absent": 0,
                            "Unknown": 2
                        }
                        Unknwon_replace = {
                            "Unknown" : 2,
                        }
                        value_map = {
                            0.0: np.nan
                        }
                        columns = {
                            "Age (yrs)": "age",
                            "Sex": "sex",
                            "Height(cm)": "height",
                            "Weight(kg)": "weight",
                            "Lab_A": "lab_a",
                            "Lab_B": "lab_b",
                            "Score_1": "score_1",
                            "Test_Result_A": "test_a",
                            "Imaging_Feature": "img_feat",
                            "Surgery_Type": "surg_type",
                            "Follow-up Duration": "followup_dur",
                            "Other_Feature_X": "feat_x"
                        }
                        renamed_data = {columns.get(k, k): v for k, v in data.items()}   
                        mapped_data = {k: value_map.get(v, v) for k, v in renamed_data.items()}
                        new_data = {k: replace_map.get(v, v) for k, v in mapped_data.items()}
                        empty_df = pd.read_csv("./datasets/bladder_example.csv", index_col=0)
                        
                        for key, value in new_data.items():
                            empty_df[key] = value 
                        empty_df = empty_df[~empty_df.index.isna()]
                        empty_df["UA_Bilirubin"] = 0
                        csv_data = empty_df
                        st.session_state.csv_data = csv_data
                        st.session_state.input_saved = True
                        csv = csv_data

    st.write("")
    # Determine button status
    ready = (
            st.session_state.file_image is not None and st.session_state.csv_data is not None and
            st.session_state.input_saved
        )

    if ready:
        if st.button("Get Results", type="primary"):
            st.session_state.page = 'result'
            st.rerun()
    else:
        st.info("📌 Please upload both image and patient data to proceed.")


elif st.session_state.page == 'result':
    if 'survival_model' not in st.session_state:
        try:
            with custom_object_scope({"weibull_nll": weibull_nll}):
                st.session_state.survival_model = load_model("./model/model_best_fin_EORTC_multi_input_fin_v12_50.h5")
        except Exception as e:
            st.error("⚠️ Failed to load model. Please contact administrator.")
            st.stop()
    st.title("📈 Prediction Results")
    
    
    file_image = st.session_state.get("file_image") 
    csv_data = st.session_state.get("csv_data")

    if file_image is None:
        st.error("No image found. Please upload an image first.")
        st.stop()
    file_image.seek(0)
    
    survival_model = st.session_state.survival_model

    
    
    summary_lines, fig, back_remove_image, preprocess_image = survival(csv_data, file_image, survival_model)


    if 'back_image_bytes' not in st.session_state:
        with open(back_remove_image, "rb") as f:
            st.session_state.back_image_bytes = f.read()

    if 'pre_image_bytes' not in st.session_state:
        with open(preprocess_image, "rb") as f:
            st.session_state.pre_image_bytes = f.read()

    with st.sidebar:
        st.write("")
        st.subheader("📁 Download Processed Images")

        st.write("background remove image")
        st.download_button(
            label="Download",
            data=io.BytesIO(st.session_state.back_image_bytes),
            file_name=os.path.basename(back_remove_image),
            mime="image/png",
            key="download_back_img"
        )

        st.write("preprocessed image")
        st.download_button(
            label="Download",
            data=io.BytesIO(st.session_state.pre_image_bytes),
            file_name=os.path.basename(preprocess_image),
            mime="image/png",
            key="download_preprocessed_image"
        )
       
    summary_text = "\n\n".join(map(str, summary_lines))

    # write result
    st.write("")
    st.markdown(summary_text, unsafe_allow_html=True)
    col1, col2= st.columns([.6, .4])
    with col1:
        st.pyplot(fig)

    if st.button("🔙 Back to input"):
        st.session_state.page = 'input'
        st.session_state.file_image = None
        st.session_state.cls_data = None
        st.session_state.survival_data = None
        st.session_state.input_saved = False
        st.rerun()