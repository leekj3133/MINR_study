# NMIBC AI Tool (DEMO)


## Description:

This repository contains the frontend UI code and utility modules for a bladder cancer recurrence prediction model.  
It supports user input, preprocessing, and integration with a deep learning–based survival prediction model.

⚠️ Due to internal restrictions, the trained model weights and execution pipeline are not included in this repository.  
This version is intended for code structure review and documentation purposes only.

## Files:

- home.py            : Streamlit-based UI logic
- preprocess.py      : Image preprocessing functions
- util.py            : General utility functions
- requirements.txt   : Python package dependencies (original environment)
- LICENSE            : Project license
- README.md          : Markdown version with full documentation
- .gitignore         : Git exclusion configuration

## Author:

Developer: leekj3133
GitHub: https://github.com/leekj3133/MIBR_study

## Code and Data Availability

The recurrence prediction model developed in this study was implemented in **Python 3.10** using **TensorFlow 2.14** and the **Keras API**.

- **Web Demonstration**  
  A live demonstration of the recurrence prediction model is accessible at:  
  👉 https://bladder-cancer-recur.streamlit.app/
  
  This web application allows users to interactively test the model using example input data, without requiring local setup.  
  If you're interested in collaboration or would like to explore the model further, please contact the author.
  
- **Source Code (Web Interface & Sample Inference)**  
  The source code for the web-based interface and sample inference logic is publicly available on GitHub:  
  👉 [https://github.com/leekj3133/MIBR_study](https://github.com/leekj3133/MIBR_study)

- **Model Weights & Training Code**  
  The full training pipeline, model weights, and backend modules are not publicly released due to institutional and legal restrictions  
on clinical data originating from South Korea. These materials may be shared upon reasonable request and subject to approval  
by the corresponding institutional review board (IRB), if applicable.

- **Data Availability**  
  The clinical and imaging datasets used in this study contain sensitive patient information and cannot be publicly disclosed.  
However, a synthetic example dataset (not derived from real patient data) is included in the repository to demonstrate input format  
and support reproducibility of the web interface.

This setup ensures transparency while upholding data privacy and ethical research standards.

## Release & Citation
This demonstration is archived under: **DOI 10.5281/zenodo.16262080**

Please cite as:
JuYoung Lee & Se Young Choi (2025). Multimodal Deep Learning for Predicting Recurrence in NMIBC: Comparison with Traditional Risk Models. Zenodo. https://doi.org/10.5281/zenodo.16262080
