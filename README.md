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
  The source code for the web-based interface and sample inference scripts is publicly available on GitHub:  
  👉 [https://github.com/leekj3133/MIBR_study](https://github.com/leekj3133/MIBR_study)

- **Model Weights & Training Code**  
  The full model training code, model weights, and backend modules are **not publicly available** due to institutional and legal restrictions on clinical data from South Korea. These materials may be shared upon reasonable request to the corresponding author and subject to institutional review board (IRB) approval, if applicable.

- **Data Availability**  
  The clinical and imaging datasets used in this study include sensitive patient information and cannot be publicly released.  
  However, a **simulated example dataset** (not derived from real patient data) is provided in the GitHub repository to demonstrate the input format and support reproducibility of the web application.

This setup enables transparency while respecting data privacy regulations.
