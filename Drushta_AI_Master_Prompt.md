# 🚀 Drushta AI (दृष्ट): Universal AI Bias Auditor - Master MLOps Prompt

**Dear AI Coding Assistant:**
You are acting as an **Expert Full-Stack MLOps Engineer**. Your objective is to guide me step-by-step in building "Drushta AI," a Universal AI Bias Auditor using Python and Streamlit.

**CRITICAL INSTRUCTION FOR AI:** Do **NOT** generate all the code at once. This leads to context limits and errors. You must execute this plan **Step-by-Step**. After generating the code for a specific step, you must **STOP** and ask me to run the code, test it, and confirm there are no errors before moving to the next step.

---

## 📂 Target File Structure
```text
project_root/
│── requirements.txt           # Dependencies
│── generate_data.py           # Multi-domain synthetic biased dataset generator
│── data/                      # Folder for generated CSVs
│── backend_engines/           # Decoupled backend logic
│   ├── __init__.py
│   ├── data_engine.py         # Domain-agnostic CSV handling & preprocessing
│   ├── audit_engine.py        # Fairness metrics & SHAP explanations
│   └── refine_engine.py       # TFMOT Pruning, Quantization, Equalized Odds
└── app.py                     # The main Streamlit frontend (Stateful & Multi-domain)
```

---

## 🛠 Step 1: Environment & Multi-Domain Data (Wait for my Go)
**AI Task:**
1. Generate `requirements.txt` including: `streamlit`, `tensorflow`, `tensorflow-model-optimization`, `shap`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`.
2. Generate `generate_data.py`. Create a script that generates **THREE** distinct synthetic datasets (10,000 rows each) and saves them to a `data/` folder:
    * `hr_hiring_data.csv`: Features `[Years_Experience, Education, Age, Gender, Interview_Score]` -> Target `[Hired]`. Inject bias where `Age > 50` or `Gender=Female` significantly lowers the hiring chance despite good scores.
    * `finance_loan_data.csv`: Features `[Income, Credit_Score, Loan_Amount, Zip_Code, Marital_Status]` -> Target `[Loan_Approved]`. Inject bias against specific `Zip_Code`s and `Marital_Status=Single`.
    * `medical_triage_data.csv`: Features `[Symptoms_Severity, Blood_Pressure, BMI, Income_Bracket, Race_Proxy]` -> Target `[Immediate_Care_Approved]`. Inject bias against low `Income_Bracket`.

*AI: Please provide the code for Step 1, then ask me to run `python generate_data.py` and confirm the 3 CSVs are created before proceeding.*

---

## 🛠 Step 2: Backend - Agnostic Data & Audit Engines (Wait for my Go)
**AI Task:**
1. Generate `backend_engines/data_engine.py`: Create `process_upload(file_buffer)` that flexibly handles ANY of the CSVs. It must automatically identify categorical vs. numerical columns for standard preprocessing (Label Encoding/MinMax Scaling).
2. Generate `backend_engines/audit_engine.py`: 
    * Create `calculate_metrics(y_true, y_pred, sensitive_column)` to return False Negative Rates (FNR) and False Positive Rates (FPR) dynamically grouped by the selected sensitive demographic.
    * Create `generate_shap_plot(model, input_data)` using `shap.Explainer` (or `KernelExplainer`) to return a matplotlib waterfall or summary plot. Ensure it does not hardcode column names.

*AI: Please provide the code for Step 2, then ask me to verify the logic looks sound before proceeding.*

---

## 🛠 Step 3: Backend - Refinement Engine (Wait for my Go)
**AI Task:**
1. Generate `backend_engines/refine_engine.py`. This is the core ML optimization logic.
    * Create `apply_equalized_odds(probabilities, sensitive_column)`: Algorithmically calculates group-specific thresholds to equalize the FNR across demographics.
    * Create `optimize_model(keras_model)`: Applies `tfmot.sparsity.keras.prune_low_magnitude` for 3 epochs, strips the pruning wrappers, and exports to INT8 via `tf.lite.TFLiteConverter`. Return the raw bytes of the optimized `.tflite` model.

*AI: Please provide the code for Step 3, then pause for confirmation.*

---

## 🛠 Step 4: The Stateful Streamlit UI (Final Step)
**AI Task:**
Generate `app.py`. Ensure it uses `st.set_page_config(layout="wide")`.
**Critical:** Initialize `st.session_state` variables (`active_domain`, `dataset`, `raw_model`, `mitigated_model`, `audit_results`) so data persists across tabs.

Build the UI with the following layout:
* **Sidebar (Domain Setup):** A dropdown to "Select Domain" (Finance, HR, Healthcare). Dynamically load the corresponding default CSV from the `data/` folder and allow the user to select the `sensitive_column` to audit.
* **Tab 1 (📥 Audit):** Display baseline fairness metrics and a bar chart of FNR disparity for the chosen domain.
* **Tab 2 (⚙️ Mitigate):** Checkboxes for 'Apply Equalized Odds' and 'Prune & Quantize (Edge Ready)'. A button to run `refine_engine`. Show a progress bar and save the output.
* **Tab 3 (🔍 Live Validation):** Dynamically generate an input form (sliders/dropdowns) based on the loaded dataset's specific columns. Show side-by-side SHAP waterfall plots comparing the 'Original Model Prediction' vs. 'Mitigated Model Prediction'.
* **Tab 4 (📦 Export):** Display a 'Deployment Readiness Score' and use `st.download_button` to allow downloading the edge-ready `optimized_model.tflite` from session state.

*AI: Please provide the code for Step 4, then ask me to run `streamlit run app.py` to test the final application.*
