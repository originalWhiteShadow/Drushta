# Drushta AI

Drushta AI is a Streamlit-based bias auditor for tabular ML models. Upload a dataset (and optionally a trained model) to audit group fairness, apply mitigation, and export a lighter model for edge deployment.

## Highlights
- Dataset-first or model-first workflows
- Automatic target handling for binary or multiclass labels
- Bias scan to suggest a sensitive column
- Equalized Odds (binary + multiclass approximation)
- SHAP-based explainability visuals
- Model pruning and quantization to TFLite

## Quickstart
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Generate sample datasets:
   ```bash
   python generate_data.py
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## How to Use
### Dataset mode
1. Choose "Upload dataset mode".
2. Upload a CSV or TSV dataset.
3. Pick the target column (or binarize a numeric target).
4. Select a sensitive column (or use the bias scan suggestion).
5. Proceed to audit and run mitigation if desired.

### Model mode
1. Choose "Upload model mode".
2. Upload a CSV/TSV dataset.
3. Upload a Keras model (.keras or .h5).
4. Audit and mitigate the model, then export a lighter TFLite model.

## Data Requirements
- Tabular CSV/TSV with a single target column.
- Binary or multiclass targets are supported.
- Mixed categorical and numeric features are OK.

## Model Requirements (Upload Model Mode)
- Input feature count must match the processed dataset.
- Binary targets require a single-output sigmoid model.
- Multiclass targets require a softmax output with N classes.

## Mitigation & Export
- Equalized Odds uses group-specific thresholds.
- Multiclass Equalized Odds is approximated one-vs-rest.
- Pruning and quantization generate an optimized TFLite file.

## Project Structure
```
app.py
backend_engines/
  data_engine.py
  audit_engine.py
  refine_engine.py
data/
  (optional datasets)
generate_data.py
requirements.txt
```

## Notes
- Example datasets are optional. If data/ is empty, only uploads are available.
- Large datasets can make SHAP slower.

## License
See LICENSE.
