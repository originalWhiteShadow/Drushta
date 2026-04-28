import numpy as np
import shap
import matplotlib.pyplot as plt


def _rate(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _binary_rates(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    fnr = _rate(fn, fn + tp)
    fpr = _rate(fp, fp + tn)
    return fnr, fpr


def calculate_metrics(y_true, y_pred, sensitive_column):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    sensitive_column = np.asarray(sensitive_column).astype(str)

    results = {}
    for group in np.unique(sensitive_column):
        mask = sensitive_column == group
        true_group = y_true[mask]
        pred_group = y_pred[mask]
        classes = np.unique(np.concatenate([true_group, pred_group]))

        if len(classes) <= 2 and set(classes).issubset({0, 1}):
            fnr, fpr = _binary_rates(true_group, pred_group)
        else:
            fnr_values = []
            fpr_values = []
            for cls in classes:
                binary_true = (true_group == cls).astype(int)
                binary_pred = (pred_group == cls).astype(int)
                class_fnr, class_fpr = _binary_rates(binary_true, binary_pred)
                fnr_values.append(class_fnr)
                fpr_values.append(class_fpr)

            fnr = float(np.mean(fnr_values)) if fnr_values else 0.0
            fpr = float(np.mean(fpr_values)) if fpr_values else 0.0

        results[str(group)] = {
            "FNR": fnr,
            "FPR": fpr,
            "count": int(mask.sum()),
        }

    return results


def generate_shap_plot(model, input_data):
    background = input_data.sample(min(100, len(input_data)), random_state=42)

    try:
        explainer = shap.Explainer(model, background)
        shap_values = explainer(input_data)
    except Exception:
        kernel_explainer = shap.KernelExplainer(model.predict, background)
        shap_values = kernel_explainer.shap_values(input_data, nsamples=100)

    plt.close("all")
    try:
        shap.summary_plot(shap_values, input_data, show=False)
    except Exception:
        try:
            shap.waterfall_plot(shap_values[0], show=False)
        except Exception:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "SHAP plot unavailable", ha="center", va="center")
            ax.axis("off")
            return fig

    fig = plt.gcf()
    fig.tight_layout()
    return fig
