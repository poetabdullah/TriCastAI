import gradio as gr
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load models & scalers
xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model("xgb_model.json")

xgb_reg = joblib.load("xgb_pipeline_model.pkl")

scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

lstm_model = load_model("lstm_revenue_model.keras")

# Prediction + Plot functions
def classify_fn(df: pd.DataFrame):
    preds = xgb_clf.predict(df)
    probs = xgb_clf.predict_proba(df)
    fig, ax = plt.subplots()
    ax.bar(['No Bankruptcy', 'Bankruptcy'], probs[0], color=['#4CAF50', '#F44336'])
    ax.set_ylim(0, 1)
    ax.set_title('Bankruptcy Probability')
    ax.set_ylabel('Probability')
    plt.tight_layout()
    return {"Predicted Label": int(preds[0])}, fig


def regress_fn(df: pd.DataFrame):
    preds = xgb_reg.predict(df)
    fig, ax = plt.subplots()
    sns.histplot(preds, bins=20, kde=True, ax=ax)
    ax.set_title('Anomaly Score Distribution')
    ax.set_xlabel('Predicted Anomaly Score')
    plt.tight_layout()
    return preds.tolist(), fig


def lstm_fn(seq_str: str):
    vals = np.array(list(map(float, seq_str.split(',')))).reshape(1, -1)
    vals_s = scaler_X.transform(vals).reshape((1, vals.shape[1], 1))
    pred_s = lstm_model.predict(vals_s)
    pred = scaler_y.inverse_transform(pred_s)[0, 0]
    fig, ax = plt.subplots()
    ax.plot(range(10), vals.flatten(), marker='o', label='Input Revenue')
    ax.plot(10, pred, marker='X', markersize=10, color='red', label='Predicted Q10')
    ax.set_xlabel('Quarter Index (0-10)')
    ax.set_ylabel('Revenue')
    ax.set_title('Revenue Forecast')
    ax.legend()
    plt.tight_layout()
    return float(pred), fig

# Build UI
grid_css = """
body {background-color: #f7f7f7;}
.gradio-container {max-width: 800px; margin: auto; padding: 20px;}
h1, h2 {color: #333;}
"""

demo = gr.Blocks(css=grid_css)
with demo:
    gr.Markdown("# 🚀 FinSight 360™ Dashboard")
    gr.Markdown("Comprehensive financial AI:\\n- Bankruptcy Classification\\n- Anomaly Scoring\\n- Revenue Forecasting")

    with gr.Tab("🏦 Bankruptcy Classifier"):
        gr.Markdown("**Upload company features** (as DataFrame) to predict bankruptcy:")
        inp1 = gr.Dataframe(type="pandas", label="Features DataFrame")
        out1 = gr.Label(label="Predicted Label")
        plt1 = gr.Plot()
        inp1.submit(classify_fn, inp1, [out1, plt1])

    with gr.Tab("📈 Anomaly Regression"):
        gr.Markdown("**Upload company features** (as DataFrame) to predict anomaly score:")
        inp2 = gr.Dataframe(type="pandas", label="Features DataFrame")
        out2 = gr.Textbox(label="Predicted Scores List")
        plt2 = gr.Plot()
        inp2.submit(regress_fn, inp2, [out2, plt2])

    with gr.Tab("📊 LSTM Revenue Forecast"):
        gr.Markdown("**Enter last 10 quarterly revenues** (comma-separated) to forecast Q10 revenue:")
        inp3 = gr.Textbox(placeholder="e.g. 1000,1200,1100,...", label="Q0–Q9 Revenues")
        out3 = gr.Number(label="Predicted Q10 Revenue")
        plt3 = gr.Plot()
        inp3.submit(lstm_fn, inp3, [out3, plt3])

    gr.Markdown("---\\n*SDG 9: Industry, Innovation and Infrastructure*")

demo.launch()