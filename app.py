import gradio as gr
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Load models & scalers
xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model("xgb_model.json")
xgb_reg = joblib.load("xgb_pipeline_model.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")
lstm_model = load_model("lstm_revenue_model.keras")

# Set matplotlib style for dark theme compatibility
plt.style.use('dark_background')

def process_csv_file(file):
    """Process uploaded CSV file and return DataFrame"""
    if file is None:
        return None
    try:
        df = pd.read_csv(file.name)
        return df
    except Exception as e:
        gr.Warning(f"Error reading CSV file: {str(e)}")
        return None

def classify_fn(file):
    """Bankruptcy classification from CSV file"""
    if file is None:
        return "Please upload a CSV file", None
    
    df = process_csv_file(file)
    if df is None:
        return "Error processing file", None
    
    try:
        # Use all rows in the CSV for prediction
        preds = xgb_clf.predict(df)
        probs = xgb_clf.predict_proba(df)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1f1f1f')
        ax.set_facecolor('#1f1f1f')
        
        if len(preds) == 1:
            # Single company prediction
            bars = ax.bar(['No Bankruptcy', 'Bankruptcy'], probs[0], 
                         color=['#4CAF50', '#F44336'], alpha=0.8)
            ax.set_ylim(0, 1)
            ax.set_title('Bankruptcy Probability', color='white', fontsize=14)
            ax.set_ylabel('Probability', color='white')
            result_text = f"Prediction: {'Bankruptcy Risk' if preds[0] == 1 else 'No Bankruptcy Risk'}\nConfidence: {max(probs[0]):.2%}"
        else:
            # Multiple companies
            bankruptcy_count = np.sum(preds)
            safe_count = len(preds) - bankruptcy_count
            bars = ax.bar(['Safe Companies', 'At Risk Companies'], 
                         [safe_count, bankruptcy_count], 
                         color=['#4CAF50', '#F44336'], alpha=0.8)
            ax.set_title(f'Bankruptcy Analysis for {len(preds)} Companies', color='white', fontsize=14)
            ax.set_ylabel('Number of Companies', color='white')
            result_text = f"Total Companies: {len(preds)}\nSafe: {safe_count}\nAt Risk: {bankruptcy_count}"
        
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return result_text, fig
        
    except Exception as e:
        return f"Error in prediction: {str(e)}", None

def regress_fn(file):
    """Anomaly detection from CSV file"""
    if file is None:
        return "Please upload a CSV file", None
    
    df = process_csv_file(file)
    if df is None:
        return "Error processing file", None
    
    try:
        preds = xgb_reg.predict(df)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1f1f1f')
        ax.set_facecolor('#1f1f1f')
        
        sns.histplot(preds, bins=20, kde=True, ax=ax, color='#00BCD4', alpha=0.7)
        ax.set_title('Anomaly Score Distribution', color='white', fontsize=14)
        ax.set_xlabel('Anomaly Score', color='white')
        ax.set_ylabel('Frequency', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Summary statistics
        avg_score = np.mean(preds)
        high_risk_count = np.sum(preds > np.percentile(preds, 75))
        result_text = f"Average Anomaly Score: {avg_score:.3f}\nHigh Risk Companies: {high_risk_count}/{len(preds)}\nScore Range: {np.min(preds):.3f} - {np.max(preds):.3f}"
        
        return result_text, fig
        
    except Exception as e:
        return f"Error in prediction: {str(e)}", None

def lstm_fn(file):
    """LSTM revenue forecasting from CSV file"""
    if file is None:
        return "Please upload a CSV file", None
    
    df = process_csv_file(file)
    if df is None:
        return "Error processing file", None
    
    try:
        # Expect CSV with revenue columns or a single row with 10 revenue values
        if df.shape[1] < 10:
            return "CSV must contain at least 10 revenue columns for quarterly data", None
        
        # Take first row and first 10 columns as revenue sequence
        vals = df.iloc[0, :10].values.astype(float).reshape(1, -1)
        
        # Scale and predict
        vals_s = scaler_X.transform(vals).reshape((1, vals.shape[1], 1))
        pred_s = lstm_model.predict(vals_s)
        pred = scaler_y.inverse_transform(pred_s)[0, 0]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1f1f1f')
        ax.set_facecolor('#1f1f1f')
        
        quarters = [f'Q{i+1}' for i in range(10)]
        ax.plot(quarters, vals.flatten(), marker='o', linewidth=2, 
                markersize=8, color='#2196F3', label='Historical Revenue')
        ax.plot('Q11', pred, marker='X', markersize=15, color='#FF5722', 
                label=f'Predicted Q11: ${pred:,.0f}')
        
        ax.set_xlabel('Quarter', color='white')
        ax.set_ylabel('Revenue ($)', color='white')
        ax.set_title('Revenue Forecast - Next Quarter Prediction', color='white', fontsize=14)
        ax.legend(facecolor='#2f2f2f', edgecolor='white', labelcolor='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, color='white')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Calculate growth rate
        last_revenue = vals.flatten()[-1]
        growth_rate = ((pred - last_revenue) / last_revenue) * 100
        result_text = f"Predicted Q11 Revenue: ${pred:,.0f}\nGrowth from Q10: {growth_rate:+.1f}%"
        
        return result_text, fig
        
    except Exception as e:
        return f"Error in prediction: {str(e)}", None

# Custom CSS for proper dark mode support
custom_css = """
/* Dark theme for the entire interface */
.gradio-container {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
}

.gr-box {
    background-color: #2d2d2d !important;
    border: 1px solid #404040 !important;
}

.gr-form {
    background-color: #2d2d2d !important;
}

.gr-panel {
    background-color: #2d2d2d !important;
    border: 1px solid #404040 !important;
}

.gr-button {
    background-color: #0066cc !important;
    color: white !important;
    border: none !important;
}

.gr-button:hover {
    background-color: #0052a3 !important;
}

.gr-input, .gr-textbox {
    background-color: #2d2d2d !important;
    border: 1px solid #404040 !important;
    color: #ffffff !important;
}

.gr-upload {
    background-color: #2d2d2d !important;
    border: 2px dashed #404040 !important;
    color: #ffffff !important;
}

.gr-file {
    background-color: #2d2d2d !important;
    color: #ffffff !important;
}

/* Tab styling */
.gr-tab-nav {
    background-color: #2d2d2d !important;
    border-bottom: 1px solid #404040 !important;
}

.gr-tab-nav button {
    background-color: transparent !important;
    color: #ffffff !important;
    border: none !important;
}

.gr-tab-nav button.selected {
    background-color: #0066cc !important;
    color: white !important;
}

/* Text and markdown */
.gr-markdown {
    color: #ffffff !important;
}

.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: #ffffff !important;
}

/* Ensure plot backgrounds work with dark theme */
.gr-plot {
    background-color: #1f1f1f !important;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Base(), title="TriCast AI") as demo:
    gr.Markdown("""
    # 🚀 TriCast AI
    ### Advanced Financial Intelligence Platform
    Upload your company's financial data as a CSV file to get comprehensive AI-powered insights across three key areas.
    """)
    
    gr.Markdown("""
    **📁 CSV File Format Guidelines:**
    - **Bankruptcy & Anomaly Detection**: Include financial metrics as columns (revenue, debt, assets, etc.)
    - **Revenue Forecasting**: First 10 columns should contain quarterly revenue data
    - Each row represents one company's data
    """)
    
    with gr.Tab("🏦 Bankruptcy Risk Assessment"):
        gr.Markdown("**Upload CSV with company financial data to assess bankruptcy risk**")
        with gr.Row():
            with gr.Column():
                file1 = gr.File(label="Upload CSV File", file_types=[".csv"])
                classify_btn = gr.Button("🔍 Analyze Bankruptcy Risk", variant="primary")
            with gr.Column():
                out1 = gr.Textbox(label="Analysis Results", lines=4)
                plt1 = gr.Plot(label="Risk Visualization")
        classify_btn.click(fn=classify_fn, inputs=file1, outputs=[out1, plt1])
    
    with gr.Tab("📊 Anomaly Detection"):
        gr.Markdown("**Upload CSV with company financial data to detect anomalies**")
        with gr.Row():
            with gr.Column():
                file2 = gr.File(label="Upload CSV File", file_types=[".csv"])
                regress_btn = gr.Button("🔎 Detect Anomalies", variant="primary")
            with gr.Column():
                out2 = gr.Textbox(label="Anomaly Analysis", lines=4)
                plt2 = gr.Plot(label="Score Distribution")
        regress_btn.click(fn=regress_fn, inputs=file2, outputs=[out2, plt2])
    
    with gr.Tab("📈 Revenue Forecasting"):
        gr.Markdown("**Upload CSV with quarterly revenue data (10 quarters) to forecast next quarter**")
        with gr.Row():
            with gr.Column():
                file3 = gr.File(label="Upload CSV File", file_types=[".csv"])
                forecast_btn = gr.Button("📊 Forecast Revenue", variant="primary")
            with gr.Column():
                out3 = gr.Textbox(label="Forecast Results", lines=4)
                plt3 = gr.Plot(label="Revenue Trend & Prediction")
        forecast_btn.click(fn=lstm_fn, inputs=file3, outputs=[out3, plt3])
    
    with gr.Tab("📋 Sample Data Format"):
        gr.Markdown("""
        ### Sample CSV Formats:
        
        **For Bankruptcy & Anomaly Detection:**
        ```
        company_name,total_assets,total_liabilities,revenue,debt_ratio,current_ratio
        Company A,1000000,500000,800000,0.5,2.1
        Company B,2000000,1800000,600000,0.9,0.8
        ```
        
        **For Revenue Forecasting:**
        ```
        q1_revenue,q2_revenue,q3_revenue,q4_revenue,q5_revenue,q6_revenue,q7_revenue,q8_revenue,q9_revenue,q10_revenue
        100000,120000,110000,130000,125000,140000,135000,150000,145000,160000
        ```
        """)
    
    gr.Markdown("---")
    gr.Markdown("*TriCast AI - Powered by Advanced Machine Learning | Industry, Innovation and Infrastructure*")

if __name__ == "__main__":
    demo.launch()