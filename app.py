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

def run_all_models(file):
    """Run all three models on the uploaded CSV file"""
    if file is None:
        return "Please upload a CSV file", None, None, None, None, None
    
    df = process_csv_file(file)
    if df is None:
        return "Error processing file", None, None, None, None, None
    
    try:
        # Prepare data for models (assuming same feature set as training)
        model_features = df.copy()
        
        # Remove non-feature columns if they exist
        cols_to_remove = ['Id', 'anomaly_score', 'risk_flag'] 
        for col in cols_to_remove:
            if col in model_features.columns:
                model_features = model_features.drop(col, axis=1)
        
        # Handle missing values
        model_features = model_features.fillna(0)

        for col in model_features.select_dtypes(include=['object']).columns:
            model_features[col] = model_features[col].astype(str)
            model_features[col] = model_features[col].fillna("Unknown")
            model_features[col] = model_features[col].astype("category").cat.codes
                
        # 1. BANKRUPTCY CLASSIFICATION
        bankruptcy_preds = xgb_clf.predict(model_features)
        bankruptcy_probs = xgb_clf.predict_proba(model_features)
        
        # Create bankruptcy visualization
        fig1, ax1 = plt.subplots(figsize=(10, 6), facecolor='#1f1f1f')
        ax1.set_facecolor('#1f1f1f')
        
        if len(bankruptcy_preds) == 1:
            bars = ax1.bar(['No Bankruptcy', 'Bankruptcy'], bankruptcy_probs[0], 
                          color=['#4CAF50', '#F44336'], alpha=0.8)
            ax1.set_ylim(0, 1)
            ax1.set_title('Bankruptcy Risk Probability', color='white', fontsize=14)
            ax1.set_ylabel('Probability', color='white')
            bankruptcy_result = f"Prediction: {'High Bankruptcy Risk' if bankruptcy_preds[0] == 1 else 'Low Bankruptcy Risk'}\nConfidence: {max(bankruptcy_probs[0]):.2%}"
        else:
            bankruptcy_count = np.sum(bankruptcy_preds)
            safe_count = len(bankruptcy_preds) - bankruptcy_count
            bars = ax1.bar(['Safe Companies', 'At Risk Companies'], 
                          [safe_count, bankruptcy_count], 
                          color=['#4CAF50', '#F44336'], alpha=0.8)
            ax1.set_title(f'Bankruptcy Analysis for {len(bankruptcy_preds)} Companies', color='white', fontsize=14)
            ax1.set_ylabel('Number of Companies', color='white')
            bankruptcy_result = f"Total Companies: {len(bankruptcy_preds)}\nSafe: {safe_count}\nAt Risk: {bankruptcy_count}"
        
        ax1.tick_params(colors='white')
        ax1.spines['bottom'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        plt.tight_layout()
        
        # 2. ANOMALY DETECTION
        anomaly_preds = xgb_reg.predict(model_features)
        
        # Create anomaly visualization
        fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor='#1f1f1f')
        ax2.set_facecolor('#1f1f1f')
        
        sns.histplot(anomaly_preds, bins=20, kde=True, ax=ax2, color='#00BCD4', alpha=0.7)
        ax2.set_title('Anomaly Score Distribution', color='white', fontsize=14)
        ax2.set_xlabel('Anomaly Score', color='white')
        ax2.set_ylabel('Frequency', color='white')
        ax2.tick_params(colors='white')
        ax2.spines['bottom'].set_color('white')
        ax2.spines['left'].set_color('white')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.tight_layout()
        
        avg_score = np.mean(anomaly_preds)
        high_risk_count = np.sum(anomaly_preds > np.percentile(anomaly_preds, 75))
        anomaly_result = f"Average Anomaly Score: {avg_score:.3f}\nHigh Risk Companies: {high_risk_count}/{len(anomaly_preds)}\nScore Range: {np.min(anomaly_preds):.3f} - {np.max(anomaly_preds):.3f}"
        
        # 3. LSTM REVENUE FORECASTING
        # Extract revenue data from Q1_REVENUES to Q10_REVENUES
        revenue_cols = [f'Q{i}_REVENUES' for i in range(1, 11)]
        missing_cols = [col for col in revenue_cols if col not in df.columns]
        
        if missing_cols:
            lstm_result = f"Missing revenue columns for LSTM: {missing_cols}"
            fig3 = plt.figure(figsize=(10, 6), facecolor='#1f1f1f')
            ax3 = fig3.add_subplot(111, facecolor='#1f1f1f')
            ax3.text(0.5, 0.5, 'Revenue columns not found in dataset', 
                    ha='center', va='center', color='white', fontsize=14)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
        else:
            # Use first company's revenue data for LSTM prediction
            revenue_data = df[revenue_cols].iloc[0].values.astype(float)
            
            # Handle missing values in revenue data
            if np.any(np.isnan(revenue_data)) or np.any(revenue_data == 0):
                # Replace NaN and zeros with interpolated values
                mask = ~np.isnan(revenue_data) & (revenue_data != 0)
                if np.sum(mask) > 1:
                    revenue_data[~mask] = np.interp(np.where(~mask)[0], np.where(mask)[0], revenue_data[mask])
                else:
                    revenue_data = np.full_like(revenue_data, np.mean(revenue_data[mask]) if np.sum(mask) > 0 else 1000000)
            
            revenue_data = revenue_data.reshape(1, -1)
            
            # Scale and predict
            revenue_scaled = scaler_X.transform(revenue_data).reshape((1, revenue_data.shape[1], 1))
            pred_scaled = lstm_model.predict(revenue_scaled)
            predicted_revenue = scaler_y.inverse_transform(pred_scaled)[0, 0]
            
            # Create LSTM visualization
            fig3, ax3 = plt.subplots(figsize=(12, 6), facecolor='#1f1f1f')
            ax3.set_facecolor('#1f1f1f')
            
            quarters = [f'Q{i}' for i in range(1, 11)]
            ax3.plot(quarters, revenue_data.flatten(), marker='o', linewidth=2, 
                    markersize=8, color='#2196F3', label='Historical Revenue')
            ax3.plot('Q11', predicted_revenue, marker='X', markersize=15, color='#FF5722', 
                    label=f'Predicted Q11: ${predicted_revenue:,.0f}')
            
            ax3.set_xlabel('Quarter', color='white')
            ax3.set_ylabel('Revenue ($)', color='white')
            ax3.set_title('Revenue Forecast - Next Quarter Prediction', color='white', fontsize=14)
            ax3.legend(facecolor='#2f2f2f', edgecolor='white', labelcolor='white')
            ax3.tick_params(colors='white')
            ax3.spines['bottom'].set_color('white')
            ax3.spines['left'].set_color('white')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.grid(True, alpha=0.3, color='white')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Calculate growth rate
            last_revenue = revenue_data.flatten()[-1]
            growth_rate = ((predicted_revenue - last_revenue) / last_revenue) * 100
            lstm_result = f"Predicted Q11 Revenue: ${predicted_revenue:,.0f}\nGrowth from Q10: {growth_rate:+.1f}%\nLast Quarter (Q10): ${last_revenue:,.0f}"
        
        return bankruptcy_result, fig1, anomaly_result, fig2, lstm_result, fig3
        
    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        return error_msg, None, error_msg, None, error_msg, None

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
    ### Comprehensive Financial Intelligence Platform
    Upload your company's financial data CSV file to get AI-powered insights across three key areas **simultaneously**.
    """)
    
    gr.Markdown("""
    **📁 Expected CSV Format:**
    Your CSV should contain financial metrics including:
    - Basic info: `industry`, `sector`, `fullTimeEmployees`
    - Risk metrics: `auditRisk`, `boardRisk`, `compensationRisk`, etc.
    - Financial ratios: `trailingPE`, `forwardPE`, `totalDebt`, `totalRevenue`, etc.
    - Quarterly data: `Q1_REVENUES`, `Q2_REVENUES`, ..., `Q10_REVENUES` (for LSTM forecasting)
    - Quarterly financials: `Q*_TOTAL_ASSETS`, `Q*_TOTAL_LIABILITIES`, etc.
    
    📊 **One Upload = Three AI Models Running Simultaneously!**
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="📁 Upload Company Financial Data (CSV)", 
                file_types=[".csv"],
                elem_id="file_upload"
            )
            analyze_btn = gr.Button(
                "🚀 Run TriCast AI Analysis", 
                variant="primary", 
                size="lg"
            )
    
    gr.Markdown("---")
    

    # Results section with three columns
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🏦 Bankruptcy Risk Assessment")
            bankruptcy_output = gr.Textbox(
                label="Risk Analysis", 
                lines=4,
                placeholder="Results will appear here..."
            )
            bankruptcy_plot = gr.Plot(label="Risk Visualization")
        
        with gr.Column():
            gr.Markdown("### 📊 Anomaly Detection")
            anomaly_output = gr.Textbox(
                label="Anomaly Analysis", 
                lines=4,
                placeholder="Results will appear here..."
            )
            anomaly_plot = gr.Plot(label="Score Distribution")
        
        with gr.Column():
            gr.Markdown("### 📈 Revenue Forecasting")
            lstm_output = gr.Textbox(
                label="Forecast Summary", 
                lines=4,
                placeholder="Results will appear here..."
            )
            lstm_plot = gr.Plot(label="Revenue Forecast")

    analyze_btn.click(
    run_all_models,
    inputs=[file_input],
    outputs=[bankruptcy_output, bankruptcy_plot, anomaly_output, anomaly_plot, lstm_output, lstm_plot]
    )

if __name__ == "__main__":

    demo.launch()
