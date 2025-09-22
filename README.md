<h1 align="center">📊 TriCast AI — Bankruptcy Prediction & Analysis</h1>
<h3 align="center">AI-powered early warning system for financial resilience</h3>

> *“What if companies didn’t have to wait for disaster to know they were heading toward bankruptcy?”*  

---

## 🌟 Overview  
**TriCast AI** is an **end-to-end bankruptcy prediction system** that provides **early warning signals** for companies, banks, and investors.  

✅ Built as part of the **Professional Practices course at UMT, Lahore**  
✅ Mapped to **SDG 9 (Industry, Innovation & Infrastructure)**  
✅ Validated by **Pak Laser Engrave (Pvt.) Ltd.** with direct feedback  

🔗 **Live Demo** → [Try TriCast AI](https://huggingface.co/spaces/AbdullahImran/TriCast-AI)  

---

## 🚀 What Does It Do?  
TriCast AI helps organizations:  
1. **Risk Classification** → *At Risk* vs. *Not At Risk*  
2. **Anomaly Detection** → unusual financial signatures flagged  
3. **Time-Series Forecasting** → LSTM predicts next quarter revenue  

All via a **web app** where users upload a CSV and instantly see their risk profile.  

---

## 🛠️ Process & Workflow  

### 🔹 1. Exploratory Data Analysis (EDA)  
- Used **Kaggle Financial Performance dataset**  
- Cleaned missing values & engineered ratios (debt-to-assets, margins, etc.)  
- Identified class imbalance (few bankrupt firms) and financial decline patterns  

### 🔹 2. Preprocessing & Feature Engineering  
- Mode + median imputation  
- Time-series gap filling (forward/backward fill + rolling means)  
- Scaled sequences for stable LSTM training  
- Added macroeconomic signals (GDP, inflation, interest rates)  

### 🔹 3. Model Training  
- **Baselines**: Logistic Regression, Altman Z-Score  
- **XGBoost Classifier** → Bankruptcy risk (binary)  
- **XGBoost Regressor** → Anomaly risk score (0–1)  
- **LSTM** → Revenue forecasting  

🎯 **Performance Highlights**  
- XGBoost Classifier: **99% Accuracy (94% F1)**  
- XGBoost Regressor: **RMSE ~0.195**  
- LSTM: **MAE ~0.0827**  

### 🔹 4. Evaluation  
- ROC-AUC, Precision-Recall, backtesting  
- Sensitivity to missing/noisy inputs  
- Blended final risk score (classifier + anomaly + LSTM)  

### 🔹 5. Deployment  
- UI: **Gradio / Streamlit**  
- Hosting: **Hugging Face Spaces**  
- Risk dashboards + CSV upload  
- Security: in-memory processing, no data stored  

### 🔹 6. Industry Pitch  
On **June 2, 2025**, pitched to **Pak Laser Engrave (Pvt.) Ltd.**:  
- ✅ *Head Manager*: “Practically useful and timely”  
- ✅ *HR Manager*: Suggested workforce planning integration  
- ✅ *Accounts & Production*: Validated real-world use cases  

---

## 🌍 Why It Matters (SDG 9)  
- 🏢 **Early Action Saves Jobs & Investments**  
- 💡 **Financial Inclusion for SMEs** → smarter loan chances  
- 🛡️ **Resilient Infrastructure** → trust with banks & regulators  

---

## ⚡ Tech Stack  
**Languages & Libraries** → Python, Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow/Keras  
**Visualization** → Matplotlib, Seaborn  
**Deployment** → Hugging Face Spaces, Gradio/Streamlit  
**Collaboration** → Git, Google Docs  

---

## 🧑‍🤝‍🧑 Team (PP-Y10-04, UMT Lahore)  
- **Abdullah Imran** → *Lead Developer* (EDA → Deployment, Documentation)  
- Shanzy Abid → Meetings, project structuring  
- Eman Ali → Stakeholder engagement, pitching  
- Malik Affan → 5Ws & 1H, budgeting  
- Ahmer Nazeer → Rejected idea evaluation  

---

## 🎯 Future Enhancements  
- HR-linked workforce planning dashboards  
- External signals: FX, sector-specific indices  
- Explainable AI dashboards (feature importance)  
- Sector-specialized models (manufacturing, services, banking)  

---

## 🖥️ Try It Yourself  
👉 [TriCast AI on Hugging Face Spaces](https://huggingface.co/spaces/AbdullahImran/TriCast-AI)  

📂 Upload your company’s financial CSV → ⚡ Get instant risk insights  

---

## 📌 Final Word  
TriCast AI started in a classroom — but it didn’t stay there.  
It’s been **tested, pitched, validated**, and deployed to the real world.  

⚡ This isn’t “just another ML project” — it’s a tool for **financial resilience**.
