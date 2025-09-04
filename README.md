# TriCast AI — Bankruptcy Prediction & Analysis

> *“What if companies didn’t have to wait for disaster to know they were heading toward bankruptcy?”*  

This was the question I set out to answer when building **TriCast AI** — an AI-powered bankruptcy prediction system that doesn’t just crunch numbers but actually provides early warning signals for companies, banks, and investors.  

I led this project as part of my **Professional Practices course at UMT, Lahore**, where we had to build something that wasn’t just academic but could *live in the real world*. We mapped it to **SDG 9 (Industry, Innovation & Infrastructure)** because, let’s be honest, financial resilience is infrastructure.  

And yes — we even pitched it to a real company, **Pak Laser Engrave (Pvt.) Ltd.**, who validated the idea and told us how useful it would be in actual operations. They especially highlighted workforce planning and data security as areas where our tool could be a game-changer.  

🔗 **Live Demo on Hugging Face Spaces**:  
👉 [Try TriCast AI here](https://huggingface.co/spaces/AbdullahImran/TriCast-AI)

---

## 🚀 What is TriCast AI?

TriCast AI is an **end-to-end bankruptcy prediction system**. From raw financial data (balance sheets, income statements, cash flow reports) to predictions and deployment — everything was handled, tested, and delivered.  

The app provides:  
1. **Risk Classification** → *At Risk* vs. *Not At Risk*  
2. **Anomaly Detection** → unusual financial signatures flagged  
3. **Time-Series Forecasting** → LSTM predicts next quarter revenue trends  

All of this is wrapped into a **web app** where users just upload a CSV file and instantly see their company’s financial risk profile.

---

## 🛠️ Process & Workflow

We didn’t just train a single model. We went full-cycle:

### 1. Exploratory Data Analysis (EDA)
- Used the **Kaggle "Financial Performance Prediction" dataset**.
- Cleaned missing values, removed anomalies, and engineered meaningful financial ratios (debt-to-assets, current ratio, profit margins, etc.).
- Visualized class imbalance (bankrupt companies are rare) and patterns like declining revenues before failure.

### 2. Data Preprocessing & Feature Engineering
- Filled missing categorical values with mode.
- Numerical features imputed with global medians.
- Time-series gaps filled with forward/backward fill + rolling means.
- Scaled sequences for stable LSTM training.
- Added macroeconomic signals (interest rates, inflation, GDP) as contextual features.

### 3. Model Training
We trained and compared multiple models:
- **Logistic Regression & Altman Z-Score** → baselines
- **XGBoost Classifier** → bankruptcy risk (binary classification)  
- **XGBoost Regressor** → anomaly scoring (0–1 risk score)  
- **LSTM** → revenue forecasting for the next quarter  

> 🎯 Final Accuracy Highlights:  
> - XGBoost Classifier: **99% Accuracy** (94% F1)  
> - XGBoost Regressor: RMSE ~ **0.195**  
> - LSTM: MAE ~ **0.0827**  

### 4. Model Testing & Evaluation
- Evaluated with ROC-AUC, Precision-Recall, and backtesting on historical bankruptcies.
- Sensitivity analysis for noisy/missing inputs.
- Combined classifier + anomaly + LSTM into a blended risk profile.

### 5. Deployment
- Built a simple but effective UI using **Gradio/Streamlit**.
- Deployed on **Hugging Face Spaces** for instant global access.
- Users can upload CSVs, get predictions, and see visual risk dashboards.
- Security: all processing happens in memory, with no data stored.

### 6. Pitching to Industry
On **June 2, 2025**, we pitched TriCast AI to **Pak Laser Engrave (Pvt.) Ltd.** in Lahore.  
- **Mr. Muhammad Asif (Head Manager)** called it “practically useful and timely,” relating it to real supplier/payment issues.  
- **Mr. Shahwaiz Ahmad (HR Manager)** suggested tying risk forecasts to workforce planning (to avoid layoffs).  
- Accounts & Production teams asked implementation-level questions, validating that this wasn’t “just student work.”  

Their feedback shaped our roadmap: stronger data security, external economic feature integration, and HR-linked forecasting.

---

## 🌍 Why It Matters (SDG 9 Alignment)
- **Early Action Saves Jobs & Investments** → catching problems before collapse.  
- **Financial Inclusion for SMEs** → small firms can access smarter tools, improving loan chances.  
- **Resilient Infrastructure** → builds trust with banks, investors, and regulators.  

This wasn’t just a project for a grade — it was a **working prototype deployed to the world**.

---

## 📂 Repository Structure
```

📦 TriCast-AI
┣ 📜 README.md
┣ 📜 requirements.txt
┣ 📂 notebooks      # EDA, preprocessing, model training
┣ 📂 models         # Saved XGBoost + LSTM models
┣ 📂 app            # Streamlit/Gradio UI code
┗ 📂 data           # Sample cleaned datasets

```

---

## ⚡ Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow/Keras)  
- **EDA & Visualization**: Matplotlib, Seaborn  
- **Deployment**: Hugging Face Spaces + Gradio  
- **Versioning & Docs**: Git, Google Docs 

---

## 🧑‍💻 Team (PP-Y10-04, UMT Lahore)
I spearheaded the **idea proposal, development, training, testing, deployment, and the bulk of this documentation**.  
Other teammates contributed in documentation, pitching, and structuring roles.  

- **Abdullah Imran (main contributer)** → Development lead (EDA → Deployment)  
- Shanzy Abid → Meeting lead, project structuring  
- Eman Ali → Stakeholder engagement, company pitching  
- Malik Affan → 5Ws & 1H, budgeting  
- Ahmer Nazeer → Evaluated rejected ideas  

---

## 🎯 Future Enhancements
- Link risk scores with HR & workforce planning dashboards.  
- Add external signals: FX rates, global market shocks, sector-specific indices.  
- Build explainable AI dashboards (feature importance per prediction).  
- Sector-specific models (manufacturing, services, banking).  

---

## 🖥️ Try It Yourself
👉 [TriCast AI on Hugging Face Spaces](https://huggingface.co/spaces/AbdullahImran/TriCast-AI)

Just upload your company’s financial CSV and let the AI do the heavy lifting.  

---

### 📌 Final Word
TriCast AI was born out of a classroom, but it didn’t stay there. It’s been tested, pitched, and validated in the real world.  
This isn’t “just another ML project” — it’s a financial resilience tool that can actually make an impact.  
