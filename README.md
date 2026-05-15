# DiabetesScan — Streamlit App

**Mohan Sundi | Diploma in CSE 2023 - 2026 | Jamshedpur**

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
Open: http://localhost:8501

---

## Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "DiabetesScan Streamlit App"
git remote add origin https://github.com/GrimmSmith/diabetesscan-streamlit.git
git push -u origin main
```

### Step 2 — Deploy
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **New app**
4. Select your repo → branch: `main` → Main file: `app.py`
5. Click **Deploy!**

Your app will be live at:
`https://diabetesscan-streamlit-grimmsmith.streamlit.app`

---

## Project Structure
```
DiabetesScan_Streamlit/
├── app.py               ← Main Streamlit app
├── best_model.pkl       ← Trained ML model bundle
├── requirements.txt     ← Python dependencies
├── .streamlit/
│   └── config.toml      ← Dark theme config
└── README.md
```
