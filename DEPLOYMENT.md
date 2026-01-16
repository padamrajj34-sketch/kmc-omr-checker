# Full Deployment Guide: KMC OMR Checker on Streamlit Community Cloud

## Prerequisites
- GitHub account
- Streamlit Community Cloud account
- Local project ready in a Git repo

---

## Step 1: Prepare Local Git Repository

```bash
# Initialize repo (if not already done)
git init
git add .
git commit -m "Initial commit: KMC OMR Checker"

# Create .gitignore (if missing)
echo "__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
.env
.DS_Store
.vscode/
.idea/
*.log
data/
*.sqlite3
*.db
poppler-*/" > .gitignore

git add .gitignore
git commit -m "Add .gitignore"
```

---

## Step 2: Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in.
2. Click **New repository**.
3. Repository name: `kmc-omr-checker` (or your choice).
4. Set to **Public** (required for free Streamlit Cloud).
5. **Do not** initialize with README, license, or .gitignore (we already have them).
6. Click **Create repository**.

---

## Step 3: Push Local Code to GitHub

Copy the HTTPS URL from the GitHub repo page (e.g., `https://github.com/username/kmc-omr-checker.git`).

```bash
git remote add origin https://github.com/username/kmc-omr-checker.git
git branch -M main
git push -u origin main
```

Verify: Open your GitHub repo – you should see `app.py`, `requirements.txt`, etc.

---

## Step 4: Deploy on Streamlit Community Cloud

1. Go to [Streamlit Community Cloud](https://share.streamlit.io).
2. Click **Sign in** and authorize with GitHub.
3. Click **New app**.
4. **Connect repository**:
   - Repository: select `kmc-omr-checker`.
   - Branch: `main`.
   - Main file path: `app.py`.
5. Click **Deploy**.

Streamlit will:
- Clone your repo
- Install dependencies from `requirements.txt`
- Launch `streamlit run app.py`

---

## Step 5: Configure Secrets (Optional)

In your Streamlit app dashboard:
- Go to **Settings → Secrets**.
- Add a secret named `admin_password` with your desired admin password.
- This overrides the default `kmc-2081-admin`.

---

## Step 6: Verify Deployment

- Your app will be available at `https://username-kmc-omr-checker-app-xxxxxx.streamlit.app`.
- Test uploading an OMR sheet.
- Switch to **Answer Key Admin** and verify password protection.

---

## Troubleshooting

### 404 NOT_FOUND on Streamlit Cloud
- Ensure `app.py` is in the repo root.
- Confirm the branch name matches (`main`).
- In Streamlit settings, double-check the repository URL and main file path.

### Build Failures
- Check the **Logs** tab in Streamlit for dependency errors.
- Ensure `requirements.txt` lists all imports used in `app.py` and `utils.py`.
- Poppler/Tesseract errors are OK on Cloud (PDF/OCR optional).

### Permission Errors
- Make sure the GitHub repo is **Public**.
- Re-authorize Streamlit Cloud to access your repo if needed.

---

## Post-Deploy Workflow

- To update: `git add . && git commit -m "Update" && git push`
- Streamlit auto-redeploys on push to the configured branch.
- Monitor logs in the Streamlit dashboard for any runtime issues.

---

## Alternative: Render.com

If Streamlit Cloud is unavailable:

1. Push to GitHub as above.
2. Go to [Render](https://render.com).
3. Create a **Web Service** → connect GitHub repo.
4. Runtime: **Python**.
5. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
6. Add environment variables/secrets as needed.
7. Deploy.

---

## Summary

1. Local git → commit → push to GitHub.
2. Create public GitHub repo.
3. Connect repo on Streamlit Cloud.
4. Deploy and test.
5. Set secrets for admin password.
6. Update by pushing to GitHub.

Your KMC OMR Checker will be live and publicly accessible.
