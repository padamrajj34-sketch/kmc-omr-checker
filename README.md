# KMC Class 11 Scholarship Examination OMR Checker

A Streamlit-based web application that processes the Kathmandu Metropolitan City (KMC) Class 11 Scholarship Examination 2081 (BS) OMR sheets. The tool performs perspective correction, bubble detection, answer extraction, scoring, and offers an admin interface for maintaining the official answer key.

## Features
- Upload OMR scans as **JPG, PNG, or PDF** (first page processed).
- Automatic sheet detection, alignment, and bubble analysis using OpenCV.
- Handles 100 MCQs arranged in five columns of 20 rows each (options A–D).
- Default answer key seeded with alternating A/B/C/D responses.
- Admin interface (password protected) for editing or bulk-importing the answer key.
- Manual overrides for detected answers with on-the-fly rescoring.
- Downloadable overlay visualizing detected marks and correctness.
- Experimental OCR for roll number extraction (requires Tesseract installed locally).
- Works entirely in-memory; no persistence of uploaded OMR images.

## Project Structure

```
.
├── app.py           # Streamlit front-end and workflow orchestration
├── database.py      # SQLite utilities for answer key persistence
├── utils.py         # Image processing and scoring helpers (OpenCV, NumPy)
├── requirements.txt # Python dependencies
└── data/omr_key.sqlite3  # Created automatically on first run
```

## Installation

1. **Install system dependencies**
   - Python 3.9+
   - For PDF support: install [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) (Windows) or via package manager (`brew install poppler` on macOS, `sudo apt install poppler-utils` on Ubuntu).
   - For OCR support (optional): install [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html) and ensure `pytesseract` can access it.

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app locally**

   ```bash
   streamlit run app.py
   ```

   The app will launch in your browser at `http://localhost:8501`.

## Usage

1. Navigate to the **OMR Checker** tab.
2. Review the scanning tips, then upload the scanned OMR sheet (JPG/PNG/PDF).
3. Wait for automatic detection and scoring. Review bubble-by-bubble results, manual overrides, and visual overlays.
4. Optional: Download the annotated overlay or adjust responses manually for auditing.

### Admin Mode

1. Switch to **Answer Key Admin** from the sidebar.
2. Enter the admin password (default `kmc-2081-admin`; override via Streamlit secrets).
3. Edit the answer key directly in the table or upload a CSV file with `question_no,correct_option` columns.
4. Download the current answer key as CSV for backups or bulk edits.

## Deployment Guide

### Streamlit Community Cloud
1. Fork/clone this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app pointing to `app.py` in your repository.
4. Set the `admin_password` secret (optional) under **App Settings → Secrets**.
5. Deploy. Subsequent commits to main will auto-redeploy.

### Render.com (Flask alternative)
1. Create a new **Web Service** -> select your repo.
2. Set runtime to Python with start command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
3. Add environment variables/secrets as needed (e.g., `admin_password`).
4. Deploy and test.

### General Tips
- Ensure the `requirements.txt` mirrors this project to include OpenCV and pdf2image.
- For PDF processing on Linux, install Poppler via package manager on the deployment platform.
- If using Tesseract OCR, make sure the binary is installed on the host and accessible in `PATH`.
- Monitor logs for `SheetDetectionError` to debug failed detections; adjust scans or bubble fill quality.

## Troubleshooting

- **Sheet not detected**: Ensure all borders are visible, increase scan contrast, or trim backgrounds.
- **Wrong bubbles detected**: Re-scan with higher DPI or better lighting; consider editing answers manually.
- **PDF uploads failing**: Confirm Poppler is installed and accessible. Without it, only image inputs are supported.
- **OCR empty**: Install Tesseract or adjust the ROI cropping logic in `utils.extract_roll_number` for your scans.

## License

This project is provided as-is for educational and administrative use by Kathmandu Metropolitan City for the Class 11 Scholarship Examination 2081 (BS).
