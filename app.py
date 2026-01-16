"""Streamlit application for the Kathmandu Metropolitan City Class 11 Scholarship OMR checker."""
from __future__ import annotations

import io
import json
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from database import (
    CHOICES,
    TOTAL_QUESTIONS,
    export_answer_key,
    fetch_answer_key,
    init_db,
    replace_answer_key,
)
from utils import (
    ProcessedSheet,
    SheetDetectionError,
    encode_image_to_png_bytes,
    load_image_from_upload,
    process_omr_sheet,
    score_answers,
)

st.set_page_config(
    page_title="KMC Class 11 Scholarship OMR Checker",
    layout="wide",
    page_icon="📝",
)

init_db()

ADMIN_PASSWORD = st.secrets.get("admin_password", "kmc-2081-admin")


def render_header():
    st.title("Kathmandu Metropolitan City | Class 11 Scholarship OMR Checker")
    st.caption(
        "Upload the scanned OMR sheet for the 2081 (BS) examination to automatically compute scores."
    )


def build_answer_key_dataframe(answer_key: Dict[int, str]) -> pd.DataFrame:
    data = {
        "Question": list(range(1, TOTAL_QUESTIONS + 1)),
        "Correct Option": [answer_key.get(q, "") for q in range(1, TOTAL_QUESTIONS + 1)],
    }
    df = pd.DataFrame(data)
    df["Correct Option"] = df["Correct Option"].replace("", np.nan)
    return df


def render_instructions():
    with st.expander("📋 Scanning and Upload Instructions", expanded=True):
        st.markdown(
            """
            **To ensure accurate detection:**
            1. Place the OMR sheet on a flat surface with good lighting.
            2. Avoid shadows, folds, or glare across the sheet.
            3. Capture the full sheet including the dark border markers.
            4. Save the image as **JPG/PNG** using 300 DPI if possible; **PDF** uploads are also supported.
            5. Verify that bubbles are filled completely with a black or dark blue pen.
            """
        )


def render_processing_results(sheet: ProcessedSheet, answer_key: Dict[int, str]):
    results_data = []
    mismatches = []
    for result in sheet.question_results:
        correct_option = answer_key.get(result.question_no)
        status = "Blank"
        if result.multi_marked:
            status = "Multiple"
        elif result.selected_option is not None:
            status = "Correct" if result.selected_option == correct_option else "Incorrect"
        results_data.append(
            {
                "Question": result.question_no,
                "Detected": result.selected_option or "",
                "Correct": correct_option or "",
                "Status": status,
                "Fill Ratios": json.dumps(result.fill_ratios, ensure_ascii=False),
            }
        )
        if status == "Incorrect":
            mismatches.append(result.question_no)

    score = score_answers(sheet.question_results, answer_key)
    st.subheader("Score Summary")
    cols = st.columns(5)
    cols[0].metric("Score", f"{score.score}/{score.total_questions}")
    cols[1].metric("Percentage", f"{score.percentage:.1f}%")
    cols[2].metric("Correct", score.correct)
    cols[3].metric("Incorrect", score.incorrect)
    cols[4].metric("Blank / Multi", f"{score.blank} blank, {score.multi_marked} multi")

    if score.percentage >= 40:
        st.success("Pass threshold met (>= 40%).")
    else:
        st.warning("Score below 40% pass benchmark.")

    st.markdown("### Detailed Question Analysis")
    df = pd.DataFrame(results_data)
    edited_df = st.data_editor(
        df,
        key="results_editor",
        disabled_columns=["Fill Ratios"],
        column_config={
            "Detected": st.column_config.SelectboxColumn(options=("",) + CHOICES),
            "Status": st.column_config.SelectboxColumn(
                options=("Correct", "Incorrect", "Blank", "Multiple")
            ),
        },
        use_container_width=True,
    )

    if st.button("Recalculate score with manual adjustments"):
        overrides = {}
        for _, row in edited_df.iterrows():
            detected = row["Detected"]
            overrides[int(row["Question"])] = detected if detected else None
        adjusted_results = []
        for result in sheet.question_results:
            override_value = overrides.get(result.question_no, result.selected_option)
            adjusted_results.append(
                type(result)(
                    question_no=result.question_no,
                    selected_option=override_value,
                    multi_marked=result.multi_marked,
                    fill_ratios=result.fill_ratios,
                )
            )
        score_override = score_answers(adjusted_results, answer_key)
        st.info(
            f"Adjusted Score: {score_override.score}/{score_override.total_questions}"
            f" ({score_override.percentage:.1f}%)."
        )

    if mismatches:
        st.markdown(
            "**Questions to review manually:** "
            + ", ".join(f"Q{num}" for num in mismatches)
        )

    aligned_png = encode_image_to_png_bytes(sheet.aligned_image)
    overlay_png = encode_image_to_png_bytes(sheet.overlay_image)
    thresh_visual = encode_image_to_png_bytes(cv_to_rgb(sheet.threshold_image))

    st.markdown("### Visual Feedback")
    tab_aligned, tab_overlay, tab_thresh = st.tabs(
        ["Aligned Sheet", "Evaluation Overlay", "Threshold Mask"]
    )
    with tab_aligned:
        st.image(aligned_png, caption="Perspective-corrected OMR sheet")
    with tab_overlay:
        st.image(overlay_png, caption="Detected bubbles (green=correct, red=filled but wrong)")
        st.download_button(
            "Download overlay as PNG",
            data=overlay_png,
            file_name="kmc_omr_overlay.png",
            mime="image/png",
        )
    with tab_thresh:
        st.image(thresh_visual, caption="Binary mask used for bubble detection")

    if sheet.roll_number:
        st.info(f"Detected Roll Number (experimental OCR): {sheet.roll_number}")
    else:
        st.caption("Roll number OCR not available or detection failed.")


def cv_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.merge([image, image, image])
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def render_omr_checker(answer_key: Dict[int, str]):
    render_instructions()
    uploaded = st.file_uploader(
        "Upload OMR sheet image or PDF", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=False
    )
    if not uploaded:
        st.info("Awaiting upload…")
        return

    uploaded.seek(0)
    try:
        with st.spinner("Processing OMR sheet…"):
            image = load_image_from_upload(uploaded)
            sheet = process_omr_sheet(image, answer_key)
    except SheetDetectionError as err:
        st.error(f"Sheet detection failed: {err}")
        return
    except RuntimeError as err:
        st.error(f"Processing error: {err}")
        return
    except Exception as exc:
        st.exception(exc)
        return

    render_processing_results(sheet, answer_key)


def render_admin_mode(answer_key: Dict[int, str]):
    st.subheader("Answer Key Administration")
    password_input = st.text_input("Admin password", type="password")
    if password_input != ADMIN_PASSWORD:
        st.warning("Enter the administrator password to edit the answer key.")
        return

    st.success("Authenticated. Update the answer key below.")
    df = build_answer_key_dataframe(answer_key)
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        column_config={
            "Correct Option": st.column_config.SelectboxColumn(options=CHOICES),
        },
        use_container_width=True,
    )
    if st.button("Save changes", type="primary"):
        try:
            entries = [
                (int(row["Question"]), str(row["Correct Option"]).strip().upper())
                for _, row in edited_df.iterrows()
            ]
            replace_answer_key(entries)
            st.success("Answer key updated successfully.")
        except Exception as exc:
            st.error(f"Failed to update answer key: {exc}")

    st.markdown("#### Bulk upload (CSV)")
    uploaded_csv = st.file_uploader(
        "Upload CSV with columns question_no, correct_option", type=["csv"], key="csv_uploader"
    )
    if uploaded_csv is not None:
        try:
            csv_df = pd.read_csv(uploaded_csv)
            entries = list(zip(csv_df["question_no"], csv_df["correct_option"]))
            replace_answer_key(entries)
            st.success("CSV answer key imported.")
        except Exception as exc:
            st.error(f"Failed to import CSV: {exc}")

    current_key_df = build_answer_key_dataframe(fetch_answer_key())
    csv_buffer = io.StringIO()
    current_key_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download current key as CSV",
        data=csv_buffer.getvalue(),
        file_name="kmc_answer_key.csv",
        mime="text/csv",
    )


def main():
    render_header()
    answer_key = fetch_answer_key()

    mode = st.sidebar.radio("Mode", ("OMR Checker", "Answer Key Admin"))
    st.sidebar.markdown(
        "**Default Admin Password:** `kmc-2081-admin`\n"
        "You can override via Streamlit secrets (`admin_password`)."
    )

    if mode == "OMR Checker":
        render_omr_checker(answer_key)
    else:
        render_admin_mode(answer_key)

    st.sidebar.markdown(
        "---\n"
        "**Troubleshooting Tips**\n"
        "- Ensure all four corners of the sheet are visible.\n"
        "- Use higher DPI or brighter lighting if bubbles are missed.\n"
        "- Recalibrate by editing the detected responses or updating the answer key."
    )


if __name__ == "__main__":
    import cv2  # Local import to satisfy Streamlit reruns

    main()
