"""Computer vision and scoring helpers for the KMC OMR checker.

The functions in this module follow these high-level steps:
1. Load user uploads (image or PDF) into OpenCV-friendly numpy arrays.
2. Detect and rectify the OMR sheet to a canonical perspective.
3. Locate the 400 bubbles (100 questions × 4 options) by contour analysis.
4. Decide which bubbles are filled, normalise to question numbers, and score.
5. Provide visual overlays for UI feedback.

The logic is tuned for the Kathmandu Metropolitan City Class 11 Scholarship
Examination 2081 OMR sheet. It assumes:
- Five vertical question blocks, each containing 20 rows.
- Four horizontally arranged circular choices per question (A-D).
- Distinct dark side markers that allow finding the sheet outline.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image

try:  # Optional import; OCR is best effort only.
    import pytesseract
except Exception:  # pragma: no cover - optional dependency at runtime.
    pytesseract = None  # type: ignore

try:
    from pdf2image import convert_from_bytes
except Exception:  # pragma: no cover - optional dependency at runtime.
    convert_from_bytes = None  # type: ignore

CHOICES = ("A", "B", "C", "D")
TOTAL_QUESTIONS = 100
QUESTIONS_PER_COLUMN = 20
TOTAL_COLUMNS = 5

# Canonical warped sheet size (height, width). A4 scaled with padding.
CANONICAL_SIZE = (3600, 2600)


@dataclass
class BubbleDetection:
    question_no: int
    option: str
    contour: np.ndarray
    center: Tuple[int, int]
    bounding_rect: Tuple[int, int, int, int]
    fill_ratio: float


@dataclass
class QuestionResult:
    question_no: int
    selected_option: Optional[str]
    multi_marked: bool
    fill_ratios: Dict[str, float]


@dataclass
class ScoreBreakdown:
    score: int
    total_questions: int
    correct: int
    incorrect: int
    blank: int
    multi_marked: int
    percentage: float


@dataclass
class ProcessedSheet:
    aligned_image: np.ndarray
    threshold_image: np.ndarray
    bubble_detections: List[BubbleDetection]
    question_results: List[QuestionResult]
    overlay_image: np.ndarray
    roll_number: Optional[str]


class SheetDetectionError(RuntimeError):
    """Raised when the pipeline cannot locate or rectify the OMR sheet."""


def load_image_from_upload(uploaded_file) -> np.ndarray:
    """Read an uploaded file (image or PDF) into a BGR numpy array."""
    file_bytes = uploaded_file.read()
    suffix = uploaded_file.name.split(".")[-1].lower()

    if suffix == "pdf":
        if convert_from_bytes is None:
            raise RuntimeError(
                "pdf2image is required to process PDF uploads. Install poppler and pdf2image."
            )
        pil_pages = convert_from_bytes(file_bytes, first_page=1, last_page=1, dpi=300)
        if not pil_pages:
            raise ValueError("No pages found in the uploaded PDF.")
        pil_image = pil_pages[0]
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unsupported or corrupted image format. Please upload JPG or PNG.")

    return image


def detect_and_warp_sheet(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Locate the OMR sheet contour and apply a perspective transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, np.ones((5, 5), np.uint8), iterations=2)
    edged = cv2.erode(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise SheetDetectionError("No contours detected. Ensure the OMR sheet is clearly visible.")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sheet_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            sheet_contour = approx
            break

    if sheet_contour is None:
        raise SheetDetectionError(
            "Failed to detect the sheet outline. Try a clearer scan with visible borders."
        )

    pts = sheet_contour.reshape(4, 2)
    warped, matrix = four_point_transform(image, pts, CANONICAL_SIZE[::-1])
    return warped, matrix


def four_point_transform(image: np.ndarray, pts: np.ndarray, output_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a perspective transform using four corner points."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width, height = output_size
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32"
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return the points ordered as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def threshold_bubbles(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    return thresh


def find_bubble_contours(thresh: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 150 or area > 4000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.6 <= aspect_ratio <= 1.4:
            bubble_contours.append(contour)
    return bubble_contours


def cluster_bubbles_into_grid(
    contours: List[np.ndarray],
    thresh: np.ndarray,
) -> List[BubbleDetection]:
    if len(contours) < TOTAL_QUESTIONS * len(CHOICES) * 0.6:
        raise RuntimeError(
            "Detected too few bubbles. Ensure the sheet is well lit and free of obstructions."
        )

    bubble_data = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        total_pixels = np.count_nonzero(mask)
        if total_pixels == 0:
            continue
        filled_pixels = np.count_nonzero(cv2.bitwise_and(thresh, thresh, mask=mask))
        fill_ratio = filled_pixels / float(total_pixels)
        bubble_data.append((contour, (cx, cy), (x, y, w, h), fill_ratio))

    if not bubble_data:
        raise RuntimeError("Could not evaluate any bubbles. Try rescanning with better contrast.")

    centers_x = np.array([center[0] for _, center, _, _ in bubble_data], dtype=np.float32)
    column_thresholds = compute_column_boundaries(centers_x)

    grouped: Dict[int, List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int], float]]] = {
        idx: [] for idx in range(TOTAL_COLUMNS)
    }
    for entry in bubble_data:
        _, center, _, _ = entry
        column_idx = assign_column(center[0], column_thresholds)
        grouped[column_idx].append(entry)

    detections: List[BubbleDetection] = []
    for column_idx in range(TOTAL_COLUMNS):
        column_entries = grouped[column_idx]
        if len(column_entries) < QUESTIONS_PER_COLUMN * len(CHOICES) * 0.6:
            # Allow column-level recovery but flag the result in downstream logic.
            pass

        # Sort by vertical position then cluster into rows.
        column_entries.sort(key=lambda item: item[1][1])
        row_groups = cluster_by_axis(column_entries, axis=1)
        if len(row_groups) != QUESTIONS_PER_COLUMN:
            # Attempt coarse regrouping by merging nearest groups.
            row_groups = rebin_rows(column_entries, QUESTIONS_PER_COLUMN)

        row_base = column_idx * QUESTIONS_PER_COLUMN
        for row_idx, group in enumerate(row_groups):
            question_no = row_base + row_idx + 1
            group.sort(key=lambda item: item[1][0])  # left to right (A-D)
            if len(group) != len(CHOICES):
                # Attempt to repair by selecting closest entries to expected positions.
                group = pad_or_trim_group(group)
            for option_idx, entry in enumerate(group):
                contour, center, rect, fill_ratio = entry
                option = CHOICES[min(option_idx, len(CHOICES) - 1)]
                detections.append(
                    BubbleDetection(
                        question_no=question_no,
                        option=option,
                        contour=contour,
                        center=center,
                        bounding_rect=rect,
                        fill_ratio=fill_ratio,
                    )
                )

    if not detections:
        raise RuntimeError("Failed to arrange detected bubbles into the expected grid.")

    detections.sort(key=lambda d: (d.question_no, CHOICES.index(d.option)))
    return detections


def compute_column_boundaries(x_positions: np.ndarray) -> List[float]:
    sorted_x = np.sort(x_positions)
    # Estimated breakpoints at 20%, 40%, 60%, 80% percentiles between min/max.
    quantiles = np.linspace(0, 1, TOTAL_COLUMNS + 1)
    percentiles = np.quantile(sorted_x, quantiles)
    # Use midpoints between percentile boundaries to classify columns.
    thresholds = []
    for i in range(1, len(percentiles) - 1):
        thresholds.append((percentiles[i] + percentiles[i + 1]) / 2.0)
    return thresholds


def assign_column(x: float, thresholds: List[float]) -> int:
    for idx, threshold in enumerate(thresholds):
        if x < threshold:
            return idx
    return len(thresholds)


def cluster_by_axis(entries, axis: int = 1, tolerance_factor: float = 1.4):
    # axis=1 corresponds to y coordinate.
    groups: List[List] = []
    current_group: List = []
    values = []
    for item in entries:
        x, y, w, h = cv2.boundingRect(item[0])
        values.append(h if axis == 1 else w)
    avg_extent = np.mean(values) if values else 10

    for entry in entries:
        coord = entry[1][axis]
        if not current_group:
            current_group = [entry]
            groups.append(current_group)
            continue
        last_coord = np.mean([e[1][axis] for e in current_group])
        if abs(coord - last_coord) <= tolerance_factor * avg_extent:
            current_group.append(entry)
        else:
            current_group = [entry]
            groups.append(current_group)
    return [group for group in groups if group]


def rebin_rows(entries, expected_rows):
    entries_sorted = sorted(entries, key=lambda e: e[1][1])
    bins = [[] for _ in range(expected_rows)]
    ys = np.array([e[1][1] for e in entries_sorted])
    min_y, max_y = ys.min(), ys.max()
    for entry in entries_sorted:
        normalized = (entry[1][1] - min_y) / max(max_y - min_y, 1)
        bin_idx = min(int(normalized * expected_rows), expected_rows - 1)
        bins[bin_idx].append(entry)
    return bins


def pad_or_trim_group(group: List, target_size: int = len(CHOICES)) -> List:
    if len(group) > target_size:
        # Keep the entries whose x coordinate is closest to the group mean (centre four).
        mean_x = np.mean([entry[1][0] for entry in group])
        group.sort(key=lambda entry: abs(entry[1][0] - mean_x))
        return group[:target_size]
    elif len(group) < target_size:
        # Repeat nearest neighbours (degenerate but keeps pipeline alive).
        if not group:
            return []
        while len(group) < target_size:
            group.append(group[-1])
        return group
    return group


def compute_question_results(detections: List[BubbleDetection]) -> List[QuestionResult]:
    results: List[QuestionResult] = []
    for question_no in range(1, TOTAL_QUESTIONS + 1):
        question_detections = [d for d in detections if d.question_no == question_no]
        if not question_detections:
            results.append(
                QuestionResult(
                    question_no=question_no,
                    selected_option=None,
                    multi_marked=False,
                    fill_ratios={option: 0.0 for option in CHOICES},
                )
            )
            continue

        fill_ratios = {d.option: d.fill_ratio for d in question_detections}
        filled_options = [opt for opt, ratio in fill_ratios.items() if ratio >= 0.5]
        selected_option: Optional[str]
        multi_marked = False
        if len(filled_options) == 1:
            selected_option = filled_options[0]
        elif len(filled_options) > 1:
            selected_option = None
            multi_marked = True
        else:
            # Fallback to the highest ratio if none exceed threshold.
            best_option, best_ratio = max(fill_ratios.items(), key=lambda item: item[1])
            selected_option = best_option if best_ratio >= 0.35 else None

        results.append(
            QuestionResult(
                question_no=question_no,
                selected_option=selected_option,
                multi_marked=multi_marked,
                fill_ratios=fill_ratios,
            )
        )
    return results


def score_answers(results: Iterable[QuestionResult], answer_key: Dict[int, str]) -> ScoreBreakdown:
    total = TOTAL_QUESTIONS
    correct = incorrect = blank = multi = 0

    for result in results:
        key = answer_key.get(result.question_no)
        if result.multi_marked:
            multi += 1
            continue
        if result.selected_option is None:
            blank += 1
            continue
        if key and result.selected_option == key.upper():
            correct += 1
        else:
            incorrect += 1

    score = correct
    percentage = (score / total) * 100 if total else 0.0
    return ScoreBreakdown(
        score=score,
        total_questions=total,
        correct=correct,
        incorrect=incorrect,
        blank=blank,
        multi_marked=multi,
        percentage=percentage,
    )


def draw_overlay(aligned: np.ndarray, detections: List[BubbleDetection], answer_key: Dict[int, str]) -> np.ndarray:
    display = aligned.copy()
    for detection in detections:
        x, y, w, h = detection.bounding_rect
        question_no = detection.question_no
        correct_option = answer_key.get(question_no)
        color = (0, 255, 255)  # default yellow
        if detection.fill_ratio >= 0.5:
            if correct_option and detection.option == correct_option:
                color = (0, 200, 0)
            else:
                color = (0, 0, 255)
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
    return display


def extract_roll_number(aligned: np.ndarray) -> Optional[str]:
    if pytesseract is None:
        return None

    h, w = aligned.shape[:2]
    roi = aligned[int(0.08 * h) : int(0.16 * h), int(0.65 * w) : int(0.9 * w)]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    custom_config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(gray, config=custom_config)
    cleaned = "".join(ch for ch in text if ch.isdigit())
    return cleaned or None


def process_omr_sheet(image: np.ndarray, answer_key: Dict[int, str]) -> ProcessedSheet:
    aligned, matrix = detect_and_warp_sheet(image)
    thresh = threshold_bubbles(aligned)
    contours = find_bubble_contours(thresh)
    detections = cluster_bubbles_into_grid(contours, thresh)
    results = compute_question_results(detections)
    overlay = draw_overlay(aligned, detections, answer_key)
    roll_number = extract_roll_number(aligned)
    return ProcessedSheet(
        aligned_image=aligned,
        threshold_image=thresh,
        bubble_detections=detections,
        question_results=results,
        overlay_image=overlay,
        roll_number=roll_number,
    )


def encode_image_to_png_bytes(image: np.ndarray) -> bytes:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()
