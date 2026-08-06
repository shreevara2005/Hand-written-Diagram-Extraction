"""
LLM-ENHANCED BACKEND - Faster R-CNN + GPT-4 Vision
File: backend.py

Combines:
1. Faster R-CNN → Detects shapes/structure (from predict.py)
2. GPT-4 Vision → Understands flowchart LOGIC and FLOW

Install: pip install openai
Set your API key as environment variable: OPENAI_API_KEY
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import base64
from PIL import Image
import io
import numpy as np
import cv2
import logging
import json
import os

# PyTorch imports
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torchvision.ops import nms

# OpenAI import
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Flowchart Recognition + LLM API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CONFIGURATION ====================
MODEL_PATH = r'E:\python\Flow_chart_Detection\checkpoints\best_model.pth'
NUM_CLASSES = 21
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
TOTAL_MARKS = 40

# ⚠️ SET YOUR OPENAI API KEY HERE OR AS ENVIRONMENT VARIABLE
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-api-key-here")
LLM_MODEL = "gpt-4o"  # gpt-4o has vision built in and is cheaper than gpt-4-vision-preview

# Marks split: Structure (Faster R-CNN) vs Logic (LLM)
STRUCTURE_MARKS = 25   # From Faster R-CNN detection
LOGIC_MARKS = 15       # From LLM analysis

CATEGORY_NAMES = {
    0: 'Flow-chart', 1: 'action', 2: 'activity', 3: 'comment',
    4: 'control_flow', 5: 'control_flowcontrol_flow', 6: 'decision_node',
    7: 'exit_node', 8: 'final_flow_node', 9: 'final_node', 10: 'fork',
    11: 'merge', 12: 'merge_node', 13: 'null', 14: 'object',
    15: 'object_flow', 16: 'signal_recept', 17: 'signal_send',
    18: 'start_node', 19: 'text'
}

CLASS_COLORS = {
    0: (128, 128, 128), 1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255),
    4: (255, 255, 0), 5: (255, 0, 255), 6: (0, 255, 255), 7: (128, 0, 128),
    8: (0, 128, 128), 9: (128, 128, 0), 10: (255, 128, 0), 11: (0, 128, 255),
    12: (128, 255, 0), 13: (255, 0, 128), 14: (0, 255, 128), 15: (128, 0, 255),
    16: (255, 128, 128), 17: (128, 255, 128), 18: (128, 128, 255), 19: (255, 255, 128),
}

MODEL = None
openai_client = None

# ==================== Response Models ====================
class EvaluationSummary(BaseModel):
    total_marks: int
    max_marks: int
    percentage: float
    grade: str

class EvaluationResponse(BaseModel):
    marks: int
    breakdown: Dict[str, int]
    errors: List[str]
    summary: EvaluationSummary
    annotated_image: str
    llm_feedback: str          # NEW: Natural language feedback from LLM
    llm_logic_score: int       # NEW: Logic correctness score

# ==================== MODEL LOADING ====================

def load_model(model_path, num_classes, device):
    logger.info(f"🔨 Loading Faster R-CNN from: {model_path}")
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logger.info(f"✅ Faster R-CNN loaded on {device}")
    return model

# ==================== DETECTION (Faster R-CNN) ====================

@torch.no_grad()
def predict_single_image(model, image_pil, device):
    """Same detection logic as predict.py"""
    image_array = np.array(image_pil.convert('RGB'))
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image_rgb).to(device)

    predictions = model([image_tensor])[0]

    boxes = predictions['boxes'].cpu()
    labels = predictions['labels'].cpu()
    scores = predictions['scores'].cpu()

    mask = scores >= CONFIDENCE_THRESHOLD
    boxes, labels, scores = boxes[mask], labels[mask], scores[mask]

    if len(boxes) > 0:
        keep = nms(boxes, scores, NMS_THRESHOLD)
        boxes, labels, scores = boxes[keep].numpy(), labels[keep].numpy(), scores[keep].numpy()
    else:
        boxes, labels, scores = boxes.numpy(), labels.numpy(), scores.numpy()

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        class_id = int(label)
        detections.append({
            'bbox': box.tolist(),
            'class_id': class_id,
            'class_name': CATEGORY_NAMES.get(class_id, f'Class_{class_id}'),
            'confidence': float(score)
        })

    logger.info(f"✅ Faster R-CNN detected {len(detections)} objects")
    return {
        'detections': detections, 'image_bgr': image_bgr,
        'boxes': boxes, 'labels': labels, 'scores': scores
    }

# ==================== STRUCTURE EVALUATION (25 marks) ====================

def count_by_type(detections, target_types):
    if isinstance(target_types, str):
        target_types = [target_types]
    return len([d for d in detections if d.get('class_name') in target_types])

def evaluate_structure(detections):
    """
    Structure-only evaluation using Faster R-CNN detections (25 marks)
    Checks: Required elements, flow connectivity, concurrent control
    """
    score, errors = 0, []

    # Required elements (10 marks)
    start = count_by_type(detections, ['start_node'])
    final = count_by_type(detections, ['final_node', 'final_flow_node', 'exit_node'])
    actions = count_by_type(detections, ['action', 'activity'])

    if start >= 1: score += 3
    else: errors.append("❌ Missing start node")
    if final >= 1: score += 3
    else: errors.append("❌ Missing final node")
    if actions >= 3: score += 4
    elif actions >= 1: score += 2
    else: errors.append("❌ No action/process nodes")

    # Flow connectivity (10 marks)
    nodes = count_by_type(detections, ['action', 'activity', 'decision_node',
                                        'fork', 'merge', 'merge_node',
                                        'start_node', 'final_node', 'exit_node'])
    flows = count_by_type(detections, ['control_flow', 'control_flowcontrol_flow'])

    if nodes == 0:
        score += 10
    else:
        ratio = flows / nodes
        if ratio >= 1.0: score += 10
        elif ratio >= 0.7: score += 7
        elif ratio >= 0.5: score += 5
        else:
            score += 2
            errors.append(f"⚠️ Weak connectivity: {flows} flows for {nodes} nodes")

    # Concurrent control / complexity (5 marks)
    total = len(detections)
    if total >= 10: score += 5
    elif total >= 6: score += 3
    else: score += 2

    return min(STRUCTURE_MARKS, int(round(score))), errors

# ==================== LLM LOGIC EVALUATION ====================

def image_to_base64(image_pil):
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def evaluate_logic_with_llm(image_pil, detections, problem_statement=None):
    """
    Use GPT-4 Vision to understand the flowchart's LOGIC and FLOW.
    Reads the image directly (handles handwriting better than OCR)
    and checks if the sequence of steps makes logical sense.
    """
    global openai_client

    if openai_client is None:
        logger.warning("⚠️ OpenAI client not initialized, skipping LLM evaluation")
        return LOGIC_MARKS, "LLM evaluation unavailable (no API key configured).", []

    # Summarize detected structure to help ground the LLM
    detected_summary = {}
    for d in detections:
        name = d['class_name']
        detected_summary[name] = detected_summary.get(name, 0) + 1

    img_b64 = image_to_base64(image_pil)

    problem_context = f"\nThe flowchart is meant to solve this problem: \"{problem_statement}\"" if problem_statement else \
        "\nNo specific problem was given — infer the intended algorithm from the diagram itself."

    prompt = f"""You are an expert examiner grading a STUDENT'S HAND-DRAWN FLOWCHART image.

{problem_context}

A computer vision model detected these raw shape counts (may contain errors, use only as a hint):
{json.dumps(detected_summary, indent=2)}

Your job: Look at the actual image and evaluate the LOGICAL CORRECTNESS and FLOW of the algorithm — 
NOT just whether shapes exist, but whether the sequence of steps actually solves the stated problem correctly.

Evaluate on a 15-point scale across these criteria:
1. Logical Flow (6 pts): Do steps follow a sensible order? No dead ends or unreachable steps?
2. Algorithm Correctness (6 pts): Does the sequence of operations actually solve the problem correctly?
3. Decision Branch Accuracy (3 pts): Are decision conditions and their Yes/No (True/False) branches logically correct?

Respond ONLY in this exact JSON format, no extra text:
{{
  "logic_score": <integer 0-15>,
  "logical_flow_score": <integer 0-6>,
  "correctness_score": <integer 0-6>,
  "decision_score": <integer 0-3>,
  "feedback": "<2-4 sentences of clear, constructive feedback in plain language for the student>",
  "issues": ["<short issue 1>", "<short issue 2>", "..."]
}}"""

    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.2
        )

        raw_text = response.choices[0].message.content.strip()

        # Clean up markdown fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]

        result = json.loads(raw_text)

        logic_score = int(result.get("logic_score", 0))
        logic_score = max(0, min(LOGIC_MARKS, logic_score))
        feedback = result.get("feedback", "No feedback provided.")
        issues = result.get("issues", [])

        logger.info(f"✅ LLM logic evaluation: {logic_score}/{LOGIC_MARKS}")
        return logic_score, feedback, issues

    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse LLM response as JSON: {e}")
        return int(LOGIC_MARKS * 0.6), "Logic evaluation had a parsing issue; partial credit given.", []
    except Exception as e:
        logger.error(f"❌ LLM evaluation failed: {e}")
        return int(LOGIC_MARKS * 0.6), f"LLM evaluation unavailable: {str(e)}", []

# ==================== COMBINED SCORING ====================

def get_grade(score):
    pct = (score / TOTAL_MARKS) * 100
    if pct >= 90: return 'A+ (Excellent)'
    elif pct >= 80: return 'A (Very Good)'
    elif pct >= 70: return 'B+ (Good)'
    elif pct >= 60: return 'B (Satisfactory)'
    elif pct >= 50: return 'C (Pass)'
    else: return 'F (Needs Improvement)'

def calculate_combined_score(detections, image_pil, problem_statement=None):
    """Combines Faster R-CNN structure score + LLM logic score"""

    structure_score, structure_errors = evaluate_structure(detections)
    logic_score, llm_feedback, llm_issues = evaluate_logic_with_llm(
        image_pil, detections, problem_statement
    )

    total = structure_score + logic_score
    total = min(total, TOTAL_MARKS)

    breakdown = {
        'structure_required_flow': structure_score,   # out of 25
        'llm_logic_correctness': logic_score           # out of 15
    }

    all_errors = structure_errors + [f"🤖 {issue}" for issue in llm_issues]

    summary = {
        'total_marks': total,
        'max_marks': TOTAL_MARKS,
        'percentage': round((total / TOTAL_MARKS) * 100, 1),
        'grade': get_grade(total)
    }

    return total, breakdown, all_errors, summary, llm_feedback, logic_score

# ==================== VISUALIZATION ====================

def visualize_predictions(image_bgr, boxes, labels, scores, summary):
    output = image_bgr.copy()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(label)
        class_name = CATEGORY_NAMES.get(class_id, f'Class_{class_id}')
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label_text = f'{class_name}: {score:.2f}'
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output, (x1, y1 - th - baseline - 5), (x1 + tw, y1), color, -1)
        cv2.putText(output, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    banner_text = f"Score: {summary['total_marks']}/40 ({summary['percentage']}%) - {summary['grade']}"
    cv2.rectangle(output, (0, 0), (output.shape[1], 50), (0, 120, 215), -1)
    cv2.putText(output, banner_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return output

# ==================== MAIN PIPELINE ====================

def analyze_flowchart(image_pil, problem_statement=None):
    logger.info("🔍 Starting analysis (Faster R-CNN + LLM)...")

    results = predict_single_image(MODEL, image_pil, DEVICE)
    detections = results['detections']

    total, breakdown, errors, summary, llm_feedback, logic_score = calculate_combined_score(
        detections, image_pil, problem_statement
    )

    annotated = visualize_predictions(
        results['image_bgr'], results['boxes'], results['labels'], results['scores'], summary
    )

    logger.info(f"✅ Complete: {total}/40 ({summary['percentage']}%)")
    return total, breakdown, errors, summary, annotated, llm_feedback, logic_score

# ==================== API ENDPOINTS ====================

@app.on_event("startup")
async def startup():
    global MODEL, openai_client
    try:
        MODEL = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)
    except Exception as e:
        logger.error(f"❌ Faster R-CNN load failed: {e}")

    try:
        if OPENAI_API_KEY and OPENAI_API_KEY != "sk-your-api-key-here":
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("✅ OpenAI client initialized")
        else:
            logger.warning("⚠️ OPENAI_API_KEY not set. LLM logic evaluation will be skipped.")
    except Exception as e:
        logger.error(f"❌ OpenAI client init failed: {e}")

@app.get("/")
async def root():
    return {
        "status": "running",
        "model_loaded": MODEL is not None,
        "llm_available": openai_client is not None,
        "device": DEVICE,
        "version": "5.0.0 - Faster R-CNN + LLM"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if MODEL else "no_model",
        "llm_status": "ready" if openai_client else "not configured",
        "structure_marks": STRUCTURE_MARKS,
        "logic_marks": LOGIC_MARKS
    }

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    file: UploadFile = File(...),
    problem_statement: Optional[str] = Form(None)
):
    if MODEL is None:
        raise HTTPException(503, "Detection model not loaded")

    logger.info(f"📥 {file.filename} | Problem: {problem_statement}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode not in ['RGB', 'RGBA']:
            image = image.convert('RGB')

        total, breakdown, errors, summary, annotated, llm_feedback, logic_score = analyze_flowchart(
            image, problem_statement
        )

        _, buffer = cv2.imencode('.png', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "marks": total,
            "breakdown": breakdown,
            "errors": errors,
            "summary": summary,
            "annotated_image": img_base64,
            "llm_feedback": llm_feedback,
            "llm_logic_score": logic_score
        }

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("="*70)
    logger.info("🚀 BACKEND v5.0 — Faster R-CNN + LLM (GPT-4 Vision)")
    logger.info("="*70)
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"LLM Model: {LLM_MODEL}")
    logger.info(f"Marks: Structure={STRUCTURE_MARKS}, Logic={LOGIC_MARKS}, Total={TOTAL_MARKS}")
    logger.info("="*70)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
