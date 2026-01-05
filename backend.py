
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import base64
from PIL import Image
import io
import numpy as np
import cv2
import logging

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torchvision.ops import nms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Flowchart Recognition API", version="4.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = r'E:\python\Flow_chart_Detection\checkpoints\best_model.pth'
NUM_CLASSES = 21  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
TOTAL_MARKS = 40

CATEGORY_NAMES = {
    0: 'Flow-chart',
    1: 'action',
    2: 'activity',
    3: 'comment',
    4: 'control_flow',
    5: 'control_flowcontrol_flow',
    6: 'decision_node',
    7: 'exit_node',
    8: 'final_flow_node',
    9: 'final_node',
    10: 'fork',
    11: 'merge',
    12: 'merge_node',
    13: 'null',
    14: 'object',
    15: 'object_flow',
    16: 'signal_recept',
    17: 'signal_send',
    18: 'start_node',
    19: 'text'
}
CLASS_COLORS = {
    0: (128, 128, 128), 1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255),
    4: (255, 255, 0), 5: (255, 0, 255), 6: (0, 255, 255), 7: (128, 0, 128),
    8: (0, 128, 128), 9: (128, 128, 0), 10: (255, 128, 0), 11: (0, 128, 255),
    12: (128, 255, 0), 13: (255, 0, 128), 14: (0, 255, 128), 15: (128, 0, 255),
    16: (255, 128, 128), 17: (128, 255, 128), 18: (128, 128, 255), 19: (255, 255, 128),
}

MODEL = None

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
def load_model(model_path, num_classes, device):
    """EXACT same as predict.py"""
    logger.info(f"🔨 Loading model from: {model_path}")
    
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"✅ Model loaded on {device}")
    if 'epoch' in checkpoint:
        logger.info(f"   Epochs: {checkpoint['epoch']}")
    
    return model
@torch.no_grad()
def predict_single_image(model, image_pil, device):

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
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    if len(boxes) > 0:
        keep_indices = nms(boxes, scores, NMS_THRESHOLD)
        boxes = boxes[keep_indices].numpy()
        labels = labels[keep_indices].numpy()
        scores = scores[keep_indices].numpy()
    else:
        boxes = boxes.numpy()
        labels = labels.numpy()
        scores = scores.numpy()

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        class_id = int(label)
        detections.append({
            'bbox': box.tolist(),
            'class_id': class_id,
            'class_name': CATEGORY_NAMES.get(class_id, f'Class_{class_id}'),
            'confidence': float(score),
            'image_bgr': image_bgr
        })
    
    logger.info(f"✅ Detected {len(detections)} objects")
    
    class_counts = {}
    for d in detections:
        name = d['class_name']
        class_counts[name] = class_counts.get(name, 0) + 1
    logger.info(f"   Detection summary: {class_counts}")
    
    return {
        'detections': detections,
        'image_bgr': image_bgr,
        'boxes': boxes,
        'labels': labels,
        'scores': scores
    }

def count_by_type(detections, target_types):
    if isinstance(target_types, str):
        target_types = [target_types]
    return len([d for d in detections if d.get('class_name') in target_types])

def get_grade(score):
    percentage = (score / TOTAL_MARKS) * 100
    if percentage >= 90: return 'A+ (Excellent)'
    elif percentage >= 80: return 'A (Very Good)'
    elif percentage >= 70: return 'B+ (Good)'
    elif percentage >= 60: return 'B (Satisfactory)'
    elif percentage >= 50: return 'C (Pass)'
    else: return 'F (Needs Improvement)'

def evaluate_required_elements(detections):
    """Required Elements (10 marks) - IMPROVED SCORING"""
    score, errors = 0, []

    start = count_by_type(detections, ['start_node', 'action']) 
    if start >= 1: 
        score += 3
    else: 
        errors.append("❌ Missing start node")
    
    final = count_by_type(detections, ['final_node', 'final_flow_node', 'exit_node'])
    if final >= 1: 
        score += 3
    else: 
      
        actions = count_by_type(detections, ['action', 'activity'])
        if actions >= 2:
            score += 2 
            errors.append("⚠️ Missing explicit final node (partial credit)")
        else:
            errors.append("❌ Missing final node")

    actions = count_by_type(detections, ['action', 'activity', 'signal_recept'])
    if actions >= 5: score += 4
    elif actions == 4: score += 4
    elif actions == 3: score += 3.5
    elif actions == 2: score += 3
    elif actions == 1: score += 2
    else: errors.append("❌ No actions")
    
    return int(round(score)), errors

def evaluate_flow_connectivity(detections):
    """Flow Connectivity (10 marks) - IMPROVED SCORING"""
    score, errors = 0, []
    
    nodes = count_by_type(detections, ['action', 'activity', 'decision_node', 
                                       'fork', 'merge', 'merge_node', 
                                       'start_node', 'final_node', 'exit_node',
                                       'signal_recept'])
    flows = count_by_type(detections, ['control_flow', 'control_flowcontrol_flow'])
 
    if nodes == 0: 
        score += 5
    elif flows >= nodes * 1.5: 
        score += 5
    elif flows >= nodes * 1.2: 
        score += 5
    elif flows >= nodes * 1.0: 
        score += 4.5
    elif flows >= nodes * 0.8: 
        score += 4
    elif flows >= nodes * 0.6: 
        score += 3
    elif flows > 0:
        score += 2
    else:
        errors.append("❌ No control flows")
  
    if flows > 0 and nodes > 0:
        ratio = flows / nodes
        if ratio >= 1.2: score += 3
        elif ratio >= 1.0: score += 3
        elif ratio >= 0.8: score += 2.5
        elif ratio >= 0.6: score += 2
        else: score += 1
    
    decisions = count_by_type(detections, 'decision_node')
    if decisions == 0:
        score += 2  
    elif decisions > 0 and flows >= decisions * 1.5:
        score += 2
    else:
        score += 1
    
    return int(round(score)), errors

def evaluate_concurrent_control(detections):
   
    score, errors = 0, []
    
    forks = count_by_type(detections, 'fork')
    merges = count_by_type(detections, ['merge', 'merge_node'])
    flows = count_by_type(detections, ['control_flow', 'control_flowcontrol_flow'])
    

    if forks == 0 and merges == 0: 
        score += 3  
    elif forks == merges: 
        score += 3
    elif abs(forks - merges) <= 1: 
        score += 2.5
    else: 
        score += 2
    
 
    if (forks + merges) == 0: 
        score += 2
    elif flows >= (forks + merges) * 1.5: 
        score += 2
    else: 
        score += 1.5
    
    total = len(detections)
    if total >= 15: score += 3     
    elif total >= 12: score += 3  
    elif total >= 10: score += 2.5 
    elif total >= 7: score += 2     
    elif total >= 5: score += 1.5
    else: score += 1
    
    return int(round(score)), errors

def evaluate_signal_object_handling(detections):

    score = 6  
    errors = []
    
    objects = count_by_type(detections, 'object')
    obj_flows = count_by_type(detections, 'object_flow')
    
    if objects > 0 and obj_flows == 0:
        score -= 0.5  
        errors.append("⚠️ Objects without object flows (minor)")
    
    return int(round(score)), errors

def evaluate_documentation(detections):
    score, errors = 0, []
    
    actions = [d for d in detections if d['class_name'] in ['action', 'activity', 'signal_recept']]
    texts = count_by_type(detections, 'text')
    comments = count_by_type(detections, 'comment')

    if not actions: 
        score += 4
    elif texts >= 5: 
        score += 4
    elif texts >= 3:  
        score += 4
    elif texts >= 2: 
        score += 3.5
    elif texts >= 1: 
        score += 3
    else:
        score += 2.5  
    
    if comments >= 1: 
        score += 2
    else: 
        score += 1.5 
    
    return int(round(score)), errors

def calculate_score(detections):
    r_score, r_err = evaluate_required_elements(detections)
    f_score, f_err = evaluate_flow_connectivity(detections)
    c_score, c_err = evaluate_concurrent_control(detections)
    s_score, s_err = evaluate_signal_object_handling(detections)
    d_score, d_err = evaluate_documentation(detections)
    
    total = r_score + f_score + c_score + s_score + d_score
   
    total = min(total, TOTAL_MARKS)
    
    breakdown = {
        'required_elements': r_score,
        'flow_connectivity': f_score,
        'concurrent_control': c_score,
        'signal_object_handling': s_score,
        'documentation': d_score
    }
    
    errors = r_err + f_err + c_err + s_err + d_err
    
    summary = {
        'total_marks': total,
        'max_marks': TOTAL_MARKS,
        'percentage': round((total / TOTAL_MARKS) * 100, 1),
        'grade': get_grade(total)
    }
    
    logger.info(f"   Evaluation breakdown: {breakdown}")
    logger.info(f"   Total: {total}/40 ({summary['percentage']}%)")
    
    return total, breakdown, errors, summary

def visualize_predictions(image_bgr, boxes, labels, scores, summary):
    """EXACT visualization from predict.py"""
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

def analyze_flowchart(image_pil):
    
    logger.info("🔍 Starting analysis...")

    results = predict_single_image(MODEL, image_pil, DEVICE)
    detections = results['detections']
    
    total, breakdown, errors, summary = calculate_score(detections)
    
    annotated = visualize_predictions(
        results['image_bgr'],
        results['boxes'],
        results['labels'],
        results['scores'],
        summary
    )
    
    logger.info(f"✅ Complete: {total}/40 ({summary['percentage']}%) - {summary['grade']}")
    
    return total, breakdown, errors, summary, annotated

@app.on_event("startup")
async def startup():
    global MODEL
    try:
        MODEL = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)
        logger.info("✅ Ready to serve")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")

@app.get("/")
async def root():
    return {
        "status": "running",
        "model_loaded": MODEL is not None,
        "device": DEVICE,
        "version": "4.1.0 - INTEGER MARKS + IMPROVED SCORING"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if MODEL else "no_model",
        "model_path": MODEL_PATH,
        "confidence": CONFIDENCE_THRESHOLD
    }

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    
    logger.info(f"📥 {file.filename}")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode not in ['RGB', 'RGBA']:
            image = image.convert('RGB')

        total, breakdown, errors, summary, annotated = analyze_flowchart(image)
 
        _, buffer = cv2.imencode('.png', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "marks": total,
            "breakdown": breakdown,
            "errors": errors,
            "summary": summary,
            "annotated_image": img_base64
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("="*70)
    logger.info("🚀 FINAL INTEGRATED BACKEND v4.1")
    logger.info("   INTEGER MARKS + IMPROVED SCORING")
    logger.info("="*70)
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Device: {DEVICE}")
    logger.info("="*70)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")