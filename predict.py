
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from torchvision.ops import nms
import argparse
import warnings
warnings.filterwarnings('ignore')


INFERENCE_CONFIG = {
    'model_path': r'E:\python\Flow_chart_Detection\checkpoints\best_model.pth',  
    'image_path': r"E:\python\Flow_chart_Detection\test_image\evaluation_image\student1_q1.jpg",  
    'images_folder': '', 
    'num_classes': 21,  +
    'confidence_threshold': 0.5,  
    'nms_threshold': 0.5,
    'output_dir': r'E:\python\Flow_chart_Detection',  
    'save_json': True, 
    'show_plot': True,  
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

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
    0: (128, 128, 128),  
    1: (0, 255, 0),      
    2: (255, 0, 0),      
    3: (0, 0, 255),      
    4: (255, 255, 0),    
    5: (255, 0, 255),    
    6: (0, 255, 255),    
    7: (128, 0, 128),    
    8: (0, 128, 128),    
    9: (128, 128, 0),    
    10: (255, 128, 0),   
    11: (0, 128, 255),   
    12: (128, 255, 0),   
    13: (255, 0, 128),   
    14: (0, 255, 128),   
    15: (128, 0, 255),   
    16: (255, 128, 128), 
    17: (128, 255, 128), 
    18: (128, 128, 255), 
    19: (255, 255, 128), 
}

def load_model(model_path, num_classes, device):
    """Load trained Faster R-CNN model"""
    
    print(f"🔨 Loading model from: {model_path}")
    
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"✓ Trained for {checkpoint['epoch']} epochs")
    if 'train_loss' in checkpoint:
        print(f"✓ Training loss: {checkpoint['train_loss']:.4f}")
    
    return model


@torch.no_grad()
def predict_single_image(model, image_path, device, confidence_threshold, nms_threshold):
    """
    Run inference on a single image
    
    Returns:
        dict with predictions and original image
    """
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image_rgb).to(device)
    
    predictions = model([image_tensor])[0]
    
    boxes = predictions['boxes'].cpu()
    labels = predictions['labels'].cpu()
    scores = predictions['scores'].cpu()
    
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    if len(boxes) > 0:
        keep_indices = nms(boxes, scores, nms_threshold)
        boxes = boxes[keep_indices].numpy()
        labels = labels[keep_indices].numpy()
        scores = scores[keep_indices].numpy()
    else:
        boxes = boxes.numpy()
        labels = labels.numpy()
        scores = scores.numpy()
    
    return {
        'boxes': boxes,
        'labels': labels,
        'scores': scores,
        'image': original_image,
        'image_path': image_path
    }


def visualize_predictions(image, boxes, labels, scores, save_path=None):
    output_image = image.copy()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        class_name = CATEGORY_NAMES.get(label, f'Class_{label}')
        color = CLASS_COLORS.get(label, (255, 255, 255))
        
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        
        label_text = f'{class_name}: {score:.2f}'
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            output_image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
       
        cv2.putText(
            output_image,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    if save_path:
        success = cv2.imwrite(save_path, output_image)
        if success:
            print(f"✓ Saved visualization: {save_path}")
        else:
            print(f"✗ Failed to save: {save_path}")
    
    return output_image

def display_results(image, boxes, labels, scores):
   
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f'Detected {len(scores)} objects', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def save_predictions_json(predictions, save_path):
    """Save predictions to JSON file"""
    
    json_data = {
        'image_path': predictions['image_path'],
        'num_detections': len(predictions['scores']),
        'detections': []
    }
    
    for box, label, score in zip(predictions['boxes'], 
                                  predictions['labels'], 
                                  predictions['scores']):
        json_data['detections'].append({
            'class_id': int(label),
            'class_name': CATEGORY_NAMES.get(label, f'Class_{label}'),
            'confidence': float(score),
            'bbox': [float(x) for x in box]  
        })
    
    with open(save_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✓ Saved predictions JSON: {save_path}")

def predict_single(model, image_path, device, output_dir):
    """Predict on a single image"""
    
    print(f"\n{'='*70}")
    print(f"Processing: {image_path}")
    print('='*70)
    
    results = predict_single_image(
        model, image_path, device,
        INFERENCE_CONFIG['confidence_threshold'],
        INFERENCE_CONFIG['nms_threshold']
    )
    
    print(f"\n✓ Detected {len(results['scores'])} objects:")
    for i, (label, score) in enumerate(zip(results['labels'], results['scores']), 1):
        class_name = CATEGORY_NAMES.get(label, f'Class_{label}')
        print(f"  {i}. {class_name}: {score:.3f}")
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_path = os.path.join(output_dir, f'{base_name}_predicted.jpg')
    output_json_path = os.path.join(output_dir, f'{base_name}_predictions.json')
    
    output_image = visualize_predictions(
        results['image'],
        results['boxes'],
        results['labels'],
        results['scores'],
        save_path=output_image_path
    )
    
    if INFERENCE_CONFIG['save_json']:
        save_predictions_json(results, output_json_path)
    
    if INFERENCE_CONFIG['show_plot']:
        display_results(output_image, results['boxes'], 
                       results['labels'], results['scores'])
    
    return results

def predict_batch(model, images_folder, device, output_dir):
    """Predict on multiple images in a folder"""
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(images_folder) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"❌ No images found in {images_folder}")
        return
    
    print(f"\n{'='*70}")
    print(f"BATCH PREDICTION")
    print(f"Found {len(image_files)} images")
    print('='*70)
    
    all_results = []
    
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(images_folder, img_file)
        
        print(f"\n[{i}/{len(image_files)}] Processing: {img_file}")
        
        try:
            results = predict_single_image(
                model, img_path, device,
                INFERENCE_CONFIG['confidence_threshold'],
                INFERENCE_CONFIG['nms_threshold']
            )
            
            print(f"   ✓ Detected {len(results['scores'])} objects")
            
            base_name = os.path.splitext(img_file)[0]
            output_image_path = os.path.join(output_dir, f'{base_name}_predicted.jpg')
            output_json_path = os.path.join(output_dir, f'{base_name}_predictions.json')
            
            visualize_predictions(
                results['image'],
                results['boxes'],
                results['labels'],
                results['scores'],
                save_path=output_image_path
            )
            
            if INFERENCE_CONFIG['save_json']:
                save_predictions_json(results, output_json_path)
            
            all_results.append(results)
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"✅ Batch prediction complete!")
    print(f"   Processed: {len(all_results)}/{len(image_files)} images")
    print(f"   Results saved to: {output_dir}")
    print('='*70)
    
    return all_results


def main():
    """Main function"""
    INFERENCE_CONFIG['output_dir'] = r'E:\python\Flow_chart_Detection\MY_RESULTS'
    print("="*70)
    print("FLOWCHART DETECTION - INFERENCE")
    print("="*70)
    print(f"\n📍 Configuration:")
    print(f"   Model: {INFERENCE_CONFIG['model_path']}")
    print(f"   Device: {INFERENCE_CONFIG['device']}")
    print(f"   Confidence Threshold: {INFERENCE_CONFIG['confidence_threshold']}")
    print(f"   NMS Threshold: {INFERENCE_CONFIG['nms_threshold']}")
    print(f"   Output Directory: {INFERENCE_CONFIG['output_dir']}")
    print("="*70)
    
    os.makedirs(INFERENCE_CONFIG['output_dir'], exist_ok=True)
    
    model = load_model(
        INFERENCE_CONFIG['model_path'],
        INFERENCE_CONFIG['num_classes'],
        INFERENCE_CONFIG['device']
    )
    
    if os.path.isfile(INFERENCE_CONFIG['image_path']):
        results = predict_single(
            model,
            INFERENCE_CONFIG['image_path'],
            INFERENCE_CONFIG['device'],
            INFERENCE_CONFIG['output_dir']
        )
    elif os.path.isdir(INFERENCE_CONFIG['images_folder']):
        results = predict_batch(
            model,
            INFERENCE_CONFIG['images_folder'],
            INFERENCE_CONFIG['device'],
            INFERENCE_CONFIG['output_dir']
        )
    else:
        print(f"\n❌ Error: Neither single image nor folder found!")
        print(f"   Check: {INFERENCE_CONFIG['image_path']}")
        print(f"   Or:    {INFERENCE_CONFIG['images_folder']}")
        return
    
    print(f"\n✅ Inference completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flowchart Detection Inference')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--output', type=str, default='predictions', 
                       help='Output directory (default: predictions)')
    parser.add_argument('--show', action='store_true', 
                       help='Display results (single image only)')
    
    args = parser.parse_args()
    
    if args.image:
        INFERENCE_CONFIG['image_path'] = args.image
    if args.folder:
        INFERENCE_CONFIG['images_folder'] = args.folder
    if args.model:
        INFERENCE_CONFIG['model_path'] = args.model
    if args.confidence:
        INFERENCE_CONFIG['confidence_threshold'] = args.confidence
    if args.output:
        INFERENCE_CONFIG['output_dir'] = args.output
    if args.show:
        INFERENCE_CONFIG['show_plot'] = True
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Inference interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()