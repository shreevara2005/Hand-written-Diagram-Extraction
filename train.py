
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'images_dir': r'E:\python\Flow_chart_Detection\flowchart_dataset\images', 
    'train_json': r'E:\python\Flow_chart_Detection\flowchart_dataset\annotations\train.json',  
    'val_json': r'E:\python\Flow_chart_Detection\flowchart_dataset\annotations\val.json', 
    'num_classes': 21, 
    'batch_size': 4,   
    'num_epochs': 30,
    'learning_rate': 0.001,
    
 
    'save_dir': 'checkpoints', 
    'log_dir': 'logs',          
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4, 
}
class FlowchartDataset(Dataset):
    """Custom Dataset for Flowchart Detection (COCO Format)"""
    
    def __init__(self, images_dir, annotation_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.image_ids = [img['id'] for img in self.coco_data['images']]
        self.id_to_image = {img['id']: img for img in self.coco_data['images']}
 
        self.image_to_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = []
            self.image_to_annotations[img_id].append(ann)

        self.categories = {cat['id']: cat['name'] 
                          for cat in self.coco_data['categories']}
        
        print(f"✓ Loaded {len(self.image_ids)} images")
        print(f"✓ Categories: {list(self.categories.values())}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.id_to_image[image_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        if not os.path.exists(img_path): 
            filename_only = os.path.basename(img_info['file_name'])
            img_path = os.path.join(self.images_dir, filename_only)
            if not os.path.exists(img_path):
                print(f"\n❌ Image not found: {img_info['file_name']}")
                print(f"   Tried: {os.path.join(self.images_dir, img_info['file_name'])}")
                print(f"   And:   {img_path}")
                raise FileNotFoundError(f"Image file not found: {img_info['file_name']}")
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image (corrupted?): {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotations = self.image_to_annotations.get(image_id, [])
        boxes = []
        labels = []
        areas = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        image_id_tensor = torch.tensor([image_id])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id_tensor,
            'area': areas,
            'iscrowd': iscrowd
        }
        if self.transforms:
            try:
                transformed = self.transforms(image=image, 
                                             bboxes=boxes.numpy(), 
                                             class_labels=labels.numpy())
                image = transformed['image']
                target['boxes'] = torch.as_tensor(transformed['bboxes'], 
                                                 dtype=torch.float32)
            except Exception as e:
                print(f"\n⚠️ Transform error on {img_path}: {e}")
                print(f"   Using image without transforms")
                image = T.ToTensor()(image)
        else:
            image = T.ToTensor()(image)
        
        return image, target

def get_train_transform():
    """Training transforms with augmentation"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Rotate(limit=15, p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.Resize(800, 800),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))

def get_valid_transform():
    """Validation transforms without augmentation"""
    return A.Compose([
        A.Resize(800, 800),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def create_model(num_classes):
    """Create Faster R-CNN model"""
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_components = {'loss_classifier': 0, 'loss_box_reg': 0, 
                      'loss_objectness': 0, 'loss_rpn_box_reg': 0}
    
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key].item()
        
        progress_bar.set_postfix({'loss': f'{losses.item():.4f}'})
    
    avg_loss = total_loss / len(data_loader)
    avg_components = {k: v / len(data_loader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components

def custom_collate_fn(batch):

    return tuple(zip(*batch))

def main():
   
    
    print("="*70)
    print("FLOWCHART DETECTION TRAINING - FASTER R-CNN")
    print("="*70)
    print(f"\n📍 Configuration:")
    print(f"   Images Directory: {CONFIG['images_dir']}")
    print(f"   Train JSON: {CONFIG['train_json']}")
    print(f"   Val JSON: {CONFIG['val_json']}")
    print(f"   Device: {CONFIG['device']}")
    print(f"   Batch Size: {CONFIG['batch_size']}")
    print(f"   Epochs: {CONFIG['num_epochs']}")
    print(f"   Learning Rate: {CONFIG['learning_rate']}")
    print("="*70 + "\n")
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    print("📂 Loading datasets...")
    train_dataset = FlowchartDataset(
        CONFIG['images_dir'],
        CONFIG['train_json'],
        transforms=get_train_transform()
    )
    
    val_dataset = FlowchartDataset(
        CONFIG['images_dir'],
        CONFIG['val_json'],
        transforms=get_valid_transform()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        collate_fn=custom_collate_fn,
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        collate_fn=custom_collate_fn,
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    print(f"\n✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}\n")
    print("🔨 Creating model...")
    model = create_model(CONFIG['num_classes'])
    model.to(CONFIG['device'])
    print("✓ Model created and moved to device\n")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=CONFIG['learning_rate'],
        momentum=0.9, 
        weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=15, 
        gamma=0.1
    )
    history = {
        'train_loss': [],
        'train_loss_classifier': [],
        'train_loss_box_reg': [],
        'train_loss_objectness': [],
        'train_loss_rpn_box_reg': [],
        'learning_rate': []
    }
    
    best_loss = float('inf')
    print("🚀 Starting training...\n")
    print("="*70)
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\n📅 EPOCH {epoch}/{CONFIG['num_epochs']}")
        print("-"*70)
        train_loss, loss_components = train_one_epoch(
            model, optimizer, train_loader, CONFIG['device'], epoch
        )
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['train_loss_classifier'].append(loss_components['loss_classifier'])
        history['train_loss_box_reg'].append(loss_components['loss_box_reg'])
        history['train_loss_objectness'].append(loss_components['loss_objectness'])
        history['train_loss_rpn_box_reg'].append(loss_components['loss_rpn_box_reg'])
        history['learning_rate'].append(current_lr)
        print(f"\n📊 Epoch {epoch} Results:")
        print(f"   Total Loss:      {train_loss:.4f}")
        print(f"   Classifier Loss: {loss_components['loss_classifier']:.4f}")
        print(f"   Box Reg Loss:    {loss_components['loss_box_reg']:.4f}")
        print(f"   Objectness Loss: {loss_components['loss_objectness']:.4f}")
        print(f"   RPN Box Loss:    {loss_components['loss_rpn_box_reg']:.4f}")
        print(f"   Learning Rate:   {current_lr:.6f}")
        if train_loss < best_loss:
            best_loss = train_loss
            save_path = os.path.join(CONFIG['save_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'loss_components': loss_components
            }, save_path)
            print(f"   ✓ Saved best model (Loss: {train_loss:.4f})")
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(CONFIG['save_dir'], f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"   ✓ Saved checkpoint at epoch {epoch}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print(f"   Best Loss: {best_loss:.4f}")
    print(f"   Model saved to: {CONFIG['save_dir']}/best_model.pth")
    print("="*70 + "\n")
    plot_training_history(history, CONFIG['log_dir'])
    
    return model, history

def plot_training_history(history, save_dir):
    """Plot and save training metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(history['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Total Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(history['train_loss_classifier'], label='Classifier', linewidth=2)
    axes[0, 1].plot(history['train_loss_box_reg'], label='Box Reg', linewidth=2)
    axes[0, 1].set_title('Classification & Box Regression Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(history['train_loss_objectness'], label='Objectness', linewidth=2)
    axes[1, 0].plot(history['train_loss_rpn_box_reg'], label='RPN Box', linewidth=2)
    axes[1, 0].set_title('RPN Losses', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(history['learning_rate'], 'r-', linewidth=2)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training history plot saved to: {save_path}")
    plt.close()
if __name__ == "__main__":
    try:
        model, history = main()
        print("\n✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()