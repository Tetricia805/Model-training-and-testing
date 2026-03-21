"""
Simple Model Testing Script - No UI Required
Test the model directly from command line or notebook
"""

import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from typing import Dict, Tuple
import json

# ============================================
# MODEL DEFINITION
# ============================================
class CattleMultiTaskModel(nn.Module):
    """MobileNetV3-Small with dual output heads"""
    
    def __init__(self, pretrained=False, freeze_backbone=False):
        super(CattleMultiTaskModel, self).__init__()
        
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        
        if pretrained:
            self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            self.backbone = mobilenet_v3_small(weights=None)
        
        self.feature_dim = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        
        self.id_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        self.diag_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        features = self.backbone(x)
        id_output = self.id_head(features)
        diag_output = self.diag_head(features)
        return id_output, diag_output

# ============================================
# MODEL TESTER CLASS
# ============================================
class CattleModelTester:
    """Simple interface for testing the cattle detection model"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the model tester
        
        Args:
            model_path: Path to the trained model file
            device: 'cuda', 'cpu', or 'auto' (auto-detect)
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.preprocessor = self._get_preprocessor()
        
        # Class names
        self.id_classes = {0: 'Cattle', 1: 'Not-Cattle'}
        self.diag_classes = {0: 'Healthy', 1: 'FMD (Infected)'}
    
    def _setup_device(self, device: str):
        """Setup compute device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        device = torch.device(device)
        print(f"✓ Using device: {device}")
        if device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        return device
    
    def _load_model(self, model_path: str):
        """Load the trained model"""
        print(f"\n📦 Loading model from: {model_path}")
        
        model = CattleMultiTaskModel(pretrained=False)
        loaded_state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(loaded_state)
        model = model.to(self.device)
        model.eval()
        
        print(f"✓ Model loaded successfully")
        return model
    
    def _get_preprocessor(self):
        """Get image preprocessing pipeline"""
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocess an image for model input
        
        Returns:
            Tuple of (tensor, PIL_image)
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Get original image info
        print(f"\n📸 Image Info:")
        print(f"   Size: {img.size}")
        print(f"   Mode: {img.mode}")
        
        # Preprocess
        tensor = self.preprocessor(img).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        return tensor, img
    
    def predict(self, image_path: str, return_raw: bool = False) -> Dict:
        """
        Run inference on an image
        
        Args:
            image_path: Path to the image file
            return_raw: If True, also return raw logits
        
        Returns:
            Dictionary with predictions and confidence scores
        """
        print(f"\n🔍 Running inference on: {Path(image_path).name}")
        
        # Preprocess
        tensor, pil_img = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            id_logits, diag_logits = self.model(tensor)
            
            id_probs = torch.softmax(id_logits, dim=1)
            
            # Get identification prediction
            id_pred = id_logits.argmax(1).item()
        
        result = {
            'identification': {
                'class_id': id_pred,
                'class_name': self.id_classes[id_pred],
                'confidence': float(id_probs[0, id_pred].item()),
                'probabilities': {
                    'cattle': float(id_probs[0, 0].item()),
                    'not_cattle': float(id_probs[0, 1].item())
                }
            },
            'diagnosis': None  # Will only be filled if cattle is detected
        }
        
        # ONLY run diagnosis if cattle is detected (class_id == 0)
        if id_pred == 0:  # Is cattle
            diag_probs = torch.softmax(diag_logits, dim=1)
            diag_pred = diag_logits.argmax(1).item()
            
            result['diagnosis'] = {
                'class_id': diag_pred,
                'class_name': self.diag_classes[diag_pred],
                'confidence': float(diag_probs[0, diag_pred].item()),
                'probabilities': {
                    'healthy': float(diag_probs[0, 0].item()),
                    'fmd': float(diag_probs[0, 1].item())
                }
            }
            
            if return_raw:
                result['raw_logits'] = {
                    'identification': id_logits[0].cpu().numpy(),
                    'diagnosis': diag_logits[0].cpu().numpy()
                }
        else:
            if return_raw:
                result['raw_logits'] = {
                    'identification': id_logits[0].cpu().numpy()
                }
        
        return result
    
    def generate_alert(self, result: Dict, confidence_threshold: float = 0.7) -> Dict:
        """Generate alert based on predictions"""
        
        id_conf = result['identification']['confidence']
        
        # FIRST CHECK: Is it cattle?
        if result['identification']['class_id'] == 1:  # Not cattle
            return {
                'level': 'NOT_CATTLE',
                'emoji': '❌',
                'message': '❌ This is NOT a cattle - Please upload a cattle image',
                'action': 'Upload an image containing CATTLE to diagnose FMD',
                'severity': 0
            }
        
        # SECOND CHECK: Is it confident cattle?
        elif id_conf < confidence_threshold:  # Unclear
            return {
                'level': 'UNCLEAR',
                'emoji': '❓',
                'message': f'❓ Unclear if this is cattle (confidence: {id_conf:.1%})',
                'action': 'Ask for better angle/lighting',
                'severity': 1
            }
        
        # THIRD CHECK: Cattle confirmed, now check diagnosis
        # Diagnosis is only available if cattle was detected
        elif result['diagnosis'] is not None:
            diag_conf = result['diagnosis']['confidence']
            
            if result['diagnosis']['class_id'] == 1:  # FMD detected
                if diag_conf > 0.85:
                    return {
                        'level': 'CRITICAL',
                        'emoji': '🚨',
                        'message': f'🚨 HIGH RISK: FMD DETECTED ({diag_conf:.1%} confidence)',
                        'action': 'ISOLATE HERD - Contact veterinarian immediately',
                        'severity': 3
                    }
                else:
                    return {
                        'level': 'WARNING',
                        'emoji': '⚠️',
                        'message': f'⚠️ CAUTION: Possible FMD ({diag_conf:.1%} confidence)',
                        'action': 'Contact veterinarian for confirmation',
                        'severity': 2
                    }
            
            else:  # Healthy
                return {
                    'level': 'OK',
                    'emoji': '✅',
                    'message': '✅ Cattle appears healthy',
                    'action': 'Continue monitoring (check again in 2 weeks)',
                    'severity': 0
                }
        
        # Fallback
        return {
            'level': 'UNKNOWN',
            'emoji': '❓',
            'message': 'Unable to determine status',
            'action': 'Try again with a clearer image',
            'severity': 1
        }
    
    def visualize_results(self, image_path: str, result: Dict, title: str = None):
        """Create visualization of results"""
        
        # Load image
        img = Image.open(image_path)
        
        # Create figure
        if result['diagnosis'] is not None:
            # 3-column layout (original, identification, diagnosis)
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(title or Path(image_path).name, fontsize=14, fontweight='bold')
            
            # Original image
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Identification probabilities
            id_probs = result['identification']['probabilities']
            axes[1].barh(['Cattle', 'Not-Cattle'], [id_probs['cattle'], id_probs['not_cattle']])
            axes[1].set_title(f"Identification\n({result['identification']['class_name']})")
            axes[1].set_xlim(0, 1)
            for i, (label, prob) in enumerate([('Cattle', id_probs['cattle']), ('Not-Cattle', id_probs['not_cattle'])]):
                axes[1].text(prob + 0.02, i, f'{prob:.1%}', va='center')
            
            # Diagnosis probabilities
            diag_probs = result['diagnosis']['probabilities']
            axes[2].barh(['Healthy', 'FMD'], [diag_probs['healthy'], diag_probs['fmd']], 
                         color=['green' if diag_probs['healthy'] > diag_probs['fmd'] else 'red',
                                'red' if diag_probs['fmd'] > diag_probs['healthy'] else 'green'])
            axes[2].set_title(f"Diagnosis (FMD Detection)\n({result['diagnosis']['class_name']})")
            axes[2].set_xlim(0, 1)
            for i, (label, prob) in enumerate([('Healthy', diag_probs['healthy']), ('FMD', diag_probs['fmd'])]):
                axes[2].text(prob + 0.02, i, f'{prob:.1%}', va='center')
        else:
            # 2-column layout (original, identification only)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(title or Path(image_path).name, fontsize=14, fontweight='bold')
            
            # Original image
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Identification probabilities
            id_probs = result['identification']['probabilities']
            axes[1].barh(['Cattle', 'Not-Cattle'], [id_probs['cattle'], id_probs['not_cattle']], 
                         color=['green', 'red'])
            axes[1].set_title(f"Identification\n({result['identification']['class_name']})\n\n⚠️ NOT A CATTLE",
                            color='red', fontweight='bold')
            axes[1].set_xlim(0, 1)
            for i, (label, prob) in enumerate([('Cattle', id_probs['cattle']), ('Not-Cattle', id_probs['not_cattle'])]):
                axes[1].text(prob + 0.02, i, f'{prob:.1%}', va='center')
        
        plt.tight_layout()
        return fig
    
    def test_batch(self, image_dir: str, save_results: bool = True):
        """Test multiple images in a directory"""
        
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        print(f"\n🔄 Testing {len(image_files)} images from: {image_dir}")
        print("=" * 60)
        
        results_list = []
        
        for idx, image_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] {image_path.name}")
            
            try:
                result = self.predict(str(image_path))
                alert = self.generate_alert(result)
                
                print(f"  Identification: {result['identification']['class_name']} "
                      f"({result['identification']['confidence']:.1%})")
                if result['diagnosis'] is not None:
                    print(f"  Diagnosis: {result['diagnosis']['class_name']} "
                          f"({result['diagnosis']['confidence']:.1%})")
                else:
                    print(f"  Diagnosis: Not applicable (not cattle)")
                print(f"  Alert: {alert['emoji']} {alert['level']}")
                
                results_list.append({
                    'image': str(image_path.name),
                    'result': result,
                    'alert': alert
                })
            
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
        
        # Summary statistics
        print("\n" + "=" * 60)
        print("📊 SUMMARY STATISTICS")
        print("=" * 60)
        
        if results_list:
            cattle_count = sum(1 for r in results_list if r['result']['identification']['class_id'] == 0)
            fmd_count = sum(1 for r in results_list if r['result']['diagnosis']['class_id'] == 1)
            
            print(f"Total images: {len(results_list)}")
            print(f"Cattle detected: {cattle_count} ({cattle_count/len(results_list)*100:.1f}%)")
            print(f"FMD cases: {fmd_count} ({fmd_count/len(results_list)*100:.1f}%)")
        
        # Save results
        if save_results:
            results_file = image_dir / 'test_results.json'
            with open(results_file, 'w') as f:
                json.dump(results_list, f, indent=2)
            print(f"\n✓ Results saved to: {results_file}")
        
        return results_list

# ============================================
# EXAMPLE USAGE
# ============================================
if __name__ == "__main__":
    
    # Initialize tester
    tester = CattleModelTester(
        model_path='pipeline_output/models/model_final.pt',
        device='auto'
    )
    
    # Test a single image
    test_image = 'test_image.jpg'  # Replace with your image path
    
    if Path(test_image).exists():
        print("\n" + "=" * 60)
        print("SINGLE IMAGE TEST")
        print("=" * 60)
        
        result = tester.predict(test_image)
        alert = tester.generate_alert(result, confidence_threshold=0.7)
        
        print("\n✓ IDENTIFICATION TASK")
        print(f"  Prediction: {result['identification']['class_name']}")
        print(f"  Confidence: {result['identification']['confidence']:.1%}")
        print(f"  Cattle probability: {result['identification']['probabilities']['cattle']:.1%}")
        print(f"  Not-Cattle probability: {result['identification']['probabilities']['not_cattle']:.1%}")
        
        if result['diagnosis'] is not None:
            print("\n✓ DIAGNOSIS TASK (FMD Detection)")
            print(f"  Prediction: {result['diagnosis']['class_name']}")
            print(f"  Confidence: {result['diagnosis']['confidence']:.1%}")
            print(f"  Healthy probability: {result['diagnosis']['probabilities']['healthy']:.1%}")
            print(f"  FMD probability: {result['diagnosis']['probabilities']['fmd']:.1%}")
        else:
            print("\n⚠️  DIAGNOSIS TASK (FMD Detection)")
            print(f"  Status: NOT AVAILABLE - This is not a cattle image")
            print(f"  Action: Please upload a cattle image for FMD diagnosis")
        
        print("\n" + alert['emoji'] + " ALERT DECISION")
        print(f"  Level: {alert['level']}")
        print(f"  Message: {alert['message']}")
        print(f"  Action: {alert['action']}")
        
        # Visualize
        fig = tester.visualize_results(test_image, result)
        plt.show()
    
    else:
        print(f"\n❌ Test image not found: {test_image}")
        print("   Please provide a valid image path in the script.")
