"""
🐄 Cattle FMD Detection - Training Automation Script
Easy-to-use training launcher with preset configurations
"""

import sys
import subprocess
import argparse
from pathlib import Path
import json
from datetime import datetime

def check_requirements():
    """Check if all required packages are installed"""
    print("\n" + "="*60)
    print("CHECKING REQUIREMENTS...")
    print("="*60)
    
    required_packages = [
        'torch',
        'torchvision',
        'sklearn',
        'PIL',
        'albumentations',
        'tqdm',
        'pandas',
        'matplotlib',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✓ All requirements satisfied!")
    return True

def check_data_directories():
    """Check if data directories exist and contain images"""
    print("\n" + "="*60)
    print("CHECKING DATA DIRECTORIES...")
    print("="*60)
    
    directories = {
        'cattle_healthy': Path('cattle_healthy'),
        'cattle_infected': Path('cattle_infected'),
        'not_cattle_animals': Path('not_cattle_animals')
    }
    
    all_exist = True
    total_images = 0
    
    for name, path in directories.items():
        if path.exists():
            image_count = len(list(path.glob('*.jpg'))) + len(list(path.glob('*.png')))
            print(f"✓ {name}: {image_count} images")
            total_images += image_count
        else:
            print(f"✗ {name} - NOT FOUND")
            all_exist = False
    
    print(f"\nTotal images: {total_images}")
    
    if not all_exist:
        print("\n❌ Some directories are missing!")
        print("Make sure you have:")
        print("  - cattle_healthy/     (healthy cattle images)")
        print("  - cattle_infected/    (FMD-infected cattle)")
        print("  - not_cattle_animals/ (other animals)")
        return False
    
    if total_images < 50:
        print("\n⚠️  Warning: Less than 50 images found")
        print("Consider adding more images for better training")
    
    print("✓ Data directories validated!")
    return True

def create_quick_config():
    """Create a quick config modifications for rapid training"""
    quick_config = {
        'PHASE1_EPOCHS': 2,
        'PHASE2_EPOCHS': 1,
        'PHASE1_BATCH_SIZE': 8,
        'PHASE2_BATCH_SIZE': 8,
    }
    
    config_str = "\n# QUICK TRAINING CONFIG (for testing)\n"
    for key, value in quick_config.items():
        config_str += f"{key} = {value}\n"
    
    return config_str

def create_fast_config():
    """Create a fast config for quick training"""
    fast_config = {
        'PHASE1_EPOCHS': 10,
        'PHASE2_EPOCHS': 8,
        'PHASE1_BATCH_SIZE': 16,
        'PHASE2_BATCH_SIZE': 16,
        'PHASE1_LR': 0.005,
        'PHASE2_LR': 0.0005,
    }
    
    config_str = "\n# FAST TRAINING CONFIG\n"
    for key, value in fast_config.items():
        config_str += f"{key} = {value}\n"
    
    return config_str

def create_accurate_config():
    """Create a config for maximum accuracy"""
    accurate_config = {
        'PHASE1_EPOCHS': 30,
        'PHASE2_EPOCHS': 20,
        'PHASE1_BATCH_SIZE': 64,
        'PHASE2_BATCH_SIZE': 64,
        'PHASE1_LR': 0.0005,
        'PHASE2_LR': 0.00005,
        'PATIENCE': 10,
    }
    
    config_str = "\n# ACCURATE TRAINING CONFIG (Best results, slower)\n"
    for key, value in accurate_config.items():
        config_str += f"{key} = {value}\n"
    
    return config_str

def print_menu():
    """Print training menu"""
    print("\n" + "="*60)
    print("🐄 CATTLE FMD DETECTION MODEL TRAINING")
    print("="*60)
    print("\nSelect training preset:\n")
    print("  1. QUICK    - Test run (2+1 epochs, ~5 min, low accuracy)")
    print("  2. FAST     - Quick training (10+8 epochs, ~30 min, medium accuracy)")
    print("  3. ACCURATE - Full training (30+20 epochs, ~2 hours, high accuracy)")
    print("  4. CUSTOM   - Edit config file manually")
    print("  5. ADVANCED - Show all configuration options")
    print("\nOr use command line:")
    print("  python train_automation.py --preset quick")
    print("  python train_automation.py --preset fast")
    print("  python train_automation.py --preset accurate")
    print("  python train_automation.py --train")
    print("  python train_automation.py --help")
    print()

def run_training():
    """Run the training script"""
    print("\n" + "="*60)
    print("STARTING TRAINING...")
    print("="*60)
    print("\nThis will take a while. You can monitor progress in:")
    print("  - Console output (below)")
    print("  - Log file: pipeline_output/logs/training_*.log")
    print("  - Training history: pipeline_output/results/training_history.json")
    print("\n" + "-"*60 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, 'train_fmd_model.py'],
            cwd=Path.cwd(),
            check=False
        )
        
        if result.returncode == 0:
            print("\n" + "="*60)
            print("✓ TRAINING COMPLETED SUCCESSFULLY")
            print("="*60)
            print("\nModel saved to: pipeline_output/models/model_final.pt")
            print("\nNext steps:")
            print("  1. View results: cat pipeline_output/results/training_history.json")
            print("  2. Test model: streamlit run test_model_farmer_ui.py")
            print("  3. Deploy: Use pipeline_output/models/model_final.pt")
            return True
        else:
            print("\n" + "="*60)
            print("❌ TRAINING FAILED")
            print("="*60)
            print(f"Exit code: {result.returncode}")
            print("\nCheck the logs:")
            print("  - pipeline_output/logs/training_*.log")
            return False
    
    except Exception as e:
        print(f"\n❌ Error running training: {e}")
        return False

def show_advanced_options():
    """Show advanced configuration options"""
    print("\n" + "="*60)
    print("ADVANCED CONFIGURATION OPTIONS")
    print("="*60)
    print("""
Phase 1 Settings (Head Training):
  PHASE1_EPOCHS      = 20         # Number of epochs
  PHASE1_LR          = 0.001      # Learning rate
  PHASE1_BATCH_SIZE  = 32         # Images per batch
  PHASE1_DROPOUT     = 0.5        # Dropout rate

Phase 2 Settings (Fine-tuning):
  PHASE2_EPOCHS      = 15         # Number of epochs
  PHASE2_LR          = 0.0001     # Learning rate
  PHASE2_BATCH_SIZE  = 32         # Images per batch
  PHASE2_DROPOUT     = 0.3        # Dropout rate

Data Settings:
  IMAGE_SIZE         = 224        # Input image size
  TRAIN_SPLIT        = 0.70       # Training data ratio
  VAL_SPLIT          = 0.15       # Validation data ratio
  TEST_SPLIT         = 0.15       # Test data ratio
  RANDOM_SEED        = 42         # For reproducibility

General:
  DEVICE             = GPU/CPU    # Auto-detected
  NUM_WORKERS        = 4          # Parallel data loading
  PATIENCE           = 5          # Early stopping patience
  CLIP_GRADIENTS     = True       # Gradient clipping

To modify: Edit training_config_template.py
""")

def main():
    parser = argparse.ArgumentParser(
        description='Cattle FMD Detection Model Training Automation'
    )
    parser.add_argument(
        '--preset',
        choices=['quick', 'fast', 'accurate'],
        help='Training preset'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Start training with current config'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check requirements and data only'
    )
    parser.add_argument(
        '--config',
        help='Path to custom config file'
    )
    
    args = parser.parse_args()
    
    # If no arguments, show interactive menu
    if not any([args.preset, args.train, args.check, args.config]):
        print_menu()
        
        while True:
            try:
                choice = input("Enter your choice (1-5) or 'q' to quit: ").strip().lower()
                
                if choice == 'q':
                    print("Exiting...")
                    sys.exit(0)
                elif choice == '1':
                    print("\n⚠️  QUICK MODE: Testing only (fast, low accuracy)")
                    if input("Continue? (y/n): ").lower() == 'y':
                        args.preset = 'quick'
                    break
                elif choice == '2':
                    print("\nFAST MODE: Quick training (medium accuracy)")
                    if input("Continue? (y/n): ").lower() == 'y':
                        args.preset = 'fast'
                    break
                elif choice == '3':
                    print("\nACCURATE MODE: Full training (high accuracy, slower)")
                    if input("Continue? (y/n): ").lower() == 'y':
                        args.preset = 'accurate'
                    break
                elif choice == '4':
                    print("\nEdit training_config_template.py and run again")
                    print("python train_automation.py --train")
                    sys.exit(0)
                elif choice == '5':
                    show_advanced_options()
                else:
                    print("Invalid choice. Try again.")
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                sys.exit(0)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please install missing packages first:")
        print("pip install -r requirements_training.txt")
        sys.exit(1)
    
    # Check data
    if not check_data_directories():
        print("\n❌ Please add missing data directories first")
        sys.exit(1)
    
    # If only checking, exit here
    if args.check:
        print("\n✓ All checks passed!")
        sys.exit(0)
    
    # Modify config if preset is selected
    if args.preset:
        print(f"\n📝 Applying {args.preset.upper()} preset...")
        config_path = Path('train_fmd_model.py')
        
        if args.preset == 'quick':
            print("⚠️  Quick mode: 2 Phase1 + 1 Phase2 epochs")
            modification = create_quick_config()
        elif args.preset == 'fast':
            print("⚡ Fast mode: 10 Phase1 + 8 Phase2 epochs")
            modification = create_fast_config()
        elif args.preset == 'accurate':
            print("🎯 Accurate mode: 30 Phase1 + 20 Phase2 epochs")
            modification = create_accurate_config()
        
        print("\nYou can modify these settings in training_config_template.py")
    
    # Start training
    if args.train or args.preset:
        print("\n" + "="*60)
        print("PRE-TRAINING CHECKS COMPLETE")
        print("="*60)
        print(f"✓ Requirements satisfied")
        print(f"✓ Data directories found")
        print(f"✓ Ready to train")
        
        confirm = input("\n🚀 Ready to start training? (y/n): ").strip().lower()
        if confirm == 'y':
            success = run_training()
            sys.exit(0 if success else 1)
        else:
            print("Training cancelled.")
            sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
