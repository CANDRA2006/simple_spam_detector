#!/usr/bin/env python
"""
Quick Start Script untuk Spam Detector
Jalankan: python run_all.py
"""

import os
import sys

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_file_exists(filepath):
    if os.path.exists(filepath):
        print(f"✅ Found: {filepath}")
        return True
    else:
        print(f"❌ Missing: {filepath}")
        return False

def run_command(description, command):
    print_header(description)
    result = os.system(command)
    if result != 0:
        print(f"\n❌ Error running: {description}")
        return False
    print(f"\n✅ Success: {description}")
    return True

def main():
    print_header("SPAM DETECTOR - QUICK START")
    
    # Check dataset
    print("📁 Checking files...")
    dataset_exists = check_file_exists('data/raw/spam.csv')
    
    if not dataset_exists:
        print("\n❌ ERROR: Dataset not found!")
        print("Please place spam.csv in data/raw/ folder")
        return
    
    # Check required folders
    folders = ['data/raw', 'data/processed', 'model', 'src', 'app']
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Creating folder: {folder}")
            os.makedirs(folder, exist_ok=True)
    
    print("\n✅ All folders ready!")
    
    # Ask user what to do
    print_header("MENU")
    print("1. Run FULL PIPELINE (Preprocess → Train → Evaluate)")
    print("2. Run PREPROCESS only")
    print("3. Run TRAIN only")
    print("4. Run EVALUATE only")
    print("5. Run WEB APP only")
    print("6. Run TEST PREDICTION")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        # Full pipeline
        if not run_command("Step 1/3: Preprocessing Data", "python src/preprocess.py"):
            return
        if not run_command("Step 2/3: Training Model", "python src/train.py"):
            return
        if not run_command("Step 3/3: Evaluating Model", "python src/evaluate.py"):
            return
        
        print_header("🎉 COMPLETE!")
        print("Model is ready! You can now:")
        print("  - Run web app: cd app && python app.py")
        print("  - Test predictions: python src/predict.py")
        
    elif choice == '2':
        run_command("Preprocessing Data", "python src/preprocess.py")
        
    elif choice == '3':
        if not check_file_exists('data/processed/X_train.pkl'):
            print("\n❌ Please run preprocessing first!")
            return
        run_command("Training Model", "python src/train.py")
        
    elif choice == '4':
        if not check_file_exists('model/spam_model.pkl'):
            print("\n❌ Please train model first!")
            return
        run_command("Evaluating Model", "python src/evaluate.py")
        
    elif choice == '5':
        if not check_file_exists('model/spam_model.pkl'):
            print("\n❌ Please train model first!")
            return
        print_header("Starting Web App")
        print("Access at: http://localhost:5000")
        print("Press Ctrl+C to stop")
        os.system("cd app && python app.py")
        
    elif choice == '6':
        if not check_file_exists('model/spam_model.pkl'):
            print("\n❌ Please train model first!")
            return
        run_command("Testing Predictions", "python src/predict.py")
        
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()