# run_air_quality_pipeline.py
import os
import sys
import argparse
from datetime import datetime
import subprocess
import time

def run_preprocessing():
    """Run data preprocessing"""
    print("\n" + "="*50)
    print("STEP 1: DATA PREPROCESSING")
    print("="*50)
    
    # Check if input file exists
    input_file = '../data/raw/air1.csv'
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Looking for air1.csv in current directory...")
        input_file = 'air1.csv'
        if not os.path.exists(input_file):
            print("❌ air1.csv not found either!")
            return False
    
    # Create command
    cmd = [
        sys.executable,  # Use current Python interpreter
        'air_quality_preprocessing.py',
        '--input', input_file,
        '--output', 'data/processed/air_quality',
        '--seq_len', '24',
        '--stride', '6'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ Preprocessing failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    print("\n✅ Preprocessing complete!")
    return True

def run_training():
    """Run TimeGAN training"""
    print("\n" + "="*50)
    print("STEP 2: TIMEGAN TRAINING")
    print("="*50)
    
    # Check if training data exists
    train_file = 'data/processed/air_quality/train.npy'
    if not os.path.exists(train_file):
        print(f"❌ Training data not found: {train_file}")
        print("You need to run preprocessing first!")
        return False
    
    cmd = [sys.executable, 'train_air_quality_timegan.py']
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    print("\n✅ Training complete!")
    return True

def run_evaluation():
    """Run evaluation"""
    print("\n" + "="*50)
    print("STEP 3: EVALUATION")
    print("="*50)
    
    # Check if synthetic data exists
    synth_dir = 'outputs/synthetic_air_quality'
    if not os.path.exists(synth_dir):
        print(f"❌ Synthetic data directory not found: {synth_dir}")
        print("You need to run training first!")
        return False
    
    synth_files = [f for f in os.listdir(synth_dir) if f.startswith('synthetic_')]
    if not synth_files:
        print("❌ No synthetic files found!")
        return False
    
    cmd = [sys.executable, 'evaluate_air_quality.py']
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    print("\n✅ Evaluation complete!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Air Quality Data Synthesis Pipeline')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--preprocess', action='store_true', help='Run preprocessing only')
    parser.add_argument('--train', action='store_true', help='Run training only')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation only')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.all or (not any([args.preprocess, args.train, args.evaluate])):
        # Run complete pipeline
        print("\n" + "="*60)
        print("STARTING COMPLETE AIR QUALITY PIPELINE")
        print("="*60)
        
        success = True
        
        # Step 1: Preprocessing
        if not run_preprocessing():
            print("❌ Pipeline stopped at preprocessing!")
            success = False
        
        # Step 2: Training
        if success:
            time.sleep(2)  # Brief pause
            if not run_training():
                print("❌ Pipeline stopped at training!")
                success = False
        
        # Step 3: Evaluation
        if success:
            time.sleep(2)  # Brief pause
            if not run_evaluation():
                print("❌ Pipeline stopped at evaluation!")
                success = False
        
        if success:
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("PIPELINE FAILED!")
            print("="*60)
    
    else:
        # Run individual steps
        if args.preprocess:
            run_preprocessing()
        if args.train:
            run_training()
        if args.evaluate:
            run_evaluation()
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir != os.getcwd():
        print(f"Changing to script directory: {script_dir}")
        os.chdir(script_dir)
    
    main()