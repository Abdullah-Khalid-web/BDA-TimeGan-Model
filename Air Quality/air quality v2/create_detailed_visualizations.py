# create_detailed_visualizations.py
"""
Create detailed visualizations for air quality Time-GAN analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

def create_pipeline_flow_chart():
    """Create pipeline flow chart"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define nodes
    nodes = [
        ("Raw Data\n(air1.csv)", (0.1, 0.5)),
        ("Preprocessing\n(air_quality_preprocessing.py)", (0.3, 0.5)),
        ("TimeGAN Training\n(train_air_quality_timegan.py)", (0.5, 0.5)),
        ("Synthetic Data\nGeneration", (0.7, 0.5)),
        ("Evaluation\n(evaluate_air_quality.py)", (0.9, 0.5))
    ]
    
    # Draw nodes
    for i, (label, pos) in enumerate(nodes):
        ax.add_patch(plt.Rectangle((pos[0]-0.08, pos[1]-0.05), 0.16, 0.1,
                                 fill=True, color='lightblue', ec='black', lw=2))
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add step numbers
        ax.text(pos[0]-0.08, pos[1]+0.08, f"Step {i+1}", ha='center', va='center', 
                fontsize=8, fontweight='bold', color='red')
    
    # Draw arrows
    for i in range(len(nodes)-1):
        x1, y1 = nodes[i][1]
        x2, y2 = nodes[i+1][1]
        ax.arrow(x1+0.08, y1, x2-x1-0.16, 0, 
                head_width=0.02, head_length=0.02, fc='black', ec='black', lw=2)
    
    # Add outputs
    outputs = [
        ("Processed Data\n(train.npy, val.npy, test.npy)", (0.3, 0.3)),
        ("Trained Model\n(checkpoints/)", (0.5, 0.3)),
        ("Synthetic Data\n(synthetic_*.npy)", (0.7, 0.3)),
        ("Evaluation Results\n(evaluation_*.json)", (0.9, 0.3))
    ]
    
    for label, pos in outputs:
        ax.add_patch(plt.Rectangle((pos[0]-0.08, pos[1]-0.04), 0.16, 0.08,
                                 fill=True, color='lightgreen', ec='black', lw=1, alpha=0.7))
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=8)
        
        # Connect to main flow
        main_y = 0.5
        ax.plot([pos[0], pos[0]], [pos[1]+0.04, main_y-0.05], 'k--', alpha=0.5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Air Quality Time-GAN Pipeline Flow Chart', fontsize=14, fontweight='bold')
    
    # Add legend
    ax.text(0.05, 0.9, 'Main Process', fontsize=10, fontweight='bold', color='blue')
    ax.text(0.05, 0.85, 'Output Files', fontsize=10, fontweight='bold', color='green')
    
    plt.tight_layout()
    
    # Save
    os.makedirs('analysis_results/visualizations', exist_ok=True)
    plt.savefig('analysis_results/visualizations/pipeline_flow_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Created pipeline flow chart")

def create_performance_dashboard():
    """Create performance dashboard"""
    # Try to load analysis results
    try:
        with open('analysis_results/analysis_results.json', 'r') as f:
            results = json.load(f)
    except:
        print("⚠️ Analysis results not found, creating template dashboard")
        results = {}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Pipeline Status (Pie Chart)
    ax = axes[0, 0]
    components = ['Preprocessing', 'Training', 'Evaluation', 'Comparison']
    status = [1 if comp.lower() in results else 0 for comp in components]
    
    colors = ['green' if s == 1 else 'red' for s in status]
    ax.pie([s+0.1 for s in status], labels=components, colors=colors, autopct='%1.0f%%')
    ax.set_title('Pipeline Component Status', fontweight='bold')
    
    # 2. Quality Scores (Bar Chart)
    ax = axes[0, 1]
    scores = {}
    
    if 'preprocessing' in results:
        scores['Preprocessing'] = results['preprocessing'].get('preprocessing_quality_score', 0)
    
    if 'training' in results:
        scores['Training'] = results['training'].get('training_score', 0)
    
    if 'evaluation' in results:
        scores['Evaluation'] = results['evaluation'].get('overall_score', 0)
    
    if 'comparison' in results:
        scores['Comparison'] = results['comparison'].get('overall_similarity', 0)
    
    if scores:
        components = list(scores.keys())
        values = list(scores.values())
        
        bars = ax.bar(components, values, color=['blue', 'green', 'orange', 'red'])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score')
        ax.set_title('Component Quality Scores', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{value:.3f}', ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'No score data available', ha='center', va='center')
        ax.set_title('Component Quality Scores', fontweight='bold')
    
    # 3. Data Statistics
    ax = axes[0, 2]
    ax.axis('off')
    
    stats_text = "Dataset Statistics\n\n"
    
    if 'preprocessing' in results and 'data_shapes' in results['preprocessing']:
        shapes = results['preprocessing']['data_shapes']
        stats_text += f"Training: {shapes['train'][0]} samples\n"
        stats_text += f"Validation: {shapes['val'][0]} samples\n"
        stats_text += f"Test: {shapes['test'][0]} samples\n\n"
        
        stats_text += f"Sequence length: {shapes['train'][1]}\n"
        stats_text += f"Features: {shapes['train'][2]}\n"
    
    if 'training' in results and 'synthetic_data' in results['training']:
        synth = results['training']['synthetic_data']
        stats_text += f"\nSynthetic Data:\n"
        stats_text += f"Samples: {synth['samples']:,}\n"
    
    ax.text(0.1, 0.9, stats_text, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Data Statistics', fontweight='bold')
    
    # 4. Key Metrics Table
    ax = axes[1, 0]
    ax.axis('off')
    
    metrics_text = "Key Evaluation Metrics\n\n"
    
    if 'evaluation' in results and 'computed_metrics' in results['evaluation']:
        metrics = results['evaluation']['computed_metrics']
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metrics_text += f"{key:25s}: {value:.4f}\n"
    else:
        metrics_text += "No evaluation metrics available"
    
    ax.text(0.1, 0.9, metrics_text, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.set_title('Evaluation Metrics', fontweight='bold')
    
    # 5. Issues & Recommendations
    ax = axes[1, 1]
    ax.axis('off')
    
    issues_text = "Issues & Recommendations\n\n"
    
    all_issues = []
    
    if 'training' in results and 'issues' in results['training']:
        all_issues.extend(results['training']['issues'])
    
    if all_issues:
        for i, issue in enumerate(all_issues[:3], 1):  # Show first 3 issues
            issues_text += f"{i}. {issue}\n"
    else:
        issues_text += "No major issues detected\n"
        issues_text += "\nRecommendations:\n"
        issues_text += "1. Generate more synthetic data\n"
        issues_text += "2. Experiment with different models\n"
        issues_text += "3. Test on downstream tasks\n"
    
    ax.text(0.1, 0.9, issues_text, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    ax.set_title('Issues & Recommendations', fontweight='bold')
    
    # 6. Overall Assessment
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate overall score
    scores_list = []
    if 'preprocessing' in results:
        scores_list.append(results['preprocessing'].get('preprocessing_quality_score', 0))
    if 'training' in results:
        scores_list.append(results['training'].get('training_score', 0))
    if 'evaluation' in results:
        scores_list.append(results['evaluation'].get('overall_score', 0))
    if 'comparison' in results:
        scores_list.append(results['comparison'].get('overall_similarity', 0))
    
    if scores_list:
        overall_score = np.mean(scores_list)
        
        if overall_score >= 0.8:
            assessment = "EXCELLENT\n\nPipeline is working\nvery effectively."
            color = 'green'
        elif overall_score >= 0.6:
            assessment = "GOOD\n\nPipeline is functional\nand producing results."
            color = 'blue'
        elif overall_score >= 0.4:
            assessment = "FAIR\n\nPipeline works but\nneeds improvements."
            color = 'orange'
        else:
            assessment = "POOR\n\nSignificant issues\nneed to be addressed."
            color = 'red'
        
        ax.text(0.5, 0.7, f"Overall Score: {overall_score:.3f}", 
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.4, assessment, ha='center', va='center', 
                fontsize=12, fontweight='bold', color=color,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    else:
        ax.text(0.5, 0.5, "No assessment data available", 
                ha='center', va='center', fontsize=12)
    
    ax.set_title('Overall Assessment', fontweight='bold')
    
    # Add title
    plt.suptitle('AIR QUALITY TIME-GAN PIPELINE PERFORMANCE DASHBOARD', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    plt.savefig('analysis_results/visualizations/performance_dashboard.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Created performance dashboard")

def create_training_progress_plot():
    """Create training progress visualization"""
    # Look for training history files
    import glob
    
    history_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if 'history' in file.lower() and file.endswith('.json'):
                history_files.append(os.path.join(root, file))
    
    if not history_files:
        print("⚠️ No training history files found")
        return
    
    # Load the most recent history
    latest_file = max(history_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r') as f:
            history = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Loss curves
        if 'd_loss' in history and 'g_loss' in history:
            ax = axes[0, 0]
            epochs = range(1, len(history['d_loss']) + 1)
            ax.plot(epochs, history['d_loss'], label='Discriminator Loss', linewidth=2)
            ax.plot(epochs, history['g_loss'], label='Generator Loss', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Losses')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Predictions
        if 'real_pred' in history and 'fake_pred' in history:
            ax = axes[0, 1]
            epochs = range(1, len(history['real_pred']) + 1)
            ax.plot(epochs, history['real_pred'], label='Real Predictions', linewidth=2)
            ax.plot(epochs, history['fake_pred'], label='Fake Predictions', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Prediction Score')
            ax.set_title('Discriminator Predictions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        # 3. Validation scores
        if 'val_scores' in history:
            ax = axes[1, 0]
            val_epochs = range(5, 5 * len(history['val_scores']) + 1, 5)
            ax.plot(val_epochs, history['val_scores'], 'g-', linewidth=2, marker='o')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Validation Score')
            ax.set_title('Validation Scores')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        # 4. Training summary
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "Training Summary\n\n"
        summary_text += f"Total Epochs: {len(history.get('d_loss', []))}\n"
        
        if 'best_score' in history:
            summary_text += f"Best Score: {history['best_score']:.4f}\n"
        
        if 'val_scores' in history:
            summary_text += f"Final Val Score: {history['val_scores'][-1]:.4f}\n"
        
        ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Summary')
        
        plt.suptitle('TimeGAN Training Progress', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        plt.savefig('analysis_results/visualizations/training_progress.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Created training progress visualization")
        
    except Exception as e:
        print(f"⚠️ Error creating training plot: {e}")

def main():
    """Create all visualizations"""
    print("\n" + "="*60)
    print("CREATING DETAILED VISUALIZATIONS")
    print("="*60)
    
    # Create output directory
    os.makedirs('analysis_results/visualizations', exist_ok=True)
    
    # Create visualizations
    create_pipeline_flow_chart()
    create_performance_dashboard()
    create_training_progress_plot()
    
    print("\n" + "="*60)
    print("VISUALIZATIONS COMPLETED!")
    print("="*60)
    print("\nVisualizations saved to: analysis_results/visualizations/")
    print("\nFiles created:")
    print("  • pipeline_flow_chart.png")
    print("  • performance_dashboard.png")
    print("  • training_progress.png")

if __name__ == "__main__":
    main()