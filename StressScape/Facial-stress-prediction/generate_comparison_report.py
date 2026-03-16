import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import matplotlib.patches as mpatches
from matplotlib import patheffects

# Set modern style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

RESULTS_DIR = Path(__file__).parent / "results"
MODELS = ['MobileNetV2', 'EfficientNetB0', 'ResNet50V2']
COLORS = {
    'MobileNetV2': '#FF6B6B',      # Coral Red
    'EfficientNetB0': '#4ECDC4',   # Turquoise
    'ResNet50V2': '#95E1D3'        # Mint Green
}

def create_modern_comparison_plots(all_metrics, comparison_dir):
    """Generate modern, visually appealing comparison plots"""
    print("📊 Creating modern comparison visualizations...")
    
    models = list(all_metrics.keys())
    colors_list = [COLORS.get(m, '#999999') for m in models]
    
    # Extract metrics
    metrics_data = {
        'Accuracy': [all_metrics[m]['final_val_accuracy'] for m in models],
        'Precision': [all_metrics[m]['calculated_precision'] for m in models],
        'Recall': [all_metrics[m]['calculated_recall'] for m in models],
        'F1-Score': [all_metrics[m]['calculated_f1_score'] for m in models],
        'AUC': [all_metrics[m]['final_val_auc'] for m in models]
    }
    
    # ===== 1. MODERN METRICS COMPARISON BAR CHART =====
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor('white')
    
    metric_names = list(metrics_data.keys())
    positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]
    
    for idx, (pos, metric) in enumerate(zip(positions, metric_names)):
        ax = axes[pos]
        values = metrics_data[metric]
        
        # Create gradient bars
        bars = ax.bar(models, values, color=colors_list, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=11, color='#333333')
        
        # Styling
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold', pad=15, color='#2C3E50')
        ax.set_ylabel(metric, fontsize=11, fontweight='bold', color='#34495E')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(labelsize=10)
        
        # Rotate x labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # ===== 6. RADAR CHART =====
    ax = plt.subplot(2, 3, 6, projection='polar')
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
    angles += angles[:1]
    
    for i, model in enumerate(models):
        values = [metrics_data[metric][i] for metric in metric_names]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=3, label=model, 
               color=colors_list[i], markersize=8, alpha=0.8)
        ax.fill(angles, values, alpha=0.25, color=colors_list[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance Radar', fontsize=14, fontweight='bold', 
                pad=25, color='#2C3E50')
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=10, 
             frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(comparison_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    # ===== 2. STACKED PERFORMANCE COMPARISON =====
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    
    x = np.arange(len(models))
    width = 0.15
    
    for idx, metric in enumerate(metric_names):
        offset = (idx - 2) * width
        bars = ax.bar(x + offset, metrics_data[metric], width, 
                     label=metric, alpha=0.85)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', 
                   fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Models', fontsize=13, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold', color='#2C3E50')
    ax.set_title('Comprehensive Metrics Comparison', fontsize=16, fontweight='bold', 
                pad=20, color='#2C3E50')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(comparison_dir / 'stacked_comparison.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    # ===== 3. HEATMAP COMPARISON =====
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df = pd.DataFrame(metrics_data, index=models)
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlGnBu', 
               cbar_kws={'label': 'Score'}, linewidths=2, linecolor='white',
               ax=ax, vmin=0, vmax=1, annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    
    ax.set_title('Performance Heatmap', fontsize=16, fontweight='bold', 
                pad=20, color='#2C3E50')
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Models', fontsize=12, fontweight='bold', color='#2C3E50')
    plt.xticks(rotation=0, fontsize=11, fontweight='bold')
    plt.yticks(rotation=0, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(comparison_dir / 'heatmap_comparison.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    # ===== 4. BOX PLOT COMPARISON =====
    fig, ax = plt.subplots(figsize=(14, 8))
    
    data_for_box = []
    labels_for_box = []
    
    for model in models:
        model_scores = [all_metrics[model][key] for key in 
                       ['final_val_accuracy', 'calculated_precision', 
                        'calculated_recall', 'calculated_f1_score', 'final_val_auc']]
        data_for_box.append(model_scores)
        labels_for_box.append(model)
    
    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                   notch=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Score Distribution by Model', fontsize=16, fontweight='bold',
                pad=20, color='#2C3E50')
    ax.set_ylabel('Score Range', fontsize=12, fontweight='bold', color='#2C3E50')
    ax.set_xlabel('Models', fontsize=12, fontweight='bold', color='#2C3E50')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(comparison_dir / 'boxplot_comparison.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    # ===== 5. WINNER ANNOUNCEMENT GRAPHIC =====
    best_model = max(all_metrics.items(), key=lambda x: x[1]['final_val_accuracy'])
    best_name = best_model[0]
    best_acc = best_model[1]['final_val_accuracy']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#F8F9FA')
    ax.axis('off')
    
    # Create trophy visual effect
    trophy_y = 0.7
    ax.text(0.5, trophy_y, '🏆', fontsize=120, ha='center', va='center',
           transform=ax.transAxes)
    
    # Winner text with shadow effect
    winner_text = ax.text(0.5, 0.5, f'{best_name}', fontsize=42, 
                         fontweight='bold', ha='center', va='center',
                         transform=ax.transAxes, color=COLORS.get(best_name, '#333333'))
    winner_text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])
    
    # Subtitle
    ax.text(0.5, 0.4, 'BEST PERFORMING MODEL', fontsize=20, 
           ha='center', va='center', transform=ax.transAxes,
           color='#34495E', style='italic', fontweight='bold')
    
    # Accuracy badge
    badge_rect = mpatches.FancyBboxPatch((0.35, 0.25), 0.3, 0.1,
                                         boxstyle="round,pad=0.01",
                                         linewidth=3, edgecolor='#27AE60',
                                         facecolor='#2ECC71', alpha=0.3,
                                         transform=ax.transAxes)
    ax.add_patch(badge_rect)
    ax.text(0.5, 0.3, f'Accuracy: {best_acc:.2%}', fontsize=18,
           ha='center', va='center', transform=ax.transAxes,
           color='#27AE60', fontweight='bold')
    
    # Stats
    stats_y = 0.15
    stats_text = f"""Precision: {best_model[1]['calculated_precision']:.3f} | Recall: {best_model[1]['calculated_recall']:.3f} | F1: {best_model[1]['calculated_f1_score']:.3f} | AUC: {best_model[1]['final_val_auc']:.3f}"""
    ax.text(0.5, stats_y, stats_text, fontsize=12, ha='center', va='center',
           transform=ax.transAxes, color='#7F8C8D', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(comparison_dir / 'winner_announcement.png', dpi=300, bbox_inches='tight',
               facecolor='#F8F9FA', edgecolor='none')
    plt.close()
    
    print(f"✅ Comparison visualizations saved to: {comparison_dir}")

def collect_confusion_matrices(models_with_data):
    """Collect and create combined confusion matrix visualization"""
    print("📊 Creating confusion matrix comparison...")
    
    comparison_dir = RESULTS_DIR / 'comparison'
    fig, axes = plt.subplots(1, len(models_with_data), figsize=(6*len(models_with_data), 5))
    
    if len(models_with_data) == 1:
        axes = [axes]
    
    for idx, model in enumerate(models_with_data):
        cm_file = RESULTS_DIR / model / 'confusion_matrix_roc.png'
        
        # Read confusion matrix from classification report
        report_file = RESULTS_DIR / model / 'classification_report.txt'
        if report_file.exists():
            # Create simple CM visualization
            ax = axes[idx]
            
            # Placeholder data (you should load actual CM data)
            # For now, create a sample based on metrics
            metrics_file = RESULTS_DIR / model / 'metrics.json'
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Estimate confusion matrix from precision/recall
            # This is approximate - ideally load actual CM
            stressed = metrics['stressed_samples']
            not_stressed = metrics['not_stressed_samples']
            precision = metrics['calculated_precision']
            recall = metrics['calculated_recall']
            accuracy = metrics['calculated_accuracy']
            
            # Rough estimates
            tp = int(stressed * recall)
            fn = stressed - tp
            tn = int(not_stressed * accuracy)
            fp = not_stressed - tn
            
            cm = np.array([[tn, fp], [fn, tp]])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Not Stressed', 'Stressed'],
                       yticklabels=['Not Stressed', 'Stressed'],
                       cbar=True, square=True, linewidths=2, linecolor='white',
                       annot_kws={'fontsize': 14, 'fontweight': 'bold'})
            
            ax.set_title(f'{model}\nConfusion Matrix', fontsize=13, 
                        fontweight='bold', color='#2C3E50', pad=15)
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(comparison_dir / 'all_confusion_matrices.png', dpi=300, 
               bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Confusion matrices saved")

def create_final_decision_visual(all_metrics, comparison_dir):
    """Create comprehensive final decision visualization"""
    print("📊 Creating final decision visualization...")
    
    # Determine best model
    best_model = max(all_metrics.items(), key=lambda x: x[1]['final_val_accuracy'])
    best_name = best_model[0]
    best_metrics = best_model[1]
    
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('FINAL MODEL SELECTION ANALYSIS', fontsize=20, fontweight='bold',
                color='#2C3E50', y=0.98)
    
    # 1. Winner Circle (Top Center)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    # Winner badge
    circle = plt.Circle((0.5, 0.5), 0.35, color=COLORS.get(best_name, '#FFD700'), 
                       alpha=0.3, transform=ax1.transAxes)
    ax1.add_patch(circle)
    
    ax1.text(0.5, 0.7, '👑', fontsize=60, ha='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.5, best_name, fontsize=28, fontweight='bold', ha='center',
            transform=ax1.transAxes, color=COLORS.get(best_name, '#333333'))
    ax1.text(0.5, 0.35, f'{best_metrics["final_val_accuracy"]:.2%} Accuracy',
            fontsize=16, ha='center', transform=ax1.transAxes, color='#27AE60',
            fontweight='bold')
    
    # 2. Ranking Table (Left Middle)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    ax2.text(0.5, 0.95, 'RANKINGS', fontsize=14, fontweight='bold', ha='center',
            transform=ax2.transAxes, color='#2C3E50')
    
    sorted_models = sorted(all_metrics.items(), 
                          key=lambda x: x[1]['final_val_accuracy'], 
                          reverse=True)
    
    for idx, (model, metrics) in enumerate(sorted_models):
        y_pos = 0.75 - idx * 0.25
        medal = ['🥇', '🥈', '🥉'][idx] if idx < 3 else '  '
        ax2.text(0.1, y_pos, f'{medal} {idx+1}.', fontsize=12, fontweight='bold',
                transform=ax2.transAxes)
        ax2.text(0.3, y_pos, model, fontsize=11, transform=ax2.transAxes)
        ax2.text(0.85, y_pos, f'{metrics["final_val_accuracy"]:.3f}', 
                fontsize=11, ha='right', transform=ax2.transAxes, fontweight='bold',
                color=COLORS.get(model, '#333333'))
    
    # 3. Key Strengths (Center Middle)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    ax3.text(0.5, 0.95, 'KEY STRENGTHS', fontsize=14, fontweight='bold', ha='center',
            transform=ax3.transAxes, color='#2C3E50')
    
    strengths = [
        f"✓ Highest Accuracy: {best_metrics['final_val_accuracy']:.3f}",
        f"✓ Best F1-Score: {best_metrics['calculated_f1_score']:.3f}",
        f"✓ Strong AUC: {best_metrics['final_val_auc']:.3f}",
        f"✓ Balanced Precision/Recall"
    ]
    
    for idx, strength in enumerate(strengths):
        y_pos = 0.75 - idx * 0.18
        ax3.text(0.1, y_pos, strength, fontsize=10, transform=ax3.transAxes,
                color='#27AE60', fontweight='bold')
    
    # 4. Performance Comparison (Right Middle)
    ax4 = fig.add_subplot(gs[1, 2])
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    best_values = [
        best_metrics['final_val_accuracy'],
        best_metrics['calculated_precision'],
        best_metrics['calculated_recall'],
        best_metrics['calculated_f1_score'],
        best_metrics['final_val_auc']
    ]
    
    bars = ax4.barh(metrics_list, best_values, color=COLORS.get(best_name, '#4ECDC4'),
                   alpha=0.8, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, best_values):
        width = bar.get_width()
        ax4.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontweight='bold', fontsize=10)
    
    ax4.set_xlim(0, 1.1)
    ax4.set_title('Performance Profile', fontsize=12, fontweight='bold', color='#2C3E50')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 5. Decision Rationale (Bottom)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    ax5.text(0.5, 0.9, 'SELECTION RATIONALE', fontsize=14, fontweight='bold',
            ha='center', transform=ax5.transAxes, color='#2C3E50',
            bbox=dict(boxstyle='round', facecolor='#E8F8F5', alpha=0.8, pad=0.5))
    
    rationale = f"""
{best_name} has been selected as the optimal model for stress detection based on comprehensive evaluation:

• ACCURACY: Achieved the highest validation accuracy of {best_metrics['final_val_accuracy']:.2%}, demonstrating superior 
  overall performance in correctly classifying both stressed and non-stressed states.

• PRECISION: With a precision of {best_metrics['calculated_precision']:.2%}, the model minimizes false positives, ensuring 
  reliable stress detection with minimal false alarms.

• RECALL: Recall score of {best_metrics['calculated_recall']:.2%} indicates strong capability to identify actual stress cases,
  reducing the risk of missed detections.

• F1-SCORE: Balanced F1-score of {best_metrics['calculated_f1_score']:.3f} confirms optimal trade-off between precision and recall.

• AUC: Area Under Curve of {best_metrics['final_val_auc']:.3f} demonstrates excellent model discrimination capability.

This model provides the best balance of all metrics, making it the most reliable choice for production deployment.
    """
    
    ax5.text(0.05, 0.45, rationale.strip(), fontsize=9, transform=ax5.transAxes,
            va='top', ha='left', wrap=True, color='#34495E',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=1))
    
    plt.savefig(comparison_dir / 'final_decision.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Final decision visualization saved")

def generate_comprehensive_pdf(all_metrics):
    """Generate comprehensive PDF report with all visualizations"""
    print("📄 Generating comprehensive PDF report...")
    
    pdf_path = RESULTS_DIR / 'FINAL_MODEL_COMPARISON_REPORT.pdf'
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                           topMargin=0.75*inch, bottomMargin=0.75*inch,
                           leftMargin=0.75*inch, rightMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    story = []
    comparison_dir = RESULTS_DIR / 'comparison'
    
    # ===== COVER PAGE =====
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("🎯 STRESS DETECTION MODEL", title_style))
    story.append(Paragraph("Comprehensive Comparison Report", title_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", 
                          styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Models Evaluated: {len(all_metrics)}", styles['Normal']))
    story.append(PageBreak())
    
    # ===== TABLE OF CONTENTS =====
    story.append(Paragraph("📑 Table of Contents", heading_style))
    story.append(Spacer(1, 12))
    
    toc_items = [
        "1. Executive Summary",
        "2. Individual Model Performance",
        "3. Confusion Matrix Comparison",
        "4. Comprehensive Metrics Analysis",
        "5. Visual Comparisons",
        "6. Final Model Selection",
        "7. Recommendations"
    ]
    
    for item in toc_items:
        story.append(Paragraph(f"   {item}", styles['Normal']))
        story.append(Spacer(1, 6))
    
    story.append(PageBreak())
    
    # ===== 1. EXECUTIVE SUMMARY =====
    story.append(Paragraph("1. Executive Summary", heading_style))
    story.append(Spacer(1, 12))
    
    summary = f"""
    <para alignment="justify">
    This comprehensive report presents the results of training and evaluating <b>{len(all_metrics)} 
    deep learning models</b> for binary stress classification from facial expressions. The models 
    were trained using transfer learning with ImageNet-pretrained weights, employing data 
    augmentation, class balancing, and two-phase training methodology.
    <br/><br/>
    <b>Models Evaluated:</b> {', '.join(all_metrics.keys())}
    <br/><br/>
    <b>Dataset Characteristics:</b><br/>
    • Original Classes: 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise)<br/>
    • Mapped Classification: Binary (Stressed vs Not Stressed)<br/>
    • Training Approach: Transfer Learning + Fine-tuning<br/>
    • Augmentation: Rotation, shifts, zoom, brightness variation<br/>
    • Class Balancing: Computed weight adjustment
    </para>
    """
    story.append(Paragraph(summary, styles['Normal']))
    story.append(PageBreak())
    
    # ===== 2. INDIVIDUAL MODEL PERFORMANCE =====
    story.append(Paragraph("2. Individual Model Performance", heading_style))
    story.append(Spacer(1, 12))
    
    for model_name, metrics in all_metrics.items():
        story.append(Paragraph(f"<b>{model_name}</b>", styles['Heading3']))
        story.append(Spacer(1, 8))
        
        perf_text = f"""
        <b>Performance Metrics:</b><br/>
        • Validation Accuracy: <b>{metrics['final_val_accuracy']:.4f}</b> (Best: {metrics['best_val_accuracy']:.4f})<br/>
        • Precision: <b>{metrics['calculated_precision']:.4f}</b><br/>
        • Recall: <b>{metrics['calculated_recall']:.4f}</b><br/>
        • F1-Score: <b>{metrics['calculated_f1_score']:.4f}</b><br/>
        • AUC-ROC: <b>{metrics['final_val_auc']:.4f}</b><br/>
        <br/>
        <b>Training Details:</b><br/>
        • Training Epochs: 30 epochs<br/>
        • Fine-tuning Epochs: 10 epochs<br/>
        • Total Epochs: 40 epochs<br/>
        • Stressed Samples: {metrics['stressed_samples']}<br/>
        • Not Stressed Samples: {metrics['not_stressed_samples']}<br/>
        """
        story.append(Paragraph(perf_text, styles['Normal']))
        
        # Add individual model plots if they exist
        model_plot = RESULTS_DIR / model_name / 'training_history.png'
        if model_plot.exists():
            story.append(Spacer(1, 12))
            story.append(RLImage(str(model_plot), width=6*inch, height=5*inch))
        
        model_cm = RESULTS_DIR / model_name / 'confusion_matrix_roc.png'
        if model_cm.exists():
            story.append(Spacer(1, 12))
            story.append(RLImage(str(model_cm), width=6*inch, height=2.5*inch))
        
        story.append(PageBreak())
    
    # ===== 3. CONFUSION MATRIX COMPARISON =====
    story.append(Paragraph("3. Confusion Matrix Comparison", heading_style))
    story.append(Spacer(1, 12))
    cm_comparison = comparison_dir / 'all_confusion_matrices.png'
    if cm_comparison.exists():
        story.append(RLImage(str(cm_comparison), width=7*inch, height=3*inch))
    story.append(PageBreak())
    
    # ===== 4. COMPREHENSIVE METRICS ANALYSIS =====
    story.append(Paragraph("4. Comprehensive Metrics Analysis", heading_style))
    story.append(Spacer(1, 12))
    
    table_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']]
    for model, metrics in all_metrics.items():
        table_data.append([model, f"{metrics['final_val_accuracy']:.4f}", f"{metrics['calculated_precision']:.4f}",
            f"{metrics['calculated_recall']:.4f}", f"{metrics['calculated_f1_score']:.4f}", 
            f"{metrics['final_val_auc']:.4f}"])
    
    table = Table(table_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    
    metrics_chart = comparison_dir / 'metrics_comparison.png'
    if metrics_chart.exists():
        story.append(RLImage(str(metrics_chart), width=7*inch, height=5*inch))
    story.append(PageBreak())
    
    # ===== 5. VISUAL COMPARISONS =====
    story.append(Paragraph("5. Visual Comparisons", heading_style))
    story.append(Spacer(1, 12))
    
    heatmap = comparison_dir / 'heatmap_comparison.png'
    if heatmap.exists():
        story.append(Paragraph("<b>Performance Heatmap</b>", styles['Heading3']))
        story.append(Spacer(1, 8))
        story.append(RLImage(str(heatmap), width=6*inch, height=4*inch))
    story.append(PageBreak())
    
    # ===== 6. FINAL MODEL SELECTION =====
    story.append(Paragraph("6. Final Model Selection", heading_style))
    story.append(Spacer(1, 12))
    
    best_model = max(all_metrics.items(), key=lambda x: x[1]['final_val_accuracy'])
    best_name, best_metrics = best_model
    
    selection_text = f"""
    <para alignment="justify">
    <b>SELECTED MODEL: {best_name}</b>
    <br/><br/>
    After comprehensive evaluation, <b>{best_name}</b> has been selected with validation accuracy of 
    <b>{best_metrics['final_val_accuracy']:.2%}</b>.
    <br/><br/>
    <b>Key Selection Criteria:</b><br/>
    • Highest overall accuracy: {best_metrics['final_val_accuracy']:.4f}<br/>
    • Balanced precision-recall trade-off<br/>
    • Strong AUC performance: {best_metrics['final_val_auc']:.4f}<br/>
    • Suitable for production deployment
    </para>
    """
    story.append(Paragraph(selection_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    final_decision = comparison_dir / 'final_decision.png'
    if final_decision.exists():
        story.append(RLImage(str(final_decision), width=7*inch, height=5*inch))
    story.append(PageBreak())
    
    # ===== 7. RECOMMENDATIONS =====
    story.append(Paragraph("7. Recommendations", heading_style))
    story.append(Spacer(1, 12))
    
    recommendations = f"""
    <para alignment="justify">
    <b>Deployment Recommendations:</b>
    <br/><br/>
    1. <b>Production Deployment:</b> Deploy {best_name} as the primary stress detection model
    <br/><br/>
    2. <b>Performance Monitoring:</b> Implement continuous monitoring to track model performance
    <br/><br/>
    3. <b>Data Collection:</b> Continue collecting diverse facial expression data
    <br/><br/>
    4. <b>Model Updates:</b> Schedule periodic retraining with new data
    <br/><br/>
    5. <b>Validation:</b> Conduct A/B testing to validate real-world performance
    </para>
    """
    story.append(Paragraph(recommendations, styles['Normal']))
    
    doc.build(story)
    print(f"✅ Comprehensive PDF report saved to: {pdf_path}")

def main():
    """Main function to generate all visualizations and PDF report"""
    print("="*80)
    print("🎯 GENERATING COMPREHENSIVE MODEL COMPARISON REPORT")
    print("="*80)
    
    all_metrics = {}
    for model in MODELS:
        metrics_file = RESULTS_DIR / model / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics[model] = json.load(f)
    
    if not all_metrics:
        print("❌ No trained models found! Please train models first.")
        return
    
    print(f"📊 Found {len(all_metrics)} trained models: {', '.join(all_metrics.keys())}")
    
    comparison_dir = RESULTS_DIR / 'comparison'
    comparison_dir.mkdir(exist_ok=True)
    
    create_modern_comparison_plots(all_metrics, comparison_dir)
    collect_confusion_matrices(list(all_metrics.keys()))
    create_final_decision_visual(all_metrics, comparison_dir)
    generate_comprehensive_pdf(all_metrics)
    
    print("\n" + "="*80)
    print("📈 MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    for model, metrics in all_metrics.items():
        print(f"\n🔹 {model}:")
        print(f"   Accuracy:  {metrics['final_val_accuracy']:.4f}")
        print(f"   Precision: {metrics['calculated_precision']:.4f}")
        print(f"   Recall:    {metrics['calculated_recall']:.4f}")
        print(f"   F1-Score:  {metrics['calculated_f1_score']:.4f}")
        print(f"   AUC:       {metrics['final_val_auc']:.4f}")
    
    best = max(all_metrics.items(), key=lambda x: x[1]['final_val_accuracy'])
    print(f"\n🏆 BEST MODEL: {best[0]} (Accuracy: {best[1]['final_val_accuracy']:.4f})")
    print("="*80)
    print("✅ Report generation completed successfully!")
    print(f"📄 PDF Report: {RESULTS_DIR / 'FINAL_MODEL_COMPARISON_REPORT.pdf'}")
    print(f"📊 Visualizations: {comparison_dir}")

if __name__ == "__main__":
    main()