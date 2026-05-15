import matplotlib.pyplot as plt

def plot_pareto_frontier():
    """Generates the Accuracy vs ECE Pareto Frontier for ARC Dataset"""
    plt.figure(figsize=(7, 5.5))
    plt.rcParams['font.family'] = 'serif'
    
    # Data vectors from your findings
    # Format: (ECE, Accuracy, Label)
    data = [
        (0.2157, 0.504, 'Baseline (Un-tuned)'),
        (0.2761, 0.566, 'QLoRA (r=8)'),
        (0.2536, 0.548, 'QLoRA (r=16)'),
        (0.2307, 0.532, 'QLoRA (r=32)'),
        (0.2379, 0.548, 'QLoRA (r=16 + T=0.5)')
    ]
    
    for ece, acc, label in data:
        color = 'red' if 'r=8' in label else ('green' if 'Baseline' in label else 'blue')
        marker = 'X' if 'T=' in label else 'o'
        plt.scatter(ece, acc, color=color, s=120, zorder=3, marker=marker)
        plt.text(ece + 0.002, acc + 0.002, label, fontsize=10, fontweight='medium')

    plt.xlabel('Expected Calibration Error (ECE) → [Lower is Better]', fontsize=11)
    plt.ylabel('Model Accuracy → [Higher is Better]', fontsize=11)
    plt.title('The Accuracy-Calibration Trade-off Frontier (ARC)', fontsize=12, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('arc_pareto_frontier.png', dpi=300)
    plt.close()

def plot_dual_axis_trends():
    """Generates the crossing-line trend plot showing Rank vs ECE/Acc"""
    ranks = [0, 8, 16, 32] # 0 represents Baseline
    ece_vals = [0.2157, 0.2761, 0.2536, 0.2307]
    acc_vals = [0.504, 0.566, 0.548, 0.532]
    
    fig, ax1 = plt.subplots(figsize=(7, 5))
    plt.rcParams['font.family'] = 'serif'
    
    # Left Axis: Accuracy
    color = 'tab:blue'
    ax1.set_xlabel('LoRA Rank ($r$)', fontsize=11)
    ax1.set_ylabel('Accuracy', color=color, fontsize=11)
    line1 = ax1.plot(ranks, acc_vals, color=color, marker='o', linewidth=2.5, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(ranks)
    ax1.set_xticklabels(['Baseline', 'r=8', 'r=16', 'r=32'])
    
    # Right Axis: ECE
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('ECE (Error)', color=color, fontsize=11)
    line2 = ax2.plot(ranks, ece_vals, color=color, marker='s', linewidth=2.5, linestyle='--', label='ECE')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add combined legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center')
    
    plt.title('Dynamic Impact of LoRA Rank Scale on Calibration vs Performance', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('rank_tradeoff_trends.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    plot_pareto_frontier()
    plot_dual_axis_trends()