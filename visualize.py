"""
Visualization Tools untuk Transformer
======================================

Visualisasi untuk:
- Attention weights (heatmap)
- Positional encoding patterns
- Token embeddings (PCA/t-SNE)
- Model predictions

NOTE: Requires matplotlib. Install dengan: pip install matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional


# ============================================================================
# ATTENTION VISUALIZATION
# ============================================================================

def visualize_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: List[str],
    layer_idx: int = 0,
    head_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Visualize attention weights sebagai heatmap.
    
    Args:
        attention_weights: Attention weights, shape [batch, n_heads, seq_len, seq_len]
        tokens: List of token strings
        layer_idx: Layer index
        head_idx: Head index
        save_path: Path untuk save figure (optional)
    """
    # Extract attention untuk specific head
    attn = attention_weights[0, head_idx]  # [seq_len, seq_len]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(attn, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)
    
    # Labels
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    ax.set_title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}', 
                 fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Add values in cells
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if j <= i:  # Only show values in lower triangle (causal)
                text = ax.text(j, i, f'{attn[i, j]:.2f}',
                             ha="center", va="center", color="white" if attn[i, j] > 0.5 else "black",
                             fontsize=8)
    
    # Add grid
    ax.set_xticks(np.arange(len(tokens)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(tokens)+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Attention heatmap saved to {save_path}")
    
    plt.show()


def visualize_all_heads(
    attention_weights: np.ndarray,
    tokens: List[str],
    layer_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Visualize semua attention heads dalam satu figure.
    
    Args:
        attention_weights: Attention weights, shape [batch, n_heads, seq_len, seq_len]
        tokens: List of token strings
        layer_idx: Layer index
        save_path: Path untuk save figure
    """
    n_heads = attention_weights.shape[1]
    
    # Calculate grid size
    n_cols = 4
    n_rows = (n_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_heads > 1 else [axes]
    
    for head_idx in range(n_heads):
        ax = axes[head_idx]
        attn = attention_weights[0, head_idx]
        
        im = ax.imshow(attn, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Head {head_idx}', fontsize=10)
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        
        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_heads, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'All Attention Heads - Layer {layer_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ All heads visualization saved to {save_path}")
    
    plt.show()


def visualize_attention_flow(
    attention_weights: np.ndarray,
    tokens: List[str],
    layer_idx: int = 0,
    head_idx: int = 0,
    threshold: float = 0.1,
    save_path: Optional[str] = None
):
    """
    Visualize attention sebagai flow diagram (arrows).
    
    Args:
        attention_weights: Attention weights
        tokens: List of tokens
        layer_idx: Layer index
        head_idx: Head index
        threshold: Minimum attention weight untuk menampilkan arrow
        save_path: Save path
    """
    attn = attention_weights[0, head_idx]
    seq_len = len(tokens)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Token positions
    y_positions = np.arange(seq_len)[::-1]  # Top to bottom
    x_positions = np.zeros(seq_len)
    
    # Draw tokens
    for i, (token, y) in enumerate(zip(tokens, y_positions)):
        ax.text(0, y, token, fontsize=12, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black'))
    
    # Draw attention arrows
    for i in range(seq_len):
        for j in range(i+1):  # Only to previous tokens (causal)
            weight = attn[i, j]
            if weight > threshold:
                # Arrow from token j to token i
                y_from = y_positions[j]
                y_to = y_positions[i]
                
                # Arrow properties based on weight
                alpha = min(weight, 1.0)
                width = weight * 2
                
                ax.annotate('', xy=(0.5, y_to), xytext=(-0.5, y_from),
                          arrowprops=dict(arrowstyle='->', lw=width, alpha=alpha, color='red'))
                
                # Add weight label
                mid_y = (y_from + y_to) / 2
                ax.text(0, mid_y, f'{weight:.2f}', fontsize=8, ha='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1, seq_len)
    ax.axis('off')
    ax.set_title(f'Attention Flow - Layer {layer_idx}, Head {head_idx}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Attention flow diagram saved to {save_path}")
    
    plt.show()


# ============================================================================
# POSITIONAL ENCODING VISUALIZATION
# ============================================================================

def visualize_positional_encoding(
    pos_encoding: np.ndarray,
    max_positions: int = 50,
    save_path: Optional[str] = None
):
    """
    Visualize sinusoidal positional encoding.
    
    Args:
        pos_encoding: Positional encoding matrix, shape [seq_len, d_model]
        max_positions: Maximum positions to show
        save_path: Save path
    """
    pos_encoding = pos_encoding[:max_positions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Heatmap
    im1 = ax1.imshow(pos_encoding.T, cmap='RdBu', aspect='auto')
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('Dimension', fontsize=12)
    ax1.set_title('Positional Encoding Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: Line plot untuk beberapa dimensi
    dimensions_to_plot = [0, 1, 2, 3, 4, 5]
    for dim in dimensions_to_plot:
        if dim < pos_encoding.shape[1]:
            ax2.plot(pos_encoding[:, dim], label=f'Dim {dim}', alpha=0.7)
    
    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Positional Encoding Patterns', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Positional encoding visualization saved to {save_path}")
    
    plt.show()


# ============================================================================
# PREDICTION VISUALIZATION
# ============================================================================

def visualize_predictions(
    probs: np.ndarray,
    tokenizer,
    top_k: int = 10,
    save_path: Optional[str] = None
):
    """
    Visualize model predictions sebagai bar chart.
    
    Args:
        probs: Probability distribution, shape [vocab_size]
        tokenizer: Tokenizer untuk decode
        top_k: Number of top predictions to show
        save_path: Save path
    """
    # Get top k predictions
    top_k_indices = np.argsort(probs)[-top_k:][::-1]
    top_k_probs = probs[top_k_indices]
    top_k_tokens = [tokenizer.id_to_token.get(idx, '<unk>') for idx in top_k_indices]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, top_k))
    bars = ax.barh(range(top_k), top_k_probs, color=colors)
    
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_k_tokens)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title(f'Top {top_k} Predictions', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Add probability labels
    for i, (bar, prob) in enumerate(zip(bars, top_k_probs)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f' {prob:.4f} ({prob*100:.2f}%)',
               ha='left', va='center', fontsize=10)
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Predictions visualization saved to {save_path}")
    
    plt.show()


# ============================================================================
# MODEL ARCHITECTURE VISUALIZATION
# ============================================================================

def visualize_model_architecture(config, save_path: Optional[str] = None):
    """
    Visualize transformer architecture diagram.
    
    Args:
        config: TransformerConfig
        save_path: Save path
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    
    # Define positions
    y = 13
    box_width = 8
    box_height = 0.8
    x_center = 5
    
    def draw_box(y_pos, text, color='lightblue'):
        rect = mpatches.FancyBboxPatch(
            (x_center - box_width/2, y_pos), box_width, box_height,
            boxstyle="round,pad=0.1", edgecolor='black', facecolor=color, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x_center, y_pos + box_height/2, text, 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    def draw_arrow(y_from, y_to):
        ax.annotate('', xy=(x_center, y_to), xytext=(x_center, y_from),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Input
    draw_box(y, f'Input: Token IDs [batch, {config.max_seq_len}]', 'lightgreen')
    y -= 1.2
    draw_arrow(y+1.2, y)
    
    # Token Embedding
    draw_box(y, f'Token Embedding [{config.vocab_size}, {config.d_model}]', 'lightyellow')
    y -= 1.2
    draw_arrow(y+1.2, y)
    
    # Positional Encoding
    draw_box(y, f'+ Positional Encoding [{config.max_seq_len}, {config.d_model}]', 'lightyellow')
    y -= 1.2
    draw_arrow(y+1.2, y)
    
    # Transformer Blocks
    for i in range(config.n_layers):
        draw_box(y, f'Transformer Block {i+1}', 'lightcoral')
        y -= 0.5
        draw_box(y, f'  Multi-Head Attention (heads={config.n_heads})', 'lightpink')
        y -= 0.5
        draw_box(y, f'  Feed-Forward Network (d_ff={config.d_ff})', 'lightpink')
        y -= 1.2
        draw_arrow(y+1.2, y)
    
    # Final Layer Norm
    draw_box(y, f'Layer Normalization', 'lightblue')
    y -= 1.2
    draw_arrow(y+1.2, y)
    
    # Output Projection
    draw_box(y, f'Output Projection [{config.d_model}, {config.vocab_size}]', 'lightgreen')
    y -= 1.2
    draw_arrow(y+1.2, y)
    
    # Softmax
    draw_box(y, f'Softmax → Probabilities', 'lightgreen')
    
    ax.set_title('Transformer Architecture', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Architecture diagram saved to {save_path}")
    
    plt.show()


# ============================================================================
# DEMO FUNCTION
# ============================================================================

def demo_visualizations():
    """
    Demo semua visualizations.
    """
    print("=" * 70)
    print("VISUALIZATION DEMO")
    print("=" * 70)
    print()
    
    # Import required modules
    from transformer import Transformer, TransformerConfig, get_sinusoidal_positional_encoding
    from utils import SimpleTokenizer
    
    # Setup
    print("1. Setting up model and data...")
    config = TransformerConfig(
        vocab_size=50,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=30
    )
    
    model = Transformer(config)
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    texts = ["the cat sat on the mat", "the dog ran in the park"]
    tokenizer.build_vocab(texts, min_freq=1)
    
    # Prepare input
    test_text = "the cat sat on"
    token_ids = tokenizer.encode(test_text, add_special_tokens=False)
    token_ids_array = np.array([token_ids])
    tokens = test_text.split()
    
    print(f"   Text: '{test_text}'")
    print(f"   Tokens: {tokens}")
    print()
    
    # Forward pass
    print("2. Running forward pass...")
    logits, probs, attn_list = model.forward(token_ids_array, return_attention=True)
    print(f"   ✓ Forward pass complete")
    print()
    
    # Visualization 1: Attention Heatmap
    print("3. Visualizing attention heatmap...")
    visualize_attention_heatmap(
        attn_list[0], 
        tokens, 
        layer_idx=0, 
        head_idx=0,
        save_path='attention_heatmap.png'
    )
    
    # Visualization 2: All Heads
    print("\n4. Visualizing all attention heads...")
    visualize_all_heads(
        attn_list[0], 
        tokens, 
        layer_idx=0,
        save_path='all_heads.png'
    )
    
    # Visualization 3: Attention Flow
    print("\n5. Visualizing attention flow...")
    visualize_attention_flow(
        attn_list[0], 
        tokens, 
        layer_idx=0, 
        head_idx=0,
        threshold=0.1,
        save_path='attention_flow.png'
    )
    
    # Visualization 4: Positional Encoding
    print("\n6. Visualizing positional encoding...")
    pe = get_sinusoidal_positional_encoding(50, config.d_model)
    visualize_positional_encoding(
        pe,
        max_positions=50,
        save_path='positional_encoding.png'
    )
    
    # Visualization 5: Predictions
    print("\n7. Visualizing predictions...")
    visualize_predictions(
        probs[0],
        tokenizer,
        top_k=10,
        save_path='predictions.png'
    )
    
    # Visualization 6: Architecture
    print("\n8. Visualizing model architecture...")
    visualize_model_architecture(
        config,
        save_path='architecture.png'
    )
    
    print("\n" + "=" * 70)
    print("✓ All visualizations generated!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        demo_visualizations()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install matplotlib:")
        print("  pip install matplotlib")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()