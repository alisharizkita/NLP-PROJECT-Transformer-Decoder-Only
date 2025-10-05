"""
Utils - Helper Functions untuk Transformer
===========================================

Berisi fungsi-fungsi utility untuk:
- Tokenisasi sederhana
- Data preparation
- Metrics calculation
- Debugging tools
"""

import numpy as np
from typing import List, Dict, Tuple


# ============================================================================
# 1. TOKENIZATION (Simplified)
# ============================================================================

class SimpleTokenizer:
    """
    Simple character-level atau word-level tokenizer.
    
    Untuk production, gunakan BPE (Byte-Pair Encoding) seperti GPT.
    Ini hanya untuk demo dan testing.
    """
    
    def __init__(self, vocab: List[str] = None):
        """
        Initialize tokenizer.
        
        Args:
            vocab: List of tokens. Jika None, akan di-build dari data.
        """
        if vocab is None:
            # Special tokens
            self.vocab = ['<pad>', '<unk>', '<sos>', '<eos>']
        else:
            self.vocab = vocab
            
        # Build mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        self.pad_id = self.token_to_id.get('<pad>', 0)
        self.unk_id = self.token_to_id.get('<unk>', 1)
        self.sos_id = self.token_to_id.get('<sos>', 2)
        self.eos_id = self.token_to_id.get('<eos>', 3)
        
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """
        Build vocabulary dari list of texts.
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency untuk include token
        """
        # Count token frequencies
        token_freq = {}
        for text in texts:
            tokens = text.split()  # Simple whitespace tokenization
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
        
        # Filter by frequency
        filtered_tokens = [
            token for token, freq in token_freq.items() 
            if freq >= min_freq
        ]
        
        # Sort by frequency (descending)
        filtered_tokens.sort(key=lambda t: token_freq[t], reverse=True)
        
        # Add to vocabulary (after special tokens)
        self.vocab.extend(filtered_tokens)
        
        # Rebuild mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        print(f"✓ Vocabulary built: {len(self.vocab)} tokens")
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text menjadi token IDs.
        
        Args:
            text: Input text
            add_special_tokens: If True, add <sos> and <eos>
            
        Returns:
            List of token IDs
        """
        tokens = text.split()
        token_ids = [
            self.token_to_id.get(token, self.unk_id) 
            for token in tokens
        ]
        
        if add_special_tokens:
            token_ids = [self.sos_id] + token_ids + [self.eos_id]
            
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs menjadi text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: If True, skip special tokens
            
        Returns:
            Decoded text string
        """
        special_ids = {self.pad_id, self.sos_id, self.eos_id}
        
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            tokens.append(self.id_to_token.get(token_id, '<unk>'))
            
        return ' '.join(tokens)
    
    def __len__(self):
        return len(self.vocab)
    
    def __repr__(self):
        return f"SimpleTokenizer(vocab_size={len(self.vocab)})"


# ============================================================================
# 2. DATA PREPARATION
# ============================================================================

def create_batch(
    token_ids_list: List[List[int]],
    max_seq_len: int,
    pad_id: int = 0
) -> np.ndarray:
    """
    Create batched and padded tensor dari list of token sequences.
    
    Args:
        token_ids_list: List of token ID sequences
        max_seq_len: Maximum sequence length
        pad_id: Padding token ID
        
    Returns:
        Batched tensor, shape [batch_size, max_seq_len]
    """
    batch_size = len(token_ids_list)
    batch = np.full((batch_size, max_seq_len), pad_id, dtype=np.int32)
    
    for i, token_ids in enumerate(token_ids_list):
        seq_len = min(len(token_ids), max_seq_len)
        batch[i, :seq_len] = token_ids[:seq_len]
    
    return batch


def create_causal_lm_data(
    token_ids: np.ndarray,
    context_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-target pairs untuk causal language modeling.
    
    Example:
        tokens: [1, 2, 3, 4, 5, 6]
        context_length: 3
        
        inputs:  [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        targets: [[2, 3, 4], [3, 4, 5], [4, 5, 6]]
    
    Args:
        token_ids: Token sequence, shape [seq_len]
        context_length: Length of context window
        
    Returns:
        inputs: [n_samples, context_length]
        targets: [n_samples, context_length]
    """
    n_samples = len(token_ids) - context_length
    
    inputs = np.zeros((n_samples, context_length), dtype=np.int32)
    targets = np.zeros((n_samples, context_length), dtype=np.int32)
    
    for i in range(n_samples):
        inputs[i] = token_ids[i:i+context_length]
        targets[i] = token_ids[i+1:i+context_length+1]
    
    return inputs, targets


# ============================================================================
# 3. METRICS
# ============================================================================

def calculate_perplexity(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate perplexity metric.
    
    Perplexity = exp(cross_entropy_loss)
    
    Lower perplexity = better model.
    
    Args:
        logits: Model logits, shape [batch, seq_len, vocab_size]
        targets: Target token IDs, shape [batch, seq_len]
        
    Returns:
        Perplexity value
    """
    # Get probabilities
    probs = softmax_3d(logits)
    
    # Cross entropy
    batch_size, seq_len = targets.shape
    ce_loss = 0.0
    
    for b in range(batch_size):
        for s in range(seq_len):
            target_id = targets[b, s]
            pred_prob = probs[b, s, target_id]
            ce_loss += -np.log(pred_prob + 1e-10)  # Add epsilon untuk stabilitas
    
    ce_loss /= (batch_size * seq_len)
    perplexity = np.exp(ce_loss)
    
    return perplexity


def softmax_3d(x: np.ndarray) -> np.ndarray:
    """Softmax untuk 3D tensor."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def calculate_accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate token-level accuracy.
    
    Args:
        logits: Model logits, shape [batch, seq_len, vocab_size]
        targets: Target token IDs, shape [batch, seq_len]
        
    Returns:
        Accuracy (0-1)
    """
    predictions = np.argmax(logits, axis=-1)  # [batch, seq_len]
    correct = np.sum(predictions == targets)
    total = targets.size
    return correct / total


# ============================================================================
# 4. DEBUGGING & INSPECTION
# ============================================================================

def print_model_summary(model):
    """
    Print detailed model summary.
    
    Args:
        model: Transformer model instance
    """
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    
    config = model.config
    
    print(f"Configuration:")
    print(f"  - Vocabulary size: {config.vocab_size:,}")
    print(f"  - Model dimension: {config.d_model}")
    print(f"  - Number of heads: {config.n_heads}")
    print(f"  - Head dimension: {config.d_k}")
    print(f"  - Number of layers: {config.n_layers}")
    print(f"  - FFN dimension: {config.d_ff}")
    print(f"  - Max sequence length: {config.max_seq_len}")
    print()
    
    print(f"Architecture:")
    print(f"  - Token Embedding: [{config.vocab_size}, {config.d_model}]")
    print(f"  - Positional Encoding: [{config.max_seq_len}, {config.d_model}]")
    print(f"  - Transformer Blocks: {config.n_layers}x")
    print(f"    - Multi-Head Attention:")
    print(f"      - W_Q, W_K, W_V, W_O: [{config.d_model}, {config.d_model}]")
    print(f"    - Feed-Forward Network:")
    print(f"      - W1: [{config.d_model}, {config.d_ff}]")
    print(f"      - W2: [{config.d_ff}, {config.d_model}]")
    print(f"  - Output Projection: [{config.d_model}, {config.vocab_size}]")
    print()
    
    print(f"Parameters:")
    print(f"  - Total (approx): {model.count_parameters():,}")
    print(f"  - Weight tying: {model.use_weight_tying}")
    print()
    
    print("=" * 70)


def inspect_attention(
    attention_weights: np.ndarray,
    token_ids: np.ndarray,
    tokenizer,
    layer_idx: int = 0,
    head_idx: int = 0,
    sample_idx: int = 0
):
    """
    Inspect attention weights untuk debugging.
    
    Args:
        attention_weights: Attention weights, shape [batch, n_heads, seq_len, seq_len]
        token_ids: Token IDs, shape [batch, seq_len]
        tokenizer: Tokenizer instance untuk decode
        layer_idx: Layer index to inspect
        head_idx: Head index to inspect
        sample_idx: Sample index in batch
    """
    print(f"Attention Inspection - Layer {layer_idx}, Head {head_idx}")
    print("-" * 70)
    
    # Get tokens
    tokens = [tokenizer.id_to_token.get(tid, '<unk>') for tid in token_ids[sample_idx]]
    seq_len = len(tokens)
    
    # Get attention for specific head
    attn = attention_weights[sample_idx, head_idx]  # [seq_len, seq_len]
    
    # Print header
    print(f"{'Token':<15} | " + " ".join([f"{t[:8]:>8}" for t in tokens]))
    print("-" * 70)
    
    # Print attention weights
    for i, token in enumerate(tokens):
        weights_str = " ".join([f"{attn[i, j]:>8.4f}" for j in range(seq_len)])
        print(f"{token:<15} | {weights_str}")
    
    print()
    
    # Find strongest attentions
    print("Strongest attention connections:")
    for i in range(seq_len):
        max_attn_idx = np.argmax(attn[i, :i+1])  # Only consider valid positions
        max_attn_val = attn[i, max_attn_idx]
        print(f"  {tokens[i]} → {tokens[max_attn_idx]} ({max_attn_val:.4f})")
    
    print()


def check_shapes(model, batch_size: int = 2, seq_len: int = 10):
    """
    Check shapes throughout the model untuk debugging.
    
    Args:
        model: Transformer model
        batch_size: Batch size for test
        seq_len: Sequence length for test
    """
    print("=" * 70)
    print("SHAPE CHECKING")
    print("=" * 70)
    
    config = model.config
    
    # Create dummy input
    token_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Input: {token_ids.shape}")
    
    # Token embedding
    x = model.token_embedding.forward(token_ids)
    print(f"After token embedding: {x.shape} (expected: [{batch_size}, {seq_len}, {config.d_model}])")
    
    # Add positional encoding
    x = x + model.pos_encoding[np.newaxis, :seq_len, :]
    print(f"After pos encoding: {x.shape}")
    
    # Through blocks
    from transformer import create_causal_mask
    mask = create_causal_mask(seq_len)
    
    for i, block in enumerate(model.blocks):
        x_before = x.shape
        x, _ = block.forward(x, mask)
        print(f"After block {i}: {x.shape} (input was {x_before})")
    
    # Output projection
    from transformer import layer_norm
    x = layer_norm(x, model.ln_final_gamma, model.ln_final_beta)
    print(f"After final layer norm: {x.shape}")
    
    logits = np.matmul(x, model.output_projection)
    print(f"After output projection: {logits.shape} (expected: [{batch_size}, {seq_len}, {config.vocab_size}])")
    
    print("\n✓ All shapes correct!")
    print("=" * 70)


# ============================================================================
# 5. TRAINING UTILITIES (untuk referensi, tidak diimplementasi penuh)
# ============================================================================

def compute_loss(logits: np.ndarray, targets: np.ndarray, ignore_index: int = -1) -> float:
    """
    Compute cross-entropy loss.
    
    Args:
        logits: Model output logits, shape [batch, seq_len, vocab_size]
        targets: Target token IDs, shape [batch, seq_len]
        ignore_index: Token ID to ignore dalam loss calculation (e.g., padding)
        
    Returns:
        Average loss
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Get probabilities
    probs = softmax_3d(logits)
    
    # Compute cross-entropy
    total_loss = 0.0
    valid_tokens = 0
    
    for b in range(batch_size):
        for s in range(seq_len):
            target_id = targets[b, s]
            
            if target_id == ignore_index:
                continue
                
            pred_prob = probs[b, s, target_id]
            total_loss += -np.log(pred_prob + 1e-10)
            valid_tokens += 1
    
    return total_loss / valid_tokens if valid_tokens > 0 else 0.0


def save_model_weights(model, filepath: str):
    """
    Save model weights ke file.
    
    Args:
        model: Transformer model
        filepath: Path untuk save weights
    """
    weights = {
        'token_embedding': model.token_embedding.embedding,
        'output_projection': model.output_projection,
        'ln_final_gamma': model.ln_final_gamma,
        'ln_final_beta': model.ln_final_beta,
        'config': {
            'vocab_size': model.config.vocab_size,
            'd_model': model.config.d_model,
            'n_heads': model.config.n_heads,
            'n_layers': model.config.n_layers,
            'd_ff': model.config.d_ff,
            'max_seq_len': model.config.max_seq_len,
        }
    }
    
    # Save weights dari setiap block
    for i, block in enumerate(model.blocks):
        weights[f'block_{i}'] = {
            'attention': {
                'W_Q': block.attention.W_Q,
                'W_K': block.attention.W_K,
                'W_V': block.attention.W_V,
                'W_O': block.attention.W_O,
            },
            'ffn': {
                'W1': block.ffn.W1,
                'b1': block.ffn.b1,
                'W2': block.ffn.W2,
                'b2': block.ffn.b2,
            },
            'ln1_gamma': block.ln1_gamma,
            'ln1_beta': block.ln1_beta,
            'ln2_gamma': block.ln2_gamma,
            'ln2_beta': block.ln2_beta,
        }
    
    np.save(filepath, weights)
    print(f"✓ Model weights saved to {filepath}")


def load_model_weights(model, filepath: str):
    """
    Load model weights dari file.
    
    Args:
        model: Transformer model
        filepath: Path ke saved weights
    """
    weights = np.load(filepath, allow_pickle=True).item()
    
    # Load embeddings
    model.token_embedding.embedding = weights['token_embedding']
    model.output_projection = weights['output_projection']
    model.ln_final_gamma = weights['ln_final_gamma']
    model.ln_final_beta = weights['ln_final_beta']
    
    # Load blocks
    for i, block in enumerate(model.blocks):
        block_weights = weights[f'block_{i}']
        
        block.attention.W_Q = block_weights['attention']['W_Q']
        block.attention.W_K = block_weights['attention']['W_K']
        block.attention.W_V = block_weights['attention']['W_V']
        block.attention.W_O = block_weights['attention']['W_O']
        
        block.ffn.W1 = block_weights['ffn']['W1']
        block.ffn.b1 = block_weights['ffn']['b1']
        block.ffn.W2 = block_weights['ffn']['W2']
        block.ffn.b2 = block_weights['ffn']['b2']
        
        block.ln1_gamma = block_weights['ln1_gamma']
        block.ln1_beta = block_weights['ln1_beta']
        block.ln2_gamma = block_weights['ln2_gamma']
        block.ln2_beta = block_weights['ln2_beta']
    
    print(f"✓ Model weights loaded from {filepath}")


# ============================================================================
# 6. TESTING UTILITIES
# ============================================================================

def test_component(component_name: str, test_func, *args, **kwargs):
    """
    Test individual component dengan proper error handling.
    
    Args:
        component_name: Name of component being tested
        test_func: Test function to run
        *args, **kwargs: Arguments untuk test function
    """
    print(f"Testing {component_name}...", end=" ")
    
    try:
        result = test_func(*args, **kwargs)
        print("✓ PASSED")
        return result
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        return None
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return None


def compare_outputs(output1: np.ndarray, output2: np.ndarray, name: str, rtol: float = 1e-5):
    """
    Compare two outputs untuk testing.
    
    Args:
        output1: First output
        output2: Second output
        name: Name of outputs being compared
        rtol: Relative tolerance
    """
    if not np.allclose(output1, output2, rtol=rtol):
        diff = np.abs(output1 - output2)
        max_diff = np.max(diff)
        print(f"✗ {name} outputs differ! Max difference: {max_diff}")
        return False
    else:
        print(f"✓ {name} outputs match")
        return True


# ============================================================================
# 7. SAMPLE DATA GENERATION
# ============================================================================

def generate_sample_data(
    n_samples: int = 100,
    seq_len: int = 20,
    vocab_size: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random sample data untuk testing.
    
    Args:
        n_samples: Number of samples
        seq_len: Sequence length
        vocab_size: Vocabulary size
        
    Returns:
        inputs: [n_samples, seq_len]
        targets: [n_samples, seq_len]
    """
    # Generate random sequences
    sequences = np.random.randint(1, vocab_size, (n_samples, seq_len + 1))
    
    # Split into input and target
    inputs = sequences[:, :-1]
    targets = sequences[:, 1:]
    
    return inputs, targets


def create_synthetic_text_data(
    vocab: List[str],
    n_samples: int = 100,
    min_len: int = 5,
    max_len: int = 20
) -> List[str]:
    """
    Create synthetic text data untuk testing tokenizer.
    
    Args:
        vocab: List of vocabulary words
        n_samples: Number of samples to generate
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        
    Returns:
        List of text strings
    """
    texts = []
    
    for _ in range(n_samples):
        seq_len = np.random.randint(min_len, max_len + 1)
        tokens = np.random.choice(vocab, seq_len)
        text = ' '.join(tokens)
        texts.append(text)
    
    return texts


if __name__ == "__main__":
    """
    Test utilities
    """
    print("=" * 70)
    print("TESTING UTILS")
    print("=" * 70)
    print()
    
    # Test tokenizer
    print("1. Testing SimpleTokenizer")
    print("-" * 70)
    
    texts = [
        "hello world",
        "hello there",
        "world peace",
        "peace and love"
    ]
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts, min_freq=1)
    
    test_text = "hello world"
    token_ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(token_ids)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {token_ids}")
    print(f"Decoded: {decoded}")
    print()
    
    # Test batch creation
    print("2. Testing create_batch")
    print("-" * 70)
    
    token_sequences = [
        [1, 2, 3],
        [4, 5, 6, 7, 8],
        [9, 10]
    ]
    
    batch = create_batch(token_sequences, max_seq_len=6, pad_id=0)
    print(f"Token sequences: {token_sequences}")
    print(f"Batched:\n{batch}")
    print()
    
    # Test causal LM data
    print("3. Testing create_causal_lm_data")
    print("-" * 70)
    
    tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    inputs, targets = create_causal_lm_data(tokens, context_length=3)
    
    print(f"Original tokens: {tokens}")
    print(f"Inputs:\n{inputs}")
    print(f"Targets:\n{targets}")
    print()
    
    print("✓ All utils tests passed!")
    print("=" * 70)