"""
Test comprehensif untuk semua komponen Transformer.
============================================
"""

import numpy as np
import sys
from transformer import (
    TransformerConfig,
    TokenEmbedding,
    ScaledDotProductAttention,
    MultiHeadAttention,
    FeedForwardNetwork,
    TransformerBlock,
    Transformer,
    softmax,
    relu,
    gelu,
    layer_norm,
    get_sinusoidal_positional_encoding,
    create_causal_mask
)
from utils import (
    SimpleTokenizer,
    create_batch,
    calculate_accuracy,
    print_model_summary,
    check_shapes
)


class TestSuite:
    """Test suite untuk Transformer components."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        
    def run_test(self, test_name: str, test_func):
        """Run single test dengan error handling."""
        print(f"\n{'='*70}")
        print(f"TEST: {test_name}")
        print('='*70)
        
        try:
            test_func()
            print(f"✓ {test_name} PASSED")
            self.passed += 1
        except AssertionError as e:
            print(f"✗ {test_name} FAILED: {e}")
            self.failed += 1
            self.errors.append((test_name, str(e)))
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            self.failed += 1
            self.errors.append((test_name, str(e)))
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*70}")
        print("TEST SUMMARY")
        print('='*70)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")
        
        if self.errors:
            print(f"\nErrors:")
            for test_name, error in self.errors:
                print(f"  - {test_name}: {error}")
        else:
            print("\n🎉 All tests passed!")
        
        print('='*70)
        
        return self.failed == 0


# ============================================================================
# 1. UNIT TESTS
# ============================================================================

def test_softmax():
    """Test softmax function."""
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    
    # Check sum = 1
    assert np.isclose(np.sum(result), 1.0), f"Sum is {np.sum(result)}, expected 1.0"
    
    # Check all values in [0, 1]
    assert np.all(result >= 0) and np.all(result <= 1), "Values outside [0, 1]"
    
    # Check increasing input gives increasing output
    assert result[0] < result[1] < result[2], "Softmax not monotonic"
    
    print(f"  Input: {x}")
    print(f"  Output: {result}")
    print(f"  Sum: {np.sum(result)}")


def test_relu():
    """Test ReLU activation."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = relu(x)
    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    
    assert np.allclose(result, expected), f"ReLU failed: {result} != {expected}"
    
    print(f"  Input: {x}")
    print(f"  Output: {result}")


def test_gelu():
    """Test GELU activation."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = gelu(x)
    
    # GELU should be smooth and approximately linear for large positive x
    assert result[0] < result[1] < result[2] < result[3] < result[4], "GELU not monotonic"
    assert np.isclose(result[2], 0.0, atol=0.1), f"GELU(0) should be ~0, got {result[2]}"
    
    print(f"  Input: {x}")
    print(f"  Output: {result}")


def test_layer_norm():
    """Test layer normalization."""
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    gamma = np.ones(4)
    beta = np.zeros(4)
    
    result = layer_norm(x, gamma, beta)
    
    # Check mean ≈ 0
    assert np.isclose(np.mean(result), 0.0, atol=1e-5), f"Mean is {np.mean(result)}"
    
    # Check std ≈ 1
    assert np.isclose(np.std(result), 1.0, atol=0.1), f"Std is {np.std(result)}"
    
    print(f"  Input: {x}")
    print(f"  Output: {result}")
    print(f"  Mean: {np.mean(result):.6f}, Std: {np.std(result):.6f}")


def test_positional_encoding():
    """Test sinusoidal positional encoding."""
    seq_len = 10
    d_model = 8
    
    pe = get_sinusoidal_positional_encoding(seq_len, d_model)
    
    # Check shape
    assert pe.shape == (seq_len, d_model), f"Shape mismatch: {pe.shape}"
    
    # Check values are bounded
    assert np.all(pe >= -1.0) and np.all(pe <= 1.0), "Values outside [-1, 1]"
    
    # Check different positions have different encodings
    assert not np.allclose(pe[0], pe[1]), "Positions 0 and 1 have same encoding"
    
    print(f"  Shape: {pe.shape}")
    print(f"  First position: {pe[0][:4]}...")
    print(f"  Second position: {pe[1][:4]}...")


def test_causal_mask():
    """Test causal mask creation."""
    seq_len = 4
    mask = create_causal_mask(seq_len)
    
    # Check shape
    assert mask.shape == (1, 1, seq_len, seq_len), f"Shape mismatch: {mask.shape}"
    
    # Check upper triangle is masked
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                assert mask[0, 0, i, j] < -1e8, f"Position ({i},{j}) not masked"
            else:
                assert mask[0, 0, i, j] == 0, f"Position ({i},{j}) should not be masked"
    
    print(f"  Shape: {mask.shape}")
    print(f"  Mask matrix:\n{mask[0, 0]}")


def test_token_embedding():
    """Test token embedding layer."""
    vocab_size = 100
    d_model = 64
    batch_size = 2
    seq_len = 10
    
    embedding = TokenEmbedding(vocab_size, d_model)
    token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    output = embedding.forward(token_ids)
    
    # Check shape
    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
    
    # Check same token gives same embedding
    token_id = 5
    emb1 = embedding.forward(np.array([[token_id]]))
    emb2 = embedding.forward(np.array([[token_id]]))
    assert np.allclose(emb1, emb2), "Same token gives different embeddings"
    
    print(f"  Input shape: {token_ids.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Embedding matrix shape: {embedding.embedding.shape}")


def test_scaled_dot_product_attention():
    """Test scaled dot-product attention."""
    batch_size = 2
    n_heads = 4
    seq_len = 10
    d_k = 16
    
    attention = ScaledDotProductAttention(d_k)
    
    Q = np.random.randn(batch_size, n_heads, seq_len, d_k)
    K = np.random.randn(batch_size, n_heads, seq_len, d_k)
    V = np.random.randn(batch_size, n_heads, seq_len, d_k)
    
    output, attn_weights = attention.forward(Q, K, V)
    
    # Check output shape
    expected_output_shape = (batch_size, n_heads, seq_len, d_k)
    assert output.shape == expected_output_shape, f"Output shape mismatch: {output.shape}"
    
    # Check attention weights shape
    expected_attn_shape = (batch_size, n_heads, seq_len, seq_len)
    assert attn_weights.shape == expected_attn_shape, f"Attention shape mismatch: {attn_weights.shape}"
    
    # Check attention weights sum to 1
    attn_sums = np.sum(attn_weights, axis=-1)
    assert np.allclose(attn_sums, 1.0, atol=1e-5), f"Attention weights don't sum to 1"
    
    print(f"  Q, K, V shape: {Q.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(f"  Attention sums: {attn_sums[0, 0]}")


def test_multi_head_attention():
    """Test multi-head attention."""
    d_model = 64
    n_heads = 4
    batch_size = 2
    seq_len = 10
    
    mha = MultiHeadAttention(d_model, n_heads)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    output, attn_weights = mha.forward(x)
    
    # Check output shape
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} != {x.shape}"
    
    # Check attention weights shape
    expected_attn_shape = (batch_size, n_heads, seq_len, seq_len)
    assert attn_weights.shape == expected_attn_shape, f"Attention shape mismatch: {attn_weights.shape}"
    
    # Check with causal mask
    mask = create_causal_mask(seq_len)
    output_masked, attn_masked = mha.forward(x, mask)
    
    # Check that future positions are masked (attn weight = 0)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert np.allclose(attn_masked[:, :, i, j], 0, atol=1e-5), \
                f"Causal mask failed at position ({i},{j})"
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(f"  Causal masking: ✓")


def test_feed_forward_network():
    """Test feed-forward network."""
    d_model = 64
    d_ff = 256
    batch_size = 2
    seq_len = 10
    
    ffn = FeedForwardNetwork(d_model, d_ff)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    output = ffn.forward(x)
    
    # Check shape preservation
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    
    # Check non-linearity (output should be different from linear transformation)
    linear_output = np.matmul(x, ffn.W1) @ ffn.W2
    assert not np.allclose(output, linear_output, atol=0.1), "FFN behaves linearly"
    
    print(f"  Input shape: {x.shape}")
    print(f"  Hidden shape: [{batch_size}, {seq_len}, {d_ff}]")
    print(f"  Output shape: {output.shape}")


def test_transformer_block():
    """Test transformer block."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=50
    )
    
    block = TransformerBlock(config)
    
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, config.d_model)
    mask = create_causal_mask(seq_len)
    
    output, attn_weights = block.forward(x, mask)
    
    # Check shape preservation
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    
    # Check residual connection (output should be different from input but related)
    assert not np.allclose(output, x), "Output identical to input (no transformation)"
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")


def test_full_transformer():
    """Test full transformer model."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=50
    )
    
    model = Transformer(config)
    
    batch_size = 2
    seq_len = 10
    token_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, probs, attn_list = model.forward(token_ids, return_attention=True)
    
    # Check logits shape
    expected_logits_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_logits_shape, f"Logits shape mismatch: {logits.shape}"
    
    # Check probs shape
    expected_probs_shape = (batch_size, config.vocab_size)
    assert probs.shape == expected_probs_shape, f"Probs shape mismatch: {probs.shape}"
    
    # Check probabilities sum to 1
    prob_sums = np.sum(probs, axis=-1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5), f"Probabilities don't sum to 1: {prob_sums}"
    
    # Check probabilities in [0, 1]
    assert np.all(probs >= 0) and np.all(probs <= 1), "Probabilities outside [0, 1]"
    
    # Check attention list
    assert len(attn_list) == config.n_layers, f"Wrong number of attention layers: {len(attn_list)}"
    
    print(f"  Input shape: {token_ids.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Probs shape: {probs.shape}")
    print(f"  Prob sums: {prob_sums}")
    print(f"  Number of attention layers: {len(attn_list)}")


def test_causal_masking_in_model():
    """Test that causal masking works correctly in full model."""
    config = TransformerConfig(
        vocab_size=50,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=20
    )
    
    model = Transformer(config)
    
    batch_size = 1
    seq_len = 8
    token_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    
    _, _, attn_list = model.forward(token_ids, return_attention=True)
    
    # Check every layer's attention
    for layer_idx, attn in enumerate(attn_list):
        # attn shape: [batch, n_heads, seq_len, seq_len]
        for head_idx in range(config.n_heads):
            for i in range(seq_len):
                # Check that attention to future positions is 0
                future_attn = attn[0, head_idx, i, i+1:]
                assert np.allclose(future_attn, 0, atol=1e-5), \
                    f"Layer {layer_idx}, head {head_idx}, position {i} attends to future!"
    
    print(f"  Tested {config.n_layers} layers, {config.n_heads} heads")
    print(f"  All attention weights properly masked")


def test_generation():
    """Test autoregressive generation."""
    config = TransformerConfig(
        vocab_size=50,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=30
    )
    
    model = Transformer(config)
    
    # Generate from initial tokens
    initial_tokens = np.array([[1, 2, 3]])
    max_new_tokens = 5
    
    generated = model.generate(
        initial_tokens,
        max_new_tokens=max_new_tokens,
        temperature=1.0
    )
    
    # Check shape
    expected_len = initial_tokens.shape[1] + max_new_tokens
    assert generated.shape[1] == expected_len, \
        f"Generated length {generated.shape[1]} != expected {expected_len}"
    
    # Check that initial tokens are preserved
    assert np.array_equal(generated[0, :3], initial_tokens[0]), \
        "Initial tokens not preserved in generation"
    
    # Check that new tokens are valid
    assert np.all(generated >= 0) and np.all(generated < config.vocab_size), \
        "Generated tokens outside vocabulary range"
    
    print(f"  Initial: {initial_tokens[0]}")
    print(f"  Generated: {generated[0]}")


def test_weight_tying():
    """Test weight tying between embedding and output projection."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=50
    )
    
    model = Transformer(config)
    
    # Check that output_projection is transpose of embedding
    if model.use_weight_tying:
        expected_projection = model.token_embedding.embedding.T
        assert np.array_equal(model.output_projection, expected_projection), \
            "Weight tying not implemented correctly"
        
        print(f"  Weight tying: ✓ Enabled")
        print(f"  Embedding shape: {model.token_embedding.embedding.shape}")
        print(f"  Output projection shape: {model.output_projection.shape}")
    else:
        print(f"  Weight tying: ✗ Disabled")


def test_parameter_count():
    """Test parameter counting."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=50
    )
    
    model = Transformer(config)
    param_count = model.count_parameters()
    
    # Rough validation (should be > 0 and reasonable)
    assert param_count > 0, "Parameter count is 0"
    assert param_count < 1e9, f"Parameter count too large: {param_count}"
    
    print(f"  Total parameters: {param_count:,}")
    print(f"  Approximate size: {param_count * 4 / 1024 / 1024:.2f} MB (float32)")


def test_batch_processing():
    """Test that model handles different batch sizes correctly."""
    config = TransformerConfig(
        vocab_size=50,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=20
    )
    
    model = Transformer(config)
    seq_len = 8
    
    # Test different batch sizes
    for batch_size in [1, 2, 4, 8]:
        token_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
        logits, probs, _ = model.forward(token_ids)
        
        assert logits.shape[0] == batch_size, f"Batch size mismatch in logits"
        assert probs.shape[0] == batch_size, f"Batch size mismatch in probs"
    
    print(f"  Tested batch sizes: 1, 2, 4, 8")
    print(f"  All batch sizes handled correctly")


def test_sequence_lengths():
    """Test that model handles different sequence lengths."""
    config = TransformerConfig(
        vocab_size=50,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=50
    )
    
    model = Transformer(config)
    batch_size = 2
    
    # Test different sequence lengths
    for seq_len in [5, 10, 20, 40]:
        token_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
        logits, probs, _ = model.forward(token_ids)
        
        assert logits.shape[1] == seq_len, f"Seq len mismatch in logits"
    
    print(f"  Tested sequence lengths: 5, 10, 20, 40")
    print(f"  All sequence lengths handled correctly")


def test_determinism():
    """Test that model is deterministic (same input -> same output)."""
    config = TransformerConfig(
        vocab_size=50,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=20
    )
    
    model = Transformer(config)
    
    token_ids = np.random.randint(0, config.vocab_size, (2, 10))
    
    # Run twice
    logits1, probs1, _ = model.forward(token_ids)
    logits2, probs2, _ = model.forward(token_ids)
    
    # Should be identical
    assert np.allclose(logits1, logits2), "Model is not deterministic (logits)"
    assert np.allclose(probs1, probs2), "Model is not deterministic (probs)"
    
    print(f"  Same input produces same output: ✓")


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    config = TransformerConfig(
        vocab_size=50,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=20
    )
    
    model = Transformer(config)
    
    # Test with valid token IDs
    token_ids = np.random.randint(0, config.vocab_size, (2, 10))
    logits, probs, _ = model.forward(token_ids)
    
    # Check for NaN or Inf
    assert not np.any(np.isnan(logits)), "NaN detected in logits"
    assert not np.any(np.isinf(logits)), "Inf detected in logits"
    assert not np.any(np.isnan(probs)), "NaN detected in probs"
    assert not np.any(np.isinf(probs)), "Inf detected in probs"
    
    print(f"  No NaN or Inf values detected")
    print(f"  Numerically stable: ✓")


# ============================================================================
# 2. INTEGRATION TESTS
# ============================================================================

def test_tokenizer_integration():
    """Test tokenizer with model."""
    from utils import SimpleTokenizer
    
    # Create small vocabulary
    vocab_words = ['hello', 'world', 'this', 'is', 'a', 'test']
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(['hello world', 'this is a test'], min_freq=1)
    
    # Create model
    config = TransformerConfig(
        vocab_size=len(tokenizer),
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=20
    )
    
    model = Transformer(config)
    
    # Encode text
    text = "hello world"
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    token_ids_array = np.array([token_ids])
    
    # Forward pass
    logits, probs, _ = model.forward(token_ids_array)
    
    # Decode
    predicted_id = np.argmax(probs[0])
    predicted_token = tokenizer.id_to_token.get(predicted_id, '<unk>')
    
    print(f"  Input text: '{text}'")
    print(f"  Token IDs: {token_ids}")
    print(f"  Predicted next token ID: {predicted_id}")
    print(f"  Predicted next token: '{predicted_token}'")


def test_end_to_end_pipeline():
    """Test complete pipeline from text to prediction."""
    from utils import SimpleTokenizer, print_model_summary
    
    print("\n" + "="*70)
    print("END-TO-END PIPELINE TEST")
    print("="*70)
    
    # Step 1: Create tokenizer and vocabulary
    texts = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the bird flew over the tree"
    ]
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts, min_freq=1)
    print(f"\n1. Vocabulary: {len(tokenizer)} tokens")
    
    # Step 2: Create model
    config = TransformerConfig(
        vocab_size=len(tokenizer),
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=30
    )
    
    model = Transformer(config)
    print(f"\n2. Model created with {model.count_parameters():,} parameters")
    
    # Step 3: Encode text
    test_text = "the cat sat"
    token_ids = tokenizer.encode(test_text, add_special_tokens=True)
    token_ids_array = np.array([token_ids])
    print(f"\n3. Input: '{test_text}'")
    print(f"   Tokens: {token_ids}")
    
    # Step 4: Forward pass
    logits, probs, attn_list = model.forward(token_ids_array, return_attention=True)
    print(f"\n4. Forward pass complete")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Probs shape: {probs.shape}")
    
    # Step 5: Get prediction
    top_k = 3
    top_k_indices = np.argsort(probs[0])[-top_k:][::-1]
    top_k_probs = probs[0][top_k_indices]
    
    print(f"\n5. Top {top_k} predictions:")
    for idx, prob in zip(top_k_indices, top_k_probs):
        token = tokenizer.id_to_token.get(idx, '<unk>')
        print(f"   '{token}': {prob:.4f} ({prob*100:.2f}%)")
    
    # Step 6: Generate sequence
    generated = model.generate(token_ids_array, max_new_tokens=5, temperature=1.0)
    generated_text = tokenizer.decode(generated[0].tolist())
    
    print(f"\n6. Generated sequence:")
    print(f"   Token IDs: {generated[0]}")
    print(f"   Text: '{generated_text}'")
    
    print("\n" + "="*70)


# ============================================================================
# 3. MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TRANSFORMER IMPLEMENTATION - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    suite = TestSuite()
    
    # Unit tests - Basic functions
    print("\n" + "="*70)
    print("UNIT TESTS - BASIC FUNCTIONS")
    print("="*70)
    
    suite.run_test("Softmax", test_softmax)
    suite.run_test("ReLU", test_relu)
    suite.run_test("GELU", test_gelu)
    suite.run_test("Layer Normalization", test_layer_norm)
    suite.run_test("Positional Encoding", test_positional_encoding)
    suite.run_test("Causal Mask", test_causal_mask)
    
    # Unit tests - Components
    print("\n" + "="*70)
    print("UNIT TESTS - TRANSFORMER COMPONENTS")
    print("="*70)
    
    suite.run_test("Token Embedding", test_token_embedding)
    suite.run_test("Scaled Dot-Product Attention", test_scaled_dot_product_attention)
    suite.run_test("Multi-Head Attention", test_multi_head_attention)
    suite.run_test("Feed-Forward Network", test_feed_forward_network)
    suite.run_test("Transformer Block", test_transformer_block)
    
    # Unit tests - Full model
    print("\n" + "="*70)
    print("UNIT TESTS - FULL MODEL")
    print("="*70)
    
    suite.run_test("Full Transformer", test_full_transformer)
    suite.run_test("Causal Masking in Model", test_causal_masking_in_model)
    suite.run_test("Generation", test_generation)
    suite.run_test("Weight Tying", test_weight_tying)
    suite.run_test("Parameter Count", test_parameter_count)
    
    # Robustness tests
    print("\n" + "="*70)
    print("ROBUSTNESS TESTS")
    print("="*70)
    
    suite.run_test("Batch Processing", test_batch_processing)
    suite.run_test("Sequence Lengths", test_sequence_lengths)
    suite.run_test("Determinism", test_determinism)
    suite.run_test("Numerical Stability", test_numerical_stability)
    
    # Integration tests
    print("\n" + "="*70)
    print("INTEGRATION TESTS")
    print("="*70)
    
    suite.run_test("Tokenizer Integration", test_tokenizer_integration)
    suite.run_test("End-to-End Pipeline", test_end_to_end_pipeline)
    
    # Print summary
    all_passed = suite.print_summary()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()