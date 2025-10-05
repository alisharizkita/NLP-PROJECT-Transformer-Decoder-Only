"""
Tugas Natural Language Processing : Implementasi lengkap decoder-only Transformer (GPT-style) tanpa library deep learning.
Hanya menggunakan NumPy untuk semua operasi matematis.

Rizkita Alisha Ramadhani
22/494942/TK/54347
"""
# 
import numpy as np
from typing import Tuple, Optional


# ============================================================================
# 1. KONFIGURASI MODEL
# ============================================================================

class TransformerConfig:
    """
    Konfigurasi hyperparameter untuk model Transformer.
    
    Attributes:
        vocab_size: Ukuran vocabulary (jumlah token unik)
        d_model: Dimensi embedding dan hidden state
        n_heads: Jumlah attention heads dalam multi-head attention
        n_layers: Jumlah transformer blocks yang di-stack
        d_ff: Dimensi hidden layer di feed-forward network (biasanya 4x d_model)
        max_seq_len: Panjang sequence maksimum yang bisa diproses
        dropout: Dropout rate (tidak diimplementasi di forward pass sederhana)
    """
    def __init__(
        self,
        vocab_size: int = 100,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 50,
        dropout: float = 0.1
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        # Validasi: d_model harus habis dibagi n_heads
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) harus habis dibagi n_heads ({n_heads})"
        
        self.d_k = d_model // n_heads  # Dimensi per head
        
    def __repr__(self):
        return (f"TransformerConfig(vocab_size={self.vocab_size}, "
                f"d_model={self.d_model}, n_heads={self.n_heads}, "
                f"n_layers={self.n_layers}, d_ff={self.d_ff})")


# ============================================================================
# 2. FUNGSI AKTIVASI DAN OPERASI DASAR
# ============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax function yang numerically stable.
    
    Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))

    """
    # Kurangi dengan max untuk stabilitas numerik
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU (Rectified Linear Unit) activation function.
    
    Formula: ReLU(x) = max(0, x)
    
    Args:
        x: Input array
        
    Returns:
        Array dengan nilai negatif di-set ke 0
        
    Example:
        >>> x = np.array([-1, 0, 1, 2])
        >>> relu(x)
        array([0, 0, 1, 2])
    """
    return np.maximum(0, x)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU (Gaussian Error Linear Unit) activation function.
    
    Formula (approximation): GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    GELU lebih smooth dibanding ReLU dan sering dipakai di Transformer modern.
    
    Args:
        x: Input array
        
    Returns:
        Array setelah aktivasi GELU
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(
    x: np.ndarray, 
    gamma: np.ndarray, 
    beta: np.ndarray, 
    eps: float = 1e-5
) -> np.ndarray:
    """
    Layer Normalization.
    
    Normalisasi dilakukan per sample, per position, across features.
    Berbeda dengan Batch Norm yang normalisasi across batch.
    
    """
    # Hitung mean dan variance di dimensi terakhir (feature dimension)
    mean = np.mean(x, axis=-1, keepdims=True)  # [..., 1]
    var = np.var(x, axis=-1, keepdims=True)     # [..., 1]
    
    # Normalize
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    # Scale and shift dengan learnable parameters
    return gamma * x_norm + beta


# ============================================================================
# 3. POSITIONAL ENCODING
# ============================================================================

def get_sinusoidal_positional_encoding(
    seq_len: int, 
    d_model: int
) -> np.ndarray:
    """
    Sinusoidal Positional Encoding dari paper "Attention Is All You Need".
    
    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Intuisi:
        - Setiap posisi mendapat encoding unik
        - Dimensi genap menggunakan sin, dimensi ganjil menggunakan cos
        - Frekuensi berbeda untuk setiap dimensi (dari cepat ke lambat)
        - Model bisa belajar relative position dengan mudah
    
    Keuntungan Sinusoidal:
        - Tidak perlu training
        - Bisa generalize ke sequence length yang belum pernah dilihat
        - Smooth interpolation untuk posisi yang belum dilihat
    
    """
    # Buat array posisi: [0, 1, 2, ..., seq_len-1]
    position = np.arange(seq_len)[:, np.newaxis]  # Shape: [seq_len, 1]
    
    # Hitung division term: 10000^(2i/d_model) untuk i = 0, 1, 2, ...
    # Ini menghasilkan frekuensi yang berbeda untuk setiap dimensi
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    )  # Shape: [d_model/2]
    
    # Initialize positional encoding matrix
    pe = np.zeros((seq_len, d_model))
    
    # Dimensi genap (0, 2, 4, ...) menggunakan sin
    pe[:, 0::2] = np.sin(position * div_term)
    
    # Dimensi ganjil (1, 3, 5, ...) menggunakan cos
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe


# ============================================================================
# 4. TOKEN EMBEDDING
# ============================================================================

class TokenEmbedding:
    """
    Token Embedding Layer.
    
    Mengkonversi token IDs (integers) menjadi dense vectors.
    Setiap token di vocabulary memiliki vektor embedding unik.
    
    Embedding matrix E memiliki shape [vocab_size, d_model]:
        - Setiap row adalah embedding untuk satu token
        - Token ID digunakan sebagai index untuk mengambil row
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize embedding matrix dengan random normal distribution.
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Initialize embedding matrix dengan Xavier/Glorot initialization
        # Scale: sqrt(1/d_model) untuk stabilitas training
        self.embedding = np.random.randn(vocab_size, d_model) * np.sqrt(1.0 / d_model)
        
    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Convert token IDs ke embedding vectors.
        """
        # Indexing sederhana: ambil row dari embedding matrix
        return self.embedding[token_ids]
    
    def __repr__(self):
        return f"TokenEmbedding(vocab_size={self.vocab_size}, d_model={self.d_model})"


# ============================================================================
# 5. SCALED DOT-PRODUCT ATTENTION
# ============================================================================

class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention Mechanism.
    
    Ini adalah core mechanism dari Transformer yang memungkinkan model
    untuk "fokus" pada bagian relevan dari input sequence.
    
    Formula:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Langkah-langkah:
        1. Hitung similarity scores: Q @ K^T (dot product)
        2. Scale dengan √d_k untuk stabilitas gradient
        3. Apply mask (untuk causal attention)
        4. Softmax untuk mendapat attention weights (probabilitas)
        5. Weighted sum dari values: attention_weights @ V
    
    Intuisi:
        - Q (Query): "Apa yang saya cari?"
        - K (Key): "Apa yang saya tawarkan?"
        - V (Value): "Apa isi informasi saya?"
        - Attention weights: Seberapa relevan setiap K dengan Q
    """
    
    def __init__(self, d_k: int):
        """
        Args:
            d_k: Dimensi key/query (untuk scaling)
        """
        self.d_k = d_k
        self.sqrt_d_k = np.sqrt(d_k)
        
    def forward(
        self, 
        Q: np.ndarray, 
        K: np.ndarray, 
        V: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query matrix, shape [batch, n_heads, seq_len, d_k]
            K: Key matrix, shape [batch, n_heads, seq_len, d_k]
            V: Value matrix, shape [batch, n_heads, seq_len, d_v]
            mask: Optional mask matrix, shape [1, 1, seq_len, seq_len]
                  Isi dengan -inf untuk posisi yang harus di-mask
        
        Returns:
            output: Attention output, shape [batch, n_heads, seq_len, d_v]
            attn_weights: Attention weights, shape [batch, n_heads, seq_len, seq_len]
        
        Shape Example:
            Q: [2, 4, 10, 16]  (batch=2, heads=4, seq_len=10, d_k=16)
            K: [2, 4, 10, 16]
            V: [2, 4, 10, 16]
            Output: [2, 4, 10, 16]
        """
        # Step 1: Compute attention scores
        # Q @ K^T menghasilkan similarity score antara semua query-key pairs
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2))  # [batch, n_heads, seq_len, seq_len]
        
        # Step 2: Scale scores
        # Scaling dengan √d_k mencegah dot product menjadi terlalu besar
        # untuk dimensi tinggi, yang bisa menyebabkan softmax saturated
        scores = scores / self.sqrt_d_k
        
        # Step 3: Apply mask (jika ada)
        # Mask digunakan untuk:
        #   - Causal masking: mencegah melihat token masa depan
        #   - Padding masking: mengabaikan padding tokens
        if mask is not None:
            scores = scores + mask  # Mask berisi -inf, jadi -inf + score = -inf
        
        # Step 4: Softmax untuk mendapat attention weights
        # Softmax mengubah scores menjadi probabilitas (sum=1 di setiap row)
        attn_weights = softmax(scores, axis=-1)  # [batch, n_heads, seq_len, seq_len]
        
        # Step 5: Weighted sum of values
        # Setiap output adalah weighted combination dari semua values
        # Weight ditentukan oleh attention weights
        output = np.matmul(attn_weights, V)  # [batch, n_heads, seq_len, d_v]
        
        return output, attn_weights
    
    def __repr__(self):
        return f"ScaledDotProductAttention(d_k={self.d_k})"


# ============================================================================
# 6. MULTI-HEAD ATTENTION
# ============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention Mechanism.
    
    Ide: Jalankan attention mechanism beberapa kali secara paralel (multi-head),
    masing-masing dengan parameter berbeda. Ini memungkinkan model untuk fokus
    pada aspek yang berbeda dari input.
    
    Architecture:
        Input → Linear(W_Q, W_K, W_V) → Split ke h heads → 
        Scaled Dot-Product Attention per head → Concat → Linear(W_O) → Output
    
    Formula:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
        where head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)
    """
    
    def __init__(self, d_model: int, n_heads: int):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Dimensi model (embedding dimension)
            n_heads: Jumlah attention heads
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimensi per head
        
        # Weight matrices untuk Q, K, V projections
        # Setiap matrix memproyeksikan dari d_model ke d_model
        # Tapi kemudian akan di-split menjadi n_heads bagian
        self.W_Q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_K = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_V = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        
        # Output projection matrix
        # Setelah concat semua heads, project kembali ke d_model
        self.W_O = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        
        # Attention module
        self.attention = ScaledDotProductAttention(self.d_k)
        
    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split tensor menjadi multiple heads.
        
        Args:
            x: Input tensor, shape [batch, seq_len, d_model]
            
        Returns:
            Reshaped tensor, shape [batch, n_heads, seq_len, d_k]
            
        Example:
            Input:  [2, 10, 64]  (batch=2, seq_len=10, d_model=64)
            Output: [2, 8, 10, 8]  (batch=2, n_heads=8, seq_len=10, d_k=8)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Reshape: [batch, seq_len, d_model] → [batch, seq_len, n_heads, d_k]
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Transpose: [batch, seq_len, n_heads, d_k] → [batch, n_heads, seq_len, d_k]
        return x.transpose(0, 2, 1, 3)
    
    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Combine multiple heads kembali menjadi single tensor.
        
        Args:
            x: Input tensor, shape [batch, n_heads, seq_len, d_k]
            
        Returns:
            Combined tensor, shape [batch, seq_len, d_model]
            
        Example:
            Input:  [2, 8, 10, 8]  (batch=2, n_heads=8, seq_len=10, d_k=8)
            Output: [2, 10, 64]    (batch=2, seq_len=10, d_model=64)
        """
        batch_size, n_heads, seq_len, d_k = x.shape
        
        # Transpose: [batch, n_heads, seq_len, d_k] → [batch, seq_len, n_heads, d_k]
        x = x.transpose(0, 2, 1, 3)
        
        # Reshape: [batch, seq_len, n_heads, d_k] → [batch, seq_len, d_model]
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def forward(
        self, 
        x: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor, shape [batch, seq_len, d_model]
            mask: Optional mask, shape [1, 1, seq_len, seq_len]
            
        Returns:
            output: Attention output, shape [batch, seq_len, d_model]
            attn_weights: Attention weights, shape [batch, n_heads, seq_len, seq_len]
        """
        batch_size = x.shape[0]
        
        # Step 1: Linear projections untuk Q, K, V
        # x @ W_Q menghasilkan queries untuk semua positions
        Q = np.matmul(x, self.W_Q)  # [batch, seq_len, d_model]
        K = np.matmul(x, self.W_K)  # [batch, seq_len, d_model]
        V = np.matmul(x, self.W_V)  # [batch, seq_len, d_model]
        
        # Step 2: Split menjadi multiple heads
        Q = self.split_heads(Q)  # [batch, n_heads, seq_len, d_k]
        K = self.split_heads(K)  # [batch, n_heads, seq_len, d_k]
        V = self.split_heads(V)  # [batch, n_heads, seq_len, d_k]
        
        # Step 3: Apply attention untuk setiap head secara paralel
        attn_output, attn_weights = self.attention.forward(Q, K, V, mask)
        # attn_output: [batch, n_heads, seq_len, d_k]
        # attn_weights: [batch, n_heads, seq_len, seq_len]
        
        # Step 4: Combine heads
        attn_output = self.combine_heads(attn_output)  # [batch, seq_len, d_model]
        
        # Step 5: Final linear projection
        output = np.matmul(attn_output, self.W_O)  # [batch, seq_len, d_model]
        
        return output, attn_weights
    
    def __repr__(self):
        return f"MultiHeadAttention(d_model={self.d_model}, n_heads={self.n_heads})"


# ============================================================================
# 7. FEED-FORWARD NETWORK
# ============================================================================

class FeedForwardNetwork:
    """
    Position-wise Feed-Forward Network.
    
    FFN sederhana dengan 2 linear layers dan aktivasi non-linear di tengah.
    Diterapkan secara independen ke setiap position.
    
    Architecture:
        Input → Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model) → Output
    
    Formula:
        FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2
    """
    
    def __init__(self, d_model: int, d_ff: int):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Input/output dimensionality
            d_ff: Hidden layer dimensionality (biasanya 4 * d_model)
        """
        self.d_model = d_model
        self.d_ff = d_ff
        
        # First linear layer: d_model → d_ff
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        
        # Second linear layer: d_ff → d_model
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of FFN.
        
        Args:
            x: Input tensor, shape [batch, seq_len, d_model]
            
        Returns:
            Output tensor, shape [batch, seq_len, d_model]
            
        Example:
            Input:  [2, 10, 64]
            Hidden: [2, 10, 256]  (after W1)
            Output: [2, 10, 64]   (after W2)
        """
        # First layer + activation
        hidden = np.matmul(x, self.W1) + self.b1  # [batch, seq_len, d_ff]
        hidden = gelu(hidden)  # Apply GELU activation
        
        # Second layer
        output = np.matmul(hidden, self.W2) + self.b2  # [batch, seq_len, d_model]
        
        return output
    
    def __repr__(self):
        return f"FeedForwardNetwork(d_model={self.d_model}, d_ff={self.d_ff})"


# ============================================================================
# 8. TRANSFORMER BLOCK
# ============================================================================

class TransformerBlock:
    """
    Single Transformer Block (Decoder Block).
    
    Terdiri dari:
        1. Multi-Head Self-Attention
        2. Feed-Forward Network
        3. Residual connections di kedua sub-layer
        4. Layer normalization (pre-norm architecture)
    """
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize transformer block.
        
        Args:
            config: Konfigurasi model
        """
        self.config = config
        
        # Multi-head attention
        self.attention = MultiHeadAttention(config.d_model, config.n_heads)
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(config.d_model, config.d_ff)
        
        # Layer normalization parameters (learnable)
        # Setiap LayerNorm punya gamma (scale) dan beta (shift) sendiri
        self.ln1_gamma = np.ones(config.d_model)
        self.ln1_beta = np.zeros(config.d_model)
        self.ln2_gamma = np.ones(config.d_model)
        self.ln2_beta = np.zeros(config.d_model)
        
    def forward(
        self, 
        x: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor, shape [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: Block output, shape [batch, seq_len, d_model]
            attn_weights: Attention weights dari attention layer
        """
        # ===== Sub-layer 1: Multi-Head Self-Attention =====
        
        # Pre-norm: LayerNorm sebelum attention
        attn_input = layer_norm(x, self.ln1_gamma, self.ln1_beta)
        
        # Multi-head attention
        attn_output, attn_weights = self.attention.forward(attn_input, mask)
        
        # Residual connection
        x = x + attn_output
        
        # ===== Sub-layer 2: Feed-Forward Network =====
        
        # Pre-norm: LayerNorm sebelum FFN
        ffn_input = layer_norm(x, self.ln2_gamma, self.ln2_beta)
        
        # Feed-forward network
        ffn_output = self.ffn.forward(ffn_input)
        
        # Residual connection
        output = x + ffn_output
        
        return output, attn_weights
    
    def __repr__(self):
        return f"TransformerBlock(d_model={self.config.d_model}, n_heads={self.config.n_heads})"


# ============================================================================
# 9. FULL TRANSFORMER MODEL
# ============================================================================

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal mask untuk decoder-only Transformer.
    
    Causal mask mencegah token pada posisi i untuk "melihat" token
    pada posisi j > i (token di masa depan).
    
    Implementasi:
        - Lower triangle (termasuk diagonal) = 0 (allowed)
        - Upper triangle = -inf (masked)
    """
    # Buat upper triangular matrix (k=1 artinya diagonal tidak termasuk)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    
    # Convert 1s menjadi -inf
    mask = mask * -1e9  # -1e9 ≈ -inf (cukup besar untuk softmax → 0)
    
    # Add batch and head dimensions
    return mask[np.newaxis, np.newaxis, :, :]  # [1, 1, seq_len, seq_len]


class Transformer:
    """
    Decoder-only Transformer Model (GPT-style).
    
    Architecture (top to bottom):
        1. Token Embedding + Positional Encoding
        2. Stack of N Transformer Blocks
        3. Final Layer Normalization
        4. Output Projection (Unembedding) ke vocabulary
        5. Softmax untuk probability distribution
    
    Forward Pass:
        token_ids → embeddings → transformer_blocks → logits → probabilities
    """
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize transformer model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        
        # ===== Input Layers =====
        
        # Token embedding: token_id → vector
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        
        # Positional encoding: posisi → vector
        # Pre-computed untuk max_seq_len positions
        self.pos_encoding = get_sinusoidal_positional_encoding(
            config.max_seq_len, 
            config.d_model
        )
        
        # ===== Transformer Blocks =====
        
        # Stack N transformer blocks
        self.blocks = [
            TransformerBlock(config) for _ in range(config.n_layers)
        ]
        
        # ===== Output Layers =====
        
        # Final layer normalization
        self.ln_final_gamma = np.ones(config.d_model)
        self.ln_final_beta = np.zeros(config.d_model)
        
        # Output projection (unembedding)
        # Weight tying: gunakan transpose dari embedding matrix
        # Ini mengurangi jumlah parameters dan sering meningkatkan performa
        self.use_weight_tying = True
        if self.use_weight_tying:
            # E^T: [d_model, vocab_size]
            self.output_projection = self.token_embedding.embedding.T
        else:
            # Atau buat weight matrix terpisah
            self.output_projection = np.random.randn(
                config.d_model, config.vocab_size
            ) * np.sqrt(1.0 / config.d_model)
        
        print(f"✓ Transformer initialized: {config}")
        print(f"  - Parameters (approx): {self.count_parameters():,}")
        print(f"  - Weight tying: {self.use_weight_tying}")
        
    def count_parameters(self) -> int:
        """
        Hitung jumlah parameters (approximation).
        
        Returns:
            Jumlah total parameters
        """
        config = self.config
        
        # Token embedding
        params = config.vocab_size * config.d_model
        
        # Per transformer block:
        # - Multi-head attention: 4 * (d_model * d_model) untuk W_Q, W_K, W_V, W_O
        # - FFN: 2 * (d_model * d_ff) untuk W1, W2
        # - Layer norm: 4 * d_model untuk 2 layer norms (gamma + beta)
        per_block = (
            4 * config.d_model * config.d_model +  # Attention
            2 * config.d_model * config.d_ff +      # FFN
            4 * config.d_model                       # LayerNorm
        )
        params += config.n_layers * per_block
        
        # Final layer norm
        params += 2 * config.d_model
        
        # Output projection (jika tidak weight tying)
        if not self.use_weight_tying:
            params += config.d_model * config.vocab_size
        
        return params
    
    def forward(
        self, 
        token_ids: np.ndarray,
        return_attention: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[list]]:
        """
        Forward pass of transformer.
        
        Args:
            token_ids: Input token IDs, shape [batch_size, seq_len]
            return_attention: If True, return attention weights dari setiap layer
            
        Returns:
            logits: Output logits, shape [batch, seq_len, vocab_size]
            probs: Probability distribution untuk next token, shape [batch, vocab_size]
            attention_weights: List of attention weights per layer (optional)
        """
        batch_size, seq_len = token_ids.shape
        
        # Validasi input
        assert seq_len <= self.config.max_seq_len, \
            f"Sequence length {seq_len} exceeds maximum {self.config.max_seq_len}"
        
        # ===== Step 1: Token Embedding =====
        # Convert token IDs menjadi dense vectors
        x = self.token_embedding.forward(token_ids)  # [batch, seq_len, d_model]
        
        # ===== Step 2: Add Positional Encoding =====
        # Tambahkan informasi posisi ke setiap token
        # Broadcasting: pos_encoding[None, :seq_len, :] akan broadcast ke batch dimension
        x = x + self.pos_encoding[np.newaxis, :seq_len, :]  # [batch, seq_len, d_model]
        
        # ===== Step 3: Create Causal Mask =====
        # Mask untuk mencegah melihat future tokens
        mask = create_causal_mask(seq_len)  # [1, 1, seq_len, seq_len]
        
        # ===== Step 4: Pass through Transformer Blocks =====
        attention_weights_list = []
        
        for i, block in enumerate(self.blocks):
            x, attn_weights = block.forward(x, mask)
            # x: [batch, seq_len, d_model]
            # attn_weights: [batch, n_heads, seq_len, seq_len]
            
            if return_attention:
                attention_weights_list.append(attn_weights)
        
        # ===== Step 5: Final Layer Normalization =====
        x = layer_norm(x, self.ln_final_gamma, self.ln_final_beta)
        
        # ===== Step 6: Output Projection (Unembedding) =====
        # Project dari d_model ke vocab_size
        logits = np.matmul(x, self.output_projection)  # [batch, seq_len, vocab_size]
        
        # ===== Step 7: Get Probability Distribution =====
        # Untuk next token prediction, kita hanya perlu logits dari posisi terakhir
        last_logits = logits[:, -1, :]  # [batch, vocab_size]
        probs = softmax(last_logits, axis=-1)  # [batch, vocab_size]
        
        if return_attention:
            return logits, probs, attention_weights_list
        else:
            return logits, probs, None
    
    def generate(
        self, 
        initial_tokens: np.ndarray,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate new tokens autoregressively.
        
        Args:
            initial_tokens: Starting tokens, shape [batch, seq_len]
            max_new_tokens: Jumlah token baru yang akan di-generate
            temperature: Sampling temperature (>1 = more random, <1 = more deterministic)
            top_k: If set, only sample from top k most likely tokens
            
        Returns:
            Generated tokens, shape [batch, seq_len + max_new_tokens]
        """
        tokens = initial_tokens.copy()
        
        for _ in range(max_new_tokens):
            # Forward pass
            _, probs, _ = self.forward(tokens)  # probs: [batch, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
                probs = probs / np.sum(probs, axis=-1, keepdims=True)
            
            # Top-k sampling
            if top_k is not None:
                # Zero out probabilities for tokens not in top-k
                top_k_indices = np.argsort(probs, axis=-1)[:, -top_k:]
                mask = np.zeros_like(probs)
                np.put_along_axis(mask, top_k_indices, 1, axis=-1)
                probs = probs * mask
                probs = probs / np.sum(probs, axis=-1, keepdims=True)
            
            # Sample next token
            next_token = np.array([
                np.random.choice(self.config.vocab_size, p=probs[i])
                for i in range(probs.shape[0])
            ])[:, np.newaxis]  # [batch, 1]
            
            # Append to sequence
            tokens = np.concatenate([tokens, next_token], axis=1)
            
            # Check max sequence length
            if tokens.shape[1] >= self.config.max_seq_len:
                break
        
        return tokens
    
    def __repr__(self):
        return (f"Transformer(\n"
                f"  config={self.config},\n"
                f"  n_blocks={len(self.blocks)},\n"
                f"  parameters={self.count_parameters():,}\n"
                f")")


# ============================================================================
# 10. MAIN: DEMO DAN TESTING
# ============================================================================

def main():
    """
    Main function untuk demonstrasi dan testing.
    """
    print("=" * 70)
    print("TRANSFORMER FROM SCRATCH - DEMO")
    print("=" * 70)
    print()
    
    # ===== Setup Configuration =====
    print("1. SETUP CONFIGURATION")
    print("-" * 70)
    
    config = TransformerConfig(
        vocab_size=100,      # Small vocabulary untuk demo
        d_model=64,          # Embedding dimension
        n_heads=4,           # 4 attention heads
        n_layers=2,          # 2 transformer blocks
        d_ff=256,            # FFN hidden dimension
        max_seq_len=50       # Maximum sequence length
    )
    
    print(config)
    print()
    
    # ===== Initialize Model =====
    print("2. INITIALIZE MODEL")
    print("-" * 70)
    
    model = Transformer(config)
    print()
    
    # ===== Prepare Input =====
    print("3. PREPARE INPUT")
    print("-" * 70)
    
    batch_size = 2
    seq_len = 10
    
    # Random token IDs (simulasi input)
    np.random.seed(42)
    token_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {token_ids.shape}")
    print(f"Input token IDs:\n{token_ids}")
    print()
    
    # ===== Forward Pass =====
    print("4. FORWARD PASS")
    print("-" * 70)
    
    logits, probs, attention_weights = model.forward(
        token_ids, 
        return_attention=True
    )
    
    print(f"✓ Logits shape: {logits.shape} (expected: [{batch_size}, {seq_len}, {config.vocab_size}])")
    print(f"✓ Probs shape: {probs.shape} (expected: [{batch_size}, {config.vocab_size}])")
    print(f"✓ Number of attention layers: {len(attention_weights)}")
    print()
    
    # ===== Validation Tests =====
    print("5. VALIDATION TESTS")
    print("-" * 70)
    
    # Test 1: Dimensi check
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Logits shape mismatch!"
    assert probs.shape == (batch_size, config.vocab_size), "Probs shape mismatch!"
    print("✓ Test 1 PASSED: Dimensi output benar")
    
    # Test 2: Probability constraints
    prob_sums = np.sum(probs, axis=-1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5), "Probabilities tidak sum to 1!"
    assert np.all(probs >= 0) and np.all(probs <= 1), "Probabilities di luar range [0,1]!"
    print(f"✓ Test 2 PASSED: Probabilities valid (sum={prob_sums[0]:.6f})")
    
    # Test 3: Attention weights
    for i, attn in enumerate(attention_weights):
        attn_sum = np.sum(attn, axis=-1)
        assert np.allclose(attn_sum, 1.0, atol=1e-5), f"Attention weights layer {i} tidak sum to 1!"
    print(f"✓ Test 3 PASSED: Attention weights sum to 1 untuk semua layers")
    
    # Test 4: Causal masking
    # Check apakah attention weight untuk future positions adalah 0
    for i, attn in enumerate(attention_weights):
        # attn: [batch, n_heads, seq_len, seq_len]
        for pos in range(seq_len):
            future_attn = attn[:, :, pos, pos+1:]  # Attention ke future tokens
            assert np.allclose(future_attn, 0, atol=1e-5), \
                f"Causal mask gagal di layer {i}, position {pos}!"
    print("✓ Test 4 PASSED: Causal masking bekerja dengan benar")
    
    # Test 5: Gradient of logits (non-zero)
    assert not np.all(logits == 0), "Logits semua nol!"
    print("✓ Test 5 PASSED: Model menghasilkan output non-trivial")
    
    print()
    
    # ===== Prediction Analysis =====
    print("6. PREDICTION ANALYSIS")
    print("-" * 70)
    
    for b in range(batch_size):
        top_5_indices = np.argsort(probs[b])[-5:][::-1]
        top_5_probs = probs[b][top_5_indices]
        
        print(f"Batch {b} - Top 5 predictions:")
        for idx, prob in zip(top_5_indices, top_5_probs):
            print(f"  Token {idx:3d}: {prob:.4f} ({prob*100:.2f}%)")
        print()
    
    # ===== Attention Visualization Info =====
    print("7. ATTENTION WEIGHTS INFO")
    print("-" * 70)
    
    for i, attn in enumerate(attention_weights):
        avg_attn = np.mean(attn[0, :, -1, :])  # Average attention dari last token
        max_attn_head = np.argmax(np.mean(attn[0, :, -1, :].reshape(config.n_heads, -1), axis=1))
        
        print(f"Layer {i+1}:")
        print(f"  - Attention shape: {attn.shape}")
        print(f"  - Avg attention (last token): {avg_attn:.4f}")
        print(f"  - Most active head: {max_attn_head}")
    
    print()
    
    # ===== Generation Demo =====
    print("8. TEXT GENERATION DEMO")
    print("-" * 70)
    
    initial_tokens = np.array([[1, 2, 3]])  # Start dengan 3 tokens
    generated = model.generate(
        initial_tokens, 
        max_new_tokens=7,
        temperature=1.0
    )
    
    print(f"Initial tokens: {initial_tokens[0]}")
    print(f"Generated tokens: {generated[0]}")
    print()
    
    # ===== Summary =====
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Model successfully implemented with {model.count_parameters():,} parameters")
    print(f"✓ All validation tests passed")
    print(f"✓ Forward pass working correctly")
    print(f"✓ Causal masking verified")
    print(f"✓ Attention mechanism functioning")
    print(f"✓ Generation capability demonstrated")
    print()
    print("🎉 Transformer implementation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()