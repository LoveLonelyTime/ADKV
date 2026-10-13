def KIVI_memory(batch, hidden_dim, seq_len, group_size=32, bits=4):
    key_states_quant_trans_bytes = batch * hidden_dim * seq_len * bits / 8
    key_scale_trans_bytes = batch * hidden_dim * seq_len / group_size * 2  # FP16
    key_mn_trans_bytes = batch * hidden_dim * seq_len / group_size * 2

    val_states_quant_trans_bytes = batch * hidden_dim * seq_len * bits / 8
    val_scale_trans_bytes = batch * hidden_dim * seq_len / group_size * 2
    val_mn_trans_bytes = batch * hidden_dim * seq_len / group_size * 2

    quant_bytes = key_states_quant_trans_bytes + val_states_quant_trans_bytes
    meta_bytes = key_scale_trans_bytes + key_mn_trans_bytes + \
        val_scale_trans_bytes + val_mn_trans_bytes

    return quant_bytes, meta_bytes


def MXFP4_memory(batch, hidden_dim, seq_len, group_size=32):
    bits = 4
    key_quant_bytes = batch * hidden_dim * seq_len * bits / 8
    key_scale_factor_bytes = batch * hidden_dim * \
        seq_len / group_size  # 8-bit common scale

    val_quant_bytes = batch * hidden_dim * seq_len * bits / 8
    val_scale_factor_bytes = batch * hidden_dim * \
        seq_len / group_size  # 8-bit common scale

    quant_bytes = key_quant_bytes + val_quant_bytes
    meta_bytes = key_scale_factor_bytes + val_scale_factor_bytes
    return quant_bytes, meta_bytes


def ADKV_memory(batch, hidden_dim, seq_len, bits=4):
    # To evaluate checkpoint mode for ADKV, set ckpt_num > 1
    ckpt_num = 1

    key_quant_bytes = batch * hidden_dim * seq_len * bits / 8
    key_zero_point_bytes = batch * hidden_dim * 2 * ckpt_num  # FP16
    key_scaling_factor_bytes = batch * hidden_dim * 2 * ckpt_num
    key_alpha_bytes = batch * hidden_dim * 2
    key_beta_bytes = batch * hidden_dim * 2
    key_target_bytes = batch * hidden_dim * 2

    val_quant_bytes = batch * hidden_dim * seq_len * bits / 8
    val_zero_point_bytes = batch * hidden_dim * 2 * ckpt_num
    val_scaling_factor_bytes = batch * hidden_dim * 2 * ckpt_num
    val_alpha_bytes = batch * hidden_dim * 2
    val_beta_bytes = batch * hidden_dim * 2
    val_target_bytes = batch * hidden_dim * 2

    # Value is same as key
    quant_bytes = key_quant_bytes + val_quant_bytes
    meta_bytes = key_zero_point_bytes + key_scaling_factor_bytes + key_alpha_bytes + key_beta_bytes + key_target_bytes + \
        val_zero_point_bytes + val_scaling_factor_bytes + \
        val_alpha_bytes + val_beta_bytes + val_target_bytes
    return quant_bytes, meta_bytes


def KVQuant_memory(batch, hidden_dim, seq_len, bits=4, outlier_ratio=0.01):
    """
    KVQuant (arxiv 2401.18079) memory breakdown.

    Quantization strategy:
      Keys   -> per-channel quantization  (one scale per channel, across ALL tokens)
      Values -> per-token   quantization  (one scale per token,   across ALL channels)

    Metadata (verified against KVQuant/benchmarking/scripts/test_kernels_{key,value}.py):
      1. Key scale + zero_point    : per-channel -> O(hidden_dim), NOT O(seq_len x hidden_dim)
                                     FP32 tensors, shape (hidden_dim,) each
      2. Value scale + zero_point  : per-token -> O(seq_len), NOT O(seq_len x hidden_dim)
                                     FP32 tensors, shape (seq_len,) each
      3. Key NUQ lookup table      : per-channel non-uniform quantization codebook
                                     shape (hidden_dim, 2^bits), FP32
                                     -> O(hidden_dim x 2^bits), independent of seq_len
      4. Value NUQ lookup table    : single shared NF4 signpost table
                                     shape (2^bits,), FP32 - negligible but included
      5. Key sparse outliers       : dense-sparse: outlier_ratio x seq_len tokens per channel
                                     stored as FP32 value + INT32 column-index per entry,
                                     plus INT32 start_rows pointer array of size O(seq_len)
      6. Value sparse outliers     : outlier_ratio x hidden_dim channels per token,
                                     stored as FP32 value + INT32 row-index per entry,
                                     plus INT32 start_cols pointer array of size O(hidden_dim)
    """
    # Quantized KV data
    key_quant_bytes = batch * hidden_dim * seq_len * bits / 8
    val_quant_bytes = batch * hidden_dim * seq_len * bits / 8

    # Scale + zero point metadata
    # Keys: per-channel -> one (scale, zero point) pair per (batch, channel), FP32
    key_scale_bytes = batch * hidden_dim * 2 * 4  # 2 x FP32 per channel

    # Values: per-token -> one (scale, zero point) pair per (batch, token), FP32
    val_scale_bytes = batch * seq_len * 2 * 4  # 2 x FP32 per token

    # Non-uniform quantization (NUQ) lookup tables
    # Keys: per-channel LUT; shape (hidden_dim, 2^bits), FP32
    #   (from: lookup_table = torch.zeros((num_heads, head_dim, 2**4)), test_kernels_key.py)
    key_lut_bytes = batch * hidden_dim * (2 ** bits) * 4   # FP32 per entry

    # Values: single shared NF4 signpost table; shape (2^bits,), FP32
    #   (from: lookup_table = torch.tensor(nf4_signposts).float(), test_kernels_value.py)
    # negligible; one shared table regardless of batch
    val_lut_bytes = (2 ** bits) * 4

    # Sparse outlier metadata (dense-sparse quantization)
    # Actual code uses FP32 vals and INT32 indices (torch default dtypes):
    #   rows2/cols2 = int32 (4 B each), vals2 = float32 (4 B)
    # Key COO: (col_inde x INT32 + val FP32) per nnz, plus start_rows INT32 array O(seq_len)
    key_nnz = batch * hidden_dim * seq_len * outlier_ratio
    # nnz x (idx + val) + start_rows
    key_outlier_bytes = key_nnz * (4 + 4) + batch * seq_len * 4

    # Value COO: (row_index INT32 + val FP32) per nnz, plus start_cols INT32 array O(hidden_dim)
    val_nnz = batch * hidden_dim * seq_len * outlier_ratio
    # nnz x (idx + val) + start_cols
    val_outlier_bytes = val_nnz * (4 + 4) + batch * hidden_dim * 4

    quant_bytes = key_quant_bytes + val_quant_bytes
    meta_bytes = (key_scale_bytes + val_scale_bytes
                  + key_lut_bytes + val_lut_bytes
                  + key_outlier_bytes + val_outlier_bytes)
    return quant_bytes, meta_bytes