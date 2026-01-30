import torch


def npu_fusion_attention_forward(query, key, value, head_num, input_layout, pse=None, padding_mask=None,
                                atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647, 
                                next_tockens=2147483647, inner_precise=0, prefix=None, actual_seq_qlen=None, 
                                actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False, 
                                softmax_layout="", sink=None):
    """
    Adjust the MLA (Multi-Head Attention) architecture adopted by the DeepSeek V3 model, 
    and update it to derive the tensor shape according to the input value tensor.
    """
    B = query.size(0)
    N = head_num
    S1 = query.size(2)
    S2 = key.size(2)

    if input_layout == "BSH":
        B = query.size(0)
        S1 = query.size(1)
        S2 = key.size(1)

    if input_layout == "SBH":
        B = query.size(1)
        S1 = query.size(0)
        S2 = key.size(0)

    if input_layout == "BSND":
        S1 = query.size(1)
        S2 = key.size(1)

    seed = 0
    offset = 0
    numels = 0
    attention_score = query.new_empty(value.shape, dtype=value.dtype, device='meta')
    softmax_max = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device='meta')
    softmax_sum = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device='meta')
    softmax_out = torch.empty([0], dtype=query.dtype, device='meta')
    return (attention_score, softmax_max, softmax_sum, softmax_out, seed, offset, numels)