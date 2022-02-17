import torch

#%%
import torch


def flat2matrix(flat, seq_lens):
    """
        Flat matrix to seq matrix
    """
    device = flat.device  # 输入序列的设备
    _, dim_size = flat.shape  # 输入序列的纬度
    dtype = flat.dtype  # 输入序列的类型
    padding = torch.zeros((1, dim_size), device=device, dtype=dtype)
    pad_flat = torch.cat((padding, flat), dim=0)
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens)
    copy_indexs = []
    start_pos = 1
    end_pos = 0
    for idx, s_len in enumerate(seq_lens):
        end_pos = start_pos + s_len
        copy_indexs.extend(
            list(range(start_pos, end_pos)) + [0] * (max_seq_len - s_len))
        start_pos = end_pos
    matrix = pad_flat.index_select(0, torch.tensor(
        copy_indexs, device=device)).reshape(batch_size, max_seq_len, dim_size)
    return matrix


def matrix2flat(matrix, seq_lens):
    if matrix is None:
        return None

    flat_x = torch.cat(
        [matrix[i][:seq_lens[i], :] for i in range(0, len(seq_lens))], dim=0)

    return flat_x


# %%
if __name__ == '__main__':
    flat = torch.rand((9, 5))
    seq_lens = [2, 4, 3]
    matirx = flat2matrix(flat, seq_lens)
    print(matirx)
    flat_cuda = flat.cuda()
    matirx = flat2matrix(flat_cuda, seq_lens)
    print(matirx)
# %%
