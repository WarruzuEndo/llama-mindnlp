import torch
import mindspore


def torch_to_mindspore(ckpt_path, save_path):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    ms_ckpt = []
    for k, v in state_dict.items():
        if 'wq' in k:
            k = k.replace('wq', 'w_q')
            v = v.transpose(0, 1)
        if 'wk' in k:
            k = k.replace('wk', 'w_k')
            v = v.transpose(0, 1)
        if 'wv' in k:
            k = k.replace('wv', 'w_v')
            v = v.transpose(0, 1)
        if 'wo' in k:
            k = k.replace('wo', 'w_o')
            v = v.transpose(0, 1)
        if 'w1' in k:
            k = k.replace('w1', 'w_1')
            v = v.transpose(0, 1)
        if 'w2' in k:
            k = k.replace('w2', 'w_2')
            v = v.transpose(0, 1)
        if 'w3' in k:
            k = k.replace('w3', 'w_3')
            v = v.transpose(0, 1)
        if 'output' in k:
             v = v.transpose(0, 1)
        if 'rope' in k:
            continue
        ms_ckpt.append({'name': k, 'data': mindspore.Tensor(v.numpy())})
    mindspore.save_checkpoint(ms_ckpt, save_path)
    print('Done')


if __name__ == "__main__":
    ckpt_path = '../llama_checkpoint/13B/consolidated.01.pth'
    save_path = '13B/consolidated.01.ckpt'
    ckpt = torch_to_mindspore(ckpt_path, save_path)
