import torch

def accuracy(outputs, labels):
    num = len(labels)
    _, output_idx = torch.max(outputs, dim = 1)
    acc_list = [1 for i in range(num) if output_idx[i] == labels[i]]
    acc = sum(acc_list)/num
    print(acc)
    return acc
