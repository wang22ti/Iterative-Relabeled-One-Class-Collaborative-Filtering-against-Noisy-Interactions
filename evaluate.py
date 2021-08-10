import torch
import math
import heapq  # for retrieval topK

from model import *


def evaluate(model: BPR, test_users, test_data, klist):
    num_user, num_item = len(test_users), test_data.num_item
    mask = torch.zeros([num_user, num_item])
    for user in range(num_user):
        mask[user].scatter_(dim=0, index=torch.tensor(list(test_data.test_items[user])),
                            value=torch.tensor(1.0))
    result = torch.sigmoid(torch.mm(model.U[test_users], model.V.t()))
    result = torch.mul(mask, result)
    _, result = torch.topk(result, k=max(klist), dim=1)

    metrics = [0 for _ in range(len(klist) * 3 + 1)]  # p * 3 + r * 3 + ndcg * 3 + auc
    for user in test_users:
        pos = test_data.test_pos_items[user]
        if len(pos) == 0:
            continue
        pred = result[user, :].numpy().tolist()
        labels = [1 if item in pos else 0 for item in pred]
        for idx, k in enumerate(klist):
            metrics[idx] += get_p(labels[:k])
            metrics[3 + idx] += get_r(labels[:k], len(pos))
            metrics[6 + idx] += get_ndcg(labels[:k])
        metrics[-1] += get_auc(labels, len(pos))

    return [metric / len(test_users) for metric in metrics]

def get_ndcg(labels):
    dcg, max_dcg = 0, 0
    for i, label in enumerate(labels):
        dcg += label / math.log2(i + 2)
        max_dcg += 1 / math.log2(i + 2)
    return dcg / max_dcg


def get_p(labels):
    return sum(labels) / len(labels)


def get_r(labels, num_pos):
    return sum(labels) / num_pos

def get_auc(labels, num_pos):
    auc = 0
    for i, label in enumerate(labels[::-1]):
        auc += label * (i + 1)

    return (auc - num_pos * (num_pos + 1) / 2) / (num_pos * (len(labels) - num_pos))

