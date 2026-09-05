import torch

# `transformers.AdamW` was deprecated in transformers 4.x and removed in 4.5x/5.x.
# `torch.optim.AdamW` is the same algorithm — HF's version was a copy whose
# `correct_bias=True` default is exactly torch's behaviour — so this is a
# faithful drop-in, not a semantic change.
from torch.optim import AdamW


def get_bert_optim(network, lr, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    no_decay_params = []
    for n, p in network.named_parameters():
        if any(nd in n for nd in no_decay):
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8)
    return optimizer


def get_sgd_optim(network, lr, weight_decay):
    return torch.optim.SGD(
        network.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9)


get_optimizers = {
    "sgd": get_sgd_optim,
    "adamw": get_bert_optim
}
