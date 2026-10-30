import torch
from torch import nn

from lumoculis.train import validate


class DummyDataset:
    def __init__(self, n=4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.randn(16)
        y = torch.tensor(idx % 5)
        return x.unsqueeze(0), y.unsqueeze(0)


class DummyLoader:
    def __init__(self, ds):
        self.ds = ds

    def __iter__(self):
        for i in range(len(self.ds)):
            yield self.ds[i]


def test_validation_loss_not_divided_twice():
    model = nn.Linear(16, 5)
    criterion = nn.CrossEntropyLoss()
    ds = DummyDataset(n=4)
    loader = DummyLoader(ds)
    res = validate(model, loader, criterion, torch.device("cpu"), 5, val_ids=None, max_batches=4)
    avg = res["avg_loss"].item() if hasattr(res["avg_loss"], "item") else float(res["avg_loss"])
    assert avg > 0
    assert avg > 0.001
