from torch import nn


class pointwiseMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, y_pred, y_true):
        return self.mse(y_pred, y_true)


class pointwiseMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, y_pred, y_true):
        return self.mae(y_pred, y_true)
