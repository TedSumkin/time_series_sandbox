from torch import nn


class pointwiseMSE(nn.Module):
    def __init__(self):
        super(
            pointwiseMSE,
        ).__init__()
        self.mse = nn.MSELoss(reduce=False)

    def forward(self, y_pred, y_true):
        return self.mse(y_pred, y_true)


class pointwiseMAE(nn.Module):
    def __init__(self):
        super(
            pointwiseMAE,
        ).__init__()
        self.mae = nn.L1Loss(reduce=False)

    def forward(self, y_pred, y_true):
        return self.mae(y_pred, y_true)
