from pytorch_lightning.metrics import Metric
import torch


class BaseMetric(Metric):
    def __init__(self, dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "loss", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )

    def update(self, loss: torch.Tensor):
        self.loss += loss
        self.total += 1

    def compute(self):
        return self.loss / self.total
