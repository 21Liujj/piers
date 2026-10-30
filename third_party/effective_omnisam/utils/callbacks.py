import os
import torch
try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None
from torch.utils.tensorboard import SummaryWriter

from utils.io import save_checkpoint

class CheckpointCB:
    def __init__(self, save_dir, backbone, use_wandb: bool = False, tb_writer: SummaryWriter | None = None):
        self.save_dir = save_dir
        self.backbone = backbone
        self.use_wandb = bool(use_wandb)
        self.tb_writer = tb_writer

    def on_validation_end(self, epoch, metric, model, rank=0):
        if rank == 0:
            log_dict = {"mIoU": metric}
            if self.use_wandb and wandb is not None and getattr(wandb, "run", None) is not None:
                wandb.log(log_dict, step=epoch)
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("val/mIoU", float(metric), int(epoch))
            path_last = os.path.join(self.save_dir, f'last_{self.backbone}.pth')
            torch.save(model.state_dict(), path_last)

class LRCB:
    def __init__(self, scheduler):
        self.scheduler = scheduler
    def on_iter_end(self, global_step, loss, lr, rank=0, metrics=None, **kwargs):
        if self.scheduler is not None:
            self.scheduler.step()

class LoggerCB:
    def __init__(self, use_wandb: bool = False, tb_writer: SummaryWriter | None = None):
        self.use_wandb = bool(use_wandb)
        self.tb_writer = tb_writer

    def on_iter_end(self, global_step, loss, lr, rank=0, metrics=None, **kwargs):
        if rank == 0:
            log_dict = {"loss": loss, "lr": lr}
            if metrics is not None:
                log_dict.update(metrics)
            if self.use_wandb and wandb is not None and getattr(wandb, "run", None) is not None:
                wandb.log(log_dict, step=global_step)
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("train/loss", float(loss), int(global_step))
                self.tb_writer.add_scalar("train/lr", float(lr), int(global_step))
                if metrics is not None:
                    for k, v in metrics.items():
                        try:
                            self.tb_writer.add_scalar(f"train/{k}", float(v), int(global_step))
                        except Exception:
                            continue