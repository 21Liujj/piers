import os
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from utils.build_ddp import *
from utils.metrics import fast_hist, per_class_iu
from utils.callbacks import CheckpointCB, LRCB, LoggerCB
from utils.lr_scheduler import build_poly_warmup_scheduler
from train.strategies import WithMemStrategy, NoMemStrategy

class Trainer:
    def __init__(self, args, device, local_rank, world_size):
        self.args = args
        self.device = device
        self.local_rank = local_rank
        self.world_size = world_size

        self.train_loader = None
        self.val_loader = None
        self.train_sampler = None
        self.val_sampler = None

        class_set = getattr(self.args, "class_set", 13)
        # build_ddp resolves class names by class_set for supported datasets
        from utils.build_ddp import resolve_class_names
        self.class_names = resolve_class_names(self.args.dataset, class_set)
        self.num_classes = len(self.class_names)

        self.model = None
        self.criteria = None
        self.optimizer = None
        self.scheduler = None
        self.mem_strategy = None
        self.callbacks = None
        self.best_ckpt_path = None

        self.global_step = 0
        self.best_mIoU = 0.0

    def setup(self):
        data_builder = DatasetBuilder(self.args)
        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = data_builder.build_src()

        self.model = build_model(self.args, self.num_classes, self.args.num_maskmem, self.device, self.local_rank)
        self.mem_strategy = WithMemStrategy() if self.args.dataset != 'Stanford2D3D' else NoMemStrategy()
        if self.args.load_ckpt:
            load_checkpoint(self.model, self.args.load_ckpt)
        
        self.criteria = torch.nn.CrossEntropyLoss(ignore_index=255)
        if bool(getattr(self.args, "enable_sgdrf", False)):
            density_params = list(self.model.module.density_estimator.parameters())
            density_ids = {id(p) for p in density_params}
            main_params = [p for p in self.model.parameters() if id(p) not in density_ids]
            self.optimizer = torch.optim.AdamW([
                {"params": main_params, "lr": float(self.args.lr) * self.world_size},
                {"params": density_params, "lr": float(getattr(self.args, "sgdrf_density_lr", 5e-5)) * self.world_size},
            ], weight_decay=1e-4)
            self.model.module.sgdrf_grad_clip_norm = float(getattr(self.args, "sgdrf_grad_clip_norm", 1.0))
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr*self.world_size, weight_decay=1e-4)

        self.scheduler = build_poly_warmup_scheduler(self.optimizer, power=1.0, warmup_iters=100, total_iters=self.args.num_epochs*len(self.train_loader))

        class_set = getattr(self.args, "class_set", 13)
        save_dir = os.path.join(self.args.save_ckpt_dir, f"{self.args.dataset}_{self.args.backbone}_{class_set}cls")
        os.makedirs(save_dir, exist_ok=True)
        self.callbacks = [
            CheckpointCB(save_dir=save_dir, backbone=f"{self.args.backbone}_{class_set}cls"),
            LRCB(self.scheduler),
            LoggerCB()
        ]
    def train(self, epoch):
        self.model.train()
        if bool(getattr(self.args, "enable_sgdrf", False)):
            cold_epochs = int(getattr(self.args, "sgdrf_cold_start_epochs", 2))
            desired_stage = "cold_start" if epoch < cold_epochs else "joint"
            if self.model.module.sgdrf_training_stage != desired_stage:
                self.model.module.set_sgdrf_training_stage(desired_stage)
        self.train_sampler.set_epoch(epoch)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs} [Train]", disable=(self.local_rank != 0))
        for it, batch in enumerate(progress_bar):
            # Check iteration limits for ablation experiments
            max_iters = getattr(self.args, 'max_train_iters', -1)
            if max_iters > 0 and self.global_step >= max_iters:
                if self.local_rank == 0:
                    print(f'[ABLATION] Reached max_train_iters={max_iters}, stopping training loop')
                break
            self.optimizer.zero_grad()
            self.mem_strategy.forward(self.global_step, self.model, batch, self.criteria, self.device, self.optimizer, self.callbacks, self.local_rank)
            self.global_step += 1   
    
    def validate(self, epoch):
        self.model.eval()
        self.val_sampler.set_epoch(epoch)
        local_hist = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long, device=self.device)

        torch.cuda.synchronize(self.device)
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", disable=(self.local_rank != 0))
        with torch.no_grad():
            for idx, batch in enumerate(progress_bar):
                images, labels, *_ = batch
                images, labels = images.to(self.device), labels.to(self.device)
                frame_cnt = self.val_loader.dataset._get_frame_cnt()
                for frame_idx in range(frame_cnt):
                    preds = self.mem_strategy.forward_eval(self.model, images, frame_idx)
                    pred = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy()
                    label_np = labels[:, frame_idx].squeeze(0).cpu().numpy()
                    hist = fast_hist(label_np.flatten(), pred.flatten(), len(self.class_names))
                    hist_torch = torch.from_numpy(hist).to(self.device, dtype=torch.long)
                    local_hist += hist_torch
                self.mem_strategy.reset_memory(self.model)

        dist.all_reduce(local_hist, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(self.device)

        if self.local_rank == 0:
            global_hist_np = local_hist.cpu().numpy()
            mIoUs = per_class_iu(global_hist_np)
            for i, cname in enumerate(self.class_names):
                print(f"{cname:<15}: {round(mIoUs[i]*100, 2)}")
            cur_mIoU = round(np.nanmean(mIoUs)*100, 2)


            for cb in self.callbacks:
                if hasattr(cb, 'on_validation_end'):
                    cb.on_validation_end(epoch, cur_mIoU, self.model)

            if cur_mIoU > self.best_mIoU:
                self.best_mIoU = cur_mIoU
                self.best_ckpt_path = os.path.join(self.args.save_ckpt_dir, f"best_{self.args.backbone}_iou_{self.best_mIoU:.2f}.pth")
                torch.save(self.model.state_dict(), self.best_ckpt_path)
            print(f"Epoch {epoch+1}: val mIoU={cur_mIoU:.2f}, best={self.best_mIoU:.2f}")
                

