"""
Contains functions for training and testing a PyTorch model.
"""
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import accuracy_score

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        loss_fn: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler,
        writer: SummaryWriter
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.scheduler = scheduler
        self.writer = writer
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_fn(output, targets)
        loss.backward()
        self.optimizer.step()
        output_label = output.argmax(dim=1)
        return loss.item(), ((output_label == targets).sum().item()/len(output))
        
        
    def _test_batch(self, source, targets):
        test_output = self.model(source)
        loss = self.loss_fn(test_output, targets)
        output = test_output.argmax(dim=1)
        return loss.item(), ((output == targets).sum().item()/len(test_output))
        

    def _run_epoch(self, epoch):
        train_loss, train_acc = 0, 0
        test_loss, test_acc = 0, 0
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | Time: {datetime.now()}")
        self.train_data.sampler.set_epoch(epoch)
        self.test_data.sampler.set_epoch(epoch)
        # Run one batch at the time
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss, accu = self._run_batch(source, targets)
            train_loss += loss
            train_acc += accu
        self.scheduler.step()
        train_loss = train_loss / len(self.train_data)
        train_acc = train_acc / len(self.train_data)
            
        if epoch % 2 == 0:
            self.model.eval()
            with torch.inference_mode():
                for source, targets in self.test_data:
                    source = source.to(self.gpu_id)
                    targets = targets.to(self.gpu_id)
                    loss, accu = self._test_batch(source, targets)
                    test_loss += loss
                    test_acc += accu
            test_loss = test_loss / len(self.test_data)
            test_acc = test_acc / len(self.test_data)
            if self.writer:
                self.writer.add_scalar("test/Loss", test_loss, epoch)
                self.writer.add_scalar("test/Accuracy", test_acc, epoch)
                
                
        if self.writer:
            self.writer.add_scalar("train/Loss", train_loss, epoch)
            self.writer.add_scalar("train/Accuracy", train_acc, epoch)
        else:
            pass

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
        # Close the writer
        self.writer.close()