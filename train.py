import os
import torch
import numpy as np
import config
from ModelFrame import Model
from config import ModelConfig
from data_loader import build_passenger_flow_dataloader
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
Device = config.Device
print(Device)

if __name__ == "__main__":
    train_loader, test_loader, adj_tensor = build_passenger_flow_dataloader(
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        device=config.device,
        window_size=config.window_size,
        step_size=config.step_size,
        train_ratio=config.train_ratio,
    )

    train_dataset = train_loader.dataset

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f"===== Fold {fold + 1}/{k_folds} =====")

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_loader_fold = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True,
                                       num_workers=config.num_workers)
        val_loader_fold = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False,
                                     num_workers=config.num_workers)

        adj_tensor = adj_tensor.to(Device)
        config_model = ModelConfig()
        model = Model(
            adj=adj_tensor,
            in_channels=1,
            lable_dim=config.lable_dim,
            time_len=config.window_size,
            device=config.window_size,
            config=config_model
        ).to(Device)


        def _tensor_bytes(t):
            return t.numel() * t.element_size()


        def report_model_size(model, title="Model"):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            param_bytes = sum(_tensor_bytes(p) for p in model.parameters())
            buffer_bytes = sum(_tensor_bytes(b) for b in model.buffers())
            total_bytes = param_bytes + buffer_bytes

            MB = 1_000_000
            MiB = 1024 ** 2
            Mb = 8 * total_bytes / MB

            try:
                any_param = next(model.parameters())
                dtype_str = str(any_param.dtype)
            except StopIteration:
                dtype_str = "N/A"

            print(f"[{title}] dtype={dtype_str}")
            print(f"[{title}] Params: total={total_params:,} | trainable={trainable_params:,}")
            print(f"[{title}] Size: Parameters={param_bytes / MB:.2f} MB ({param_bytes / MiB:.2f} MiB), "
                  f"Buffers={buffer_bytes / MB:.2f} MB ({buffer_bytes / MiB:.2f} MiB)")
            print(
                f"[{title}] Total (Params+Buffers)={total_bytes / MB:.2f} MB ({total_bytes / MiB:.2f} MiB)  ≈ {Mb:.2f} Mb")


        model = Model(
            adj=adj_tensor,
            in_channels=1,
            lable_dim=config.lable_dim,
            time_len=config.window_size,
            device=Device,
            config=config_model
        ).to(Device)

        report_model_size(model, title=f"Fold {fold + 1} Initial Model")

        model.loss_history = []
        model.train_metrics = []
        model.val_metrics = []

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

        num_epochs = 2#100
        best_val_loss = float("inf")


        for epoch in range(num_epochs):
            model.train()
            epoch_train_losses = []
            train_preds = []
            train_labels = []
            for batch_idx, batch in enumerate(train_loader_fold):
                x = batch["feature"].to(Device)
                time_4d = batch["time"].to(Device)
                label = batch["label"].to(Device)
                # 调整 time 的形状：例如 squeeze 掉最后两个维度
                time_2d = time_4d.squeeze(-1).squeeze(-1)

                pred = model(x, time_2d)
                loss = criterion(pred, label.squeeze(1))
                epoch_train_losses.append(loss.item())
                model.loss_history.append(loss.item())

                train_preds.append(pred.detach().cpu().numpy())
                train_labels.append(label.squeeze(1).detach().cpu().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print(f"[Fold {fold + 1} Epoch {epoch} | Batch {batch_idx}] Train Loss = {loss.item():.4f}")

            avg_train_loss = np.mean(epoch_train_losses)
            train_preds_np = np.concatenate(train_preds, axis=0)
            train_labels_np = np.concatenate(train_labels, axis=0)
            train_mse = np.mean((train_preds_np - train_labels_np) ** 2)
            train_rmse = np.sqrt(train_mse)
            train_mae = np.mean(np.abs(train_preds_np - train_labels_np))
            train_wmape = np.sum(np.abs(train_preds_np - train_labels_np)) / np.sum(np.abs(train_labels_np))
            model.train_metrics.append({
                "avg_train_loss": avg_train_loss,
                "mse": train_mse,
                "rmse": train_rmse,
                "mae": train_mae,
                "wmape": train_wmape
            })

            model.eval()
            val_losses = []
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for batch in val_loader_fold:
                    x = batch["feature"].to(Device)
                    time_4d = batch["time"].to(Device)
                    label = batch["label"].to(Device)
                    time_2d = time_4d.squeeze(-1).squeeze(-1)

                    pred = model(x, time_2d)
                    loss = criterion(pred, label.squeeze(1))
                    val_losses.append(loss.item())

                    val_preds.append(pred.cpu().numpy())
                    val_labels.append(label.squeeze(1).cpu().numpy())

            avg_val_loss = np.mean(val_losses)
            val_preds_np = np.concatenate(val_preds, axis=0)
            val_labels_np = np.concatenate(val_labels, axis=0)
            val_mse = np.mean((val_preds_np - val_labels_np) ** 2)
            val_rmse = np.sqrt(val_mse)
            val_mae = np.mean(np.abs(val_preds_np - val_labels_np))
            val_wmape = np.sum(np.abs(val_preds_np - val_labels_np)) / np.sum(np.abs(val_labels_np))
            model.val_metrics.append({
                "avg_val_loss": avg_val_loss,
                "mse": val_mse,
                "rmse": val_rmse,
                "mae": val_mae,
                "wmape": val_wmape
            })

            print(f"Fold {fold + 1} Epoch {epoch}:")
            print(
                f"  Train Metrics: Avg Train Loss = {avg_train_loss:.4f}, MSE = {train_mse:.4f}, RMSE = {train_rmse:.4f}, "
                f"MAE = {train_mae:.4f}, WMAPE = {train_wmape:.4f}")
            print(f"  Val Metrics:   Avg Val Loss   = {avg_val_loss:.4f}, MSE = {val_mse:.4f}, RMSE = {val_rmse:.4f}, "
                  f"MAE = {val_mae:.4f}, WMAPE = {val_wmape:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            if (epoch + 1) % 1 == 0:
                checkpoint = {
                    'fold': fold + 1,
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': model.train_metrics,
                    'val_metrics': model.val_metrics
                }
                checkpoint_path = f"checkpoint/fold{fold + 1}_checkpoint_epoch{epoch + 1}.pth"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint for fold {fold + 1} at epoch {epoch + 1}")

                if epoch + 1 == 5:
                    print(f"Fold {fold + 1} - Checkpoint at Epoch {epoch + 1}:")
                    print(
                        f"  Train Metrics: Avg Train Loss = {avg_train_loss:.4f}, MSE = {train_mse:.4f}, RMSE = {train_rmse:.4f}, "
                        f"MAE = {train_mae:.4f}, WMAPE = {train_wmape:.4f}")
                    print(
                        f"  Val Metrics:   Avg Val Loss   = {avg_val_loss:.4f}, MSE = {val_mse:.4f}, RMSE = {val_rmse:.4f}, "
                        f"MAE = {val_mae:.4f}, WMAPE = {val_wmape:.4f}")
