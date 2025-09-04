# -*- coding: utf-8 -*-
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import ModelConfig, Device, window_size, lable_dim
from data_loader import build_passenger_flow_dataloader
from ModelFrame import Model as TeacherModel

class LowRankLinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=16, bias=True):
        super().__init__()
        self.u = nn.Linear(in_dim, rank, bias=False)
        self.v = nn.Linear(rank, out_dim, bias=bias)
    def forward(self, x):
        return self.v(self.u(x))

class TinyStudent(nn.Module):

    def __init__(self, in_ch, H, W, lable_dim,
                 hidden=128, rank1=16, rank2=8,
                 share_weights=True, dropout=0.1):
        super().__init__()
        self.in_dim = in_ch * H * W
        self.hidden = hidden
        self.share = share_weights

        self.ln_in = nn.LayerNorm(self.in_dim)
        self.fc1   = LowRankLinear(self.in_dim, hidden, rank=rank1)
        self.ln1   = nn.LayerNorm(hidden)
        self.drop  = nn.Dropout(dropout)

        if self.share:
            self.bottleneck = LowRankLinear(hidden, hidden, rank=rank2)
        else:
            self.bneck1 = LowRankLinear(hidden, hidden, rank=rank2)
            self.bneck2 = LowRankLinear(hidden, hidden, rank=rank2)

        self.pre_head_ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, lable_dim)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        x = self.ln_in(x)
        x = F.gelu(self.fc1(x))
        x = self.drop(self.ln1(x))

        if self.share:
            x = F.gelu(self.bottleneck(x))
            x = F.gelu(self.bottleneck(x))
        else:
            x = F.gelu(self.bneck1(x))
            x = F.gelu(self.bneck2(x))

        x = self.pre_head_ln(x)
        return self.head(x)

_cos = torch.nn.CosineEmbeddingLoss()
_huber = torch.nn.SmoothL1Loss(beta=0.5)

def kd_regression_loss(student_pred, teacher_pred, alpha=0.7):

    target = torch.ones(student_pred.size(0), device=student_pred.device)
    return alpha * _huber(student_pred, teacher_pred) + \
           (1 - alpha) * _cos(student_pred, teacher_pred, target)

def report_model_size(model, name="model"):
    with torch.no_grad():
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        total = param_bytes + buffer_bytes
        mib = total / (1024 ** 2)
        mb  = total / (1000 ** 2)
        params = sum(p.numel() for p in model.parameters())
    print(f"[size] {name}: params={params/1e6:.3f}M, bytes={total} "
          f"({mib:.2f} MiB / {mb:.2f} MB)")

def auto_infer_hw(total):
    for h in range(int(math.sqrt(total)), 0, -1):
        if total % h == 0:
            return h, total // h
    raise ValueError("无法从总元素数推导出 H 和 W。")

def safe_reshape_for_mixer(x_btn, C, H, W):

    if x_btn.dim() == 4 and x_btn.size(1) == 1:
        x_btn = x_btn.squeeze(1)
    assert x_btn.dim() == 3, f"expect [B,T,N], got {x_btn.shape}"
    B, T, N = x_btn.shape
    x = x_btn.reshape(B, C, H, W)
    return x

if __name__ == "__main__":
    torch.set_num_threads(min(8, os.cpu_count() or 8))
    os.makedirs("output", exist_ok=True)

    train_loader, val_loader, adj_tensor = build_passenger_flow_dataloader(
        batch_size=16,
        shuffle=True,
        num_workers=0,
        device=Device,
        window_size=window_size,
        step_size=1,
        train_ratio=0.8
    )
    adj_tensor = adj_tensor.to(Device)

    teacher_config = ModelConfig(in_channels=1)
    teacher_model = TeacherModel(
        adj=adj_tensor,
        in_channels=teacher_config.in_channels,
        lable_dim=lable_dim,
        time_len=window_size,
        device=Device,
        config=teacher_config
    ).to(Device)
    ckpt_path = "checkpoint/fold4_checkpoint_epoch1.pth"
    checkpoint = torch.load(ckpt_path, map_location=Device, weights_only=False)
    teacher_model.load_state_dict(checkpoint["model_state_dict"])
    teacher_model.eval()

    sample_batch = next(iter(train_loader))
    sample_input = sample_batch["feature"].squeeze(1).to(Device)  # [B,T,N]
    B, T, N = sample_input.shape
    C = teacher_config.in_channels  # 1
    assert (T * N) % C == 0, f"T*N={T*N} 不能被 in_channels={C} 整除!"
    hw = (T * N) // C
    H, W = auto_infer_hw(hw)
    print(f"[info] TinyStudent 输入尺寸: C={C}, H={H}, W={W}  (由T*N推导)")

    student_model = TinyStudent(
        in_ch=C, H=H, W=W, lable_dim=lable_dim,
        hidden=128,
        rank1=16,
        rank2=8,
        share_weights=True,
        dropout=0.1
    ).to(Device)

    report_model_size(student_model, "student(fp32)")
    report_model_size(teacher_model, "teacher(fp32)")

    optimizer = optim.Adam(student_model.parameters(), lr=1e-3, weight_decay=1e-5)
    num_epochs = 200
    best_val = float("inf")

    for epoch in range(1, num_epochs + 1):
        student_model.train()
        t0 = time.time()
        total_loss, steps = 0.0, 0

        for i, batch in enumerate(train_loader):
            x_raw = batch["feature"].to(Device)             # [B,1,T,N]
            t_raw = batch["time"].squeeze(-1).squeeze(-1).to(Device)

            x = safe_reshape_for_mixer(x_raw, C, H, W)      # [B,C,H,W]

            # 前向
            s_pred = student_model(x)
            with torch.no_grad():
                t_pred = teacher_model(x_raw, t_raw)

            loss = kd_regression_loss(s_pred, t_pred, alpha=0.7)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_train = total_loss / max(1, steps)

        student_model.eval()
        val_loss, vsteps = 0.0, 0
        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                x_raw = batch["feature"].to(Device)
                t_raw = batch["time"].squeeze(-1).squeeze(-1).to(Device)
                try:
                    x = safe_reshape_for_mixer(x_raw, C, H, W)
                except Exception:
                    continue

                s_pred = student_model(x)
                t_pred = teacher_model(x_raw, t_raw)
                v_loss = kd_regression_loss(s_pred, t_pred, alpha=0.7).item()
                val_loss += v_loss
                vsteps += 1

        avg_val = val_loss / max(1, vsteps)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d} | train {avg_train:.6f} | val {avg_val:.6f} | {elapsed:.1f}s")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(student_model.state_dict(), "output/student_best.pth")
            print("  -> saved: output/student_best.pth")

    student_model.load_state_dict(torch.load("output/student_best.pth", map_location=Device))
    student_model.eval()
    test_loss, tsteps = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            x_raw = batch["feature"].to(Device)
            t_raw = batch["time"].squeeze(-1).squeeze(-1).to(Device)
            try:
                x = safe_reshape_for_mixer(x_raw, C, H, W)
            except Exception:
                continue
            s_pred = student_model(x)
            t_pred = teacher_model(x_raw, t_raw)
            test_loss += kd_regression_loss(s_pred, t_pred, alpha=0.7).item()
            tsteps += 1
    test_loss = test_loss / max(1, tsteps)
    print(f"[final] Test KD Loss (to Teacher): {test_loss:.6f}")

    student_model_cpu = TinyStudent(
        in_ch=C, H=H, W=W, lable_dim=lable_dim,
        hidden=128, rank1=16, rank2=8, share_weights=True, dropout=0.1
    )
    student_model_cpu.load_state_dict(torch.load("output/student_best.pth", map_location="cpu"))
    student_model_cpu.eval()

    report_model_size(student_model_cpu, "student(fp32, cpu)")
    try:
        import torch.quantization as tq
        q_student = torch.quantization.quantize_dynamic(
            student_model_cpu, {nn.Linear}, dtype=torch.qint8
        )
        report_model_size(q_student, "student(int8-dynamic, cpu)")
        torch.save(q_student, "output/student_quantized.pt")
        print("  -> saved: output/student_quantized.pt")
    except Exception as e:
        print(f"[warn] 动态量化失败：{e}")
