import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# 数据与导出路径
JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "departion_data.json")
ONNX_OUT = os.path.join(os.path.dirname(__file__), "models", "region_regressor.onnx")
CKPT_PATH = os.path.join(os.path.dirname(__file__), "models", "region_ckpt.pt")
BEST_PATH = os.path.join(os.path.dirname(__file__), "models", "region_best.pt")

# 尺寸设置
SRC_H, SRC_W = 40, 26   # 原始压力图矩阵尺寸
IMG_H, IMG_W = 128, 128 # 训练输入尺寸

# 框数量与输出维度（肩、背、腰、臀、大腿、小腿）
NUM_BOX = 6
OUT_DIM = NUM_BOX * 4  # 每个框(x1,y1,x2,y2)


def _resize_bilinear(img: np.ndarray, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("需要安装opencv-python用于图像缩放") from e
    return cv2.resize(img.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


class RegionDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            items = json.load(f)

        self.x, self.y = [], []
        for item in items:
            # 解析 data：可能是数组或逗号分隔字符串，或说明性文本（需跳过）
            raw = item.get("data")
            if isinstance(raw, str):
                if "," not in raw:
                    # 诸如 "(40,26)->reshape(-1)->1040 ->list" 的说明行，跳过
                    continue
                import re as _re
                nums = _re.findall(r"[-+]?\d*\.?\d+", raw)
                if len(nums) != SRC_H * SRC_W:
                    # 数据长度不匹配，跳过
                    continue
                vals = [float(x) for x in nums]
                mat = np.array(vals, dtype=np.float32)
            else:
                mat = np.array(raw, dtype=np.float32)
                if mat.size != SRC_H * SRC_W:
                    continue
            mat = mat.reshape(SRC_H, SRC_W)
            # 归一化到0~1，避免全零除法
            min_v, max_v = float(np.min(mat)), float(np.max(mat))
            if max_v - min_v > 1e-6:
                mat = (mat - min_v) / (max_v - min_v)
            mat = _resize_bilinear(mat, IMG_H, IMG_W)[None, ...]  # 1xHxW
            self.x.append(mat)

            # 解析 region：可能是列表或空格分隔字符串（24 个数 = 6 框）
            reg = item.get("region")
            if isinstance(reg, str):
                import re as _re
                nums = _re.findall(r"[-+]?\d+", reg)
                if len(nums) < 24:
                    continue
                nums = [float(x) for x in nums[:24]]
                # 重排为6个框：[x1,x2]*6 + [y1,y2]*6 → [(x1,y1,x2,y2)]*6
                x_pairs = [(nums[2*i], nums[2*i+1]) for i in range(6)]
                y_pairs = [(nums[12+2*i], nums[12+2*i+1]) for i in range(6)]
                boxes = [(x1, y1, x2, y2) for (x1, x2), (y1, y2) in zip(x_pairs, y_pairs)]
            else:
                boxes = reg
                if not isinstance(boxes, list) or len(boxes) < 6:
                    continue
            # 归一化到0~1（原坐标基于宽26×高40）
            norm = []
            for (x1, y1, x2, y2) in boxes[:6]:
                # 保证左上到右下
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                norm += [float(x1) / SRC_W, float(y1) / SRC_H, float(x2) / SRC_W, float(y2) / SRC_H]
            self.y.append(np.array(norm, dtype=np.float32))

        if len(self.x) == 0:
            raise RuntimeError("未解析到有效样本，请检查 departion_data.json 格式")
        self.x = np.stack(self.x)
        self.y = np.stack(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 轻量CNN，兼顾在嵌入设备上的速度
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, OUT_DIM),
            nn.Sigmoid(),  # 输出0~1
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


def _boxes_to_pixels(b, w=SRC_W, h=SRC_H):
    import numpy as np
    b = np.array(b, dtype=np.float32).reshape(-1, 6, 4)
    b[..., 0] *= w; b[..., 2] *= w
    b[..., 1] *= h; b[..., 3] *= h
    return b


def _iou_xyxy(a, b):
    import numpy as np
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[..., 2] - a[..., 0]) * np.maximum(0.0, a[..., 3] - a[..., 1])
    area_b = np.maximum(0.0, b[..., 2] - b[..., 0]) * np.maximum(0.0, b[..., 3] - b[..., 1])
    union = np.maximum(1e-6, area_a + area_b - inter)
    return inter / union


def _sample_accuracy(p_norm, g_norm, thr=0.5):
    import numpy as np
    p_pix = _boxes_to_pixels(p_norm)
    g_pix = _boxes_to_pixels(g_norm)
    iou = _iou_xyxy(p_pix, g_pix)  # [N,6]
    ok = (iou >= thr).all(axis=1).astype(np.float32)  # 所有6框均达标
    return float(ok.mean()), iou

def _sample_accuracy_k(p_norm, g_norm, thr=0.5, k: int = 4):
    """至少 k/6 个框达到阈值算正确。"""
    import numpy as np
    p_pix = _boxes_to_pixels(p_norm)
    g_pix = _boxes_to_pixels(g_norm)
    iou = _iou_xyxy(p_pix, g_pix)  # [N,6]
    ok = (iou >= thr).sum(axis=1) >= k
    return float(ok.mean())


def train_and_export(epochs: int = 100, batch_size: int = 64, lr: float = 1e-3, acc_thr: float = 0.5, resume: bool = False):
    ds = RegionDataset(JSON_PATH)
    n_total = len(ds)
    n_train = int(n_total * 0.7)
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_total - n_train], generator=gen)

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleBackbone().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # 可选：OneCycleLR 提升收敛
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=max(1, len(dl_train)), epochs=epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.02)

    best_val = 1e9
    best_state = None
    train_hist, val_hist, miou_hist, mae_hist = [], [], [], []

    # 恢复
    # start_epoch = 1
    # if resume and os.path.exists(CKPT_PATH):
    #     ckpt = torch.load(CKPT_PATH, map_location=device)
    #     model.load_state_dict(ckpt.get("model_state", {}))
    #     optimizer.load_state_dict(ckpt.get("optimizer_state", {}))
    #     try:
    #         scheduler.load_state_dict(ckpt.get("scheduler_state", {}))
    #     except Exception:
    #         pass
    #     start_epoch = int(ckpt.get("epoch", 0)) + 1
    #     best_val = float(ckpt.get("best_val", best_val))
    #     train_hist = ckpt.get("train_hist", [])
    #     val_hist = ckpt.get("val_hist", [])
    #     miou_hist = ckpt.get("miou_hist", [])
    #     mae_hist = ckpt.get("mae_hist", [])
    #     print(f"Resumed from epoch {start_epoch-1}, best_val={best_val:.4f}")
    # elif resume and os.path.exists(BEST_PATH):
    #     state = torch.load(BEST_PATH, map_location=device)
    #     model.load_state_dict(state)
    #     print("Loaded best weights to continue training.")


    # --- 修改这里 ---
    # 如果 resume 为 True，且存在最佳模型权重文件，则加载
    if resume and os.path.exists(BEST_PATH):
        # 仅加载模型权重
        state = torch.load(BEST_PATH, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded best weights from {BEST_PATH} to continue training with new settings.")
    # 否则，检查点恢复逻辑不变
    elif resume and os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt.get("model_state", {}))
        optimizer.load_state_dict(ckpt.get("optimizer_state", {}))
        try:
            scheduler.load_state_dict(ckpt.get("scheduler_state", {}))
        except Exception:
            pass
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        train_hist = ckpt.get("train_hist", [])
        val_hist = ckpt.get("val_hist", [])
        miou_hist = ckpt.get("miou_hist", [])
        mae_hist = ckpt.get("mae_hist", [])
        print(f"Resumed from epoch {start_epoch-1}, best_val={best_val:.4f}")
    
    # 将start_epoch设置为1，以新的起点开始
    start_epoch = 1
    # --- 修改结束 ---



    try:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            train_loss = 0.0
            for x, y in dl_train:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * x.size(0)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in dl_val:
                    x = x.to(device)
                    y = y.to(device)
                    pred = model(x)
                    val_loss += loss_fn(pred, y).item() * x.size(0)

            train_loss /= len(train_ds)
            val_loss /= len(val_ds)
            # 计算验证mIoU与MAE
            import numpy as np
            with torch.no_grad():
                all_iou, all_mae = [], []
                acc_04, acc_05, acc_07 = [], [], []
                acc4_05 = []  # 至少4/6达标@0.5
                for x, y in dl_val:
                    x = x.to(device); y = y.to(device)
                    p = model(x).cpu().numpy()
                    g = y.cpu().numpy()
                    all_iou.append(_iou_xyxy(_boxes_to_pixels(p), _boxes_to_pixels(g)))  # [N,6]
                    all_mae.append(np.mean(np.abs(p - g), axis=1))  # [N]
                    a04, _ = _sample_accuracy(p, g, thr=0.4)
                    a05, _ = _sample_accuracy(p, g, thr=0.5)
                    a07, _ = _sample_accuracy(p, g, thr=0.7)
                    acc_04.append(a04); acc_05.append(a05); acc_07.append(a07)
                    acc4_05.append(_sample_accuracy_k(p, g, thr=0.5, k=4))
                miou = float(np.mean(np.concatenate(all_iou, axis=0)))
                mae = float(np.mean(np.concatenate(all_mae, axis=0)))
                acc = float(np.mean(acc_05))
                acc04 = float(np.mean(acc_04)); acc07 = float(np.mean(acc_07)); acc405 = float(np.mean(acc4_05))

            train_hist.append(train_loss)
            val_hist.append(val_loss)
            miou_hist.append(miou)
            mae_hist.append(mae)
            print(f"[Epoch {epoch}/{epochs}] train={train_loss:.4f} val={val_loss:.4f} mIoU={miou:.4f} MAE={mae:.4f} "
                  f"ACC@0.4={acc04*100:.2f}% ACC@0.5={acc*100:.2f}% ACC@0.7={acc07*100:.2f}% ACC4/6@0.5={acc405*100:.2f}%")

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            # 保存检查点
            os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val": best_val,
                "train_hist": train_hist,
                "val_hist": val_hist,
                "miou_hist": miou_hist,
                "mae_hist": mae_hist,
            }, CKPT_PATH)
    except KeyboardInterrupt:
        # 捕获中断，立即保存一次检查点，方便继续训练
        os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val": best_val,
            "train_hist": train_hist,
            "val_hist": val_hist,
            "miou_hist": miou_hist,
            "mae_hist": mae_hist,
        }, CKPT_PATH)
        print("\n[INFO] 捕获到中断，已保存当前检查点到:", CKPT_PATH)
        # 不中断后续导出流程，继续用当前 best_state 完成保存

    # 保存曲线
    os.makedirs(os.path.dirname(ONNX_OUT), exist_ok=True)
    curve_path = os.path.join(os.path.dirname(ONNX_OUT), "region_training_curve.png")
    plt.figure(figsize=(8,4))
    plt.plot(train_hist, label='train_loss')
    plt.plot(val_hist, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(); plt.tight_layout()
    plt.savefig(curve_path)
    print(f"saved curve => {curve_path}")

    # 保存最优权重
    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, BEST_PATH)
    # 导出ONNX
    model.load_state_dict(best_state)
    model.eval()
    dummy = torch.zeros(1, 1, IMG_H, IMG_W, dtype=torch.float32)
    try:
        import onnx  # 确保依赖存在
        torch.onnx.export(
            model,
            dummy,
            ONNX_OUT,
            input_names=["input"],
            output_names=["boxes"],
            opset_version=12,
            dynamic_axes={"input": {0: "N"}, "boxes": {0: "N"}},
        )
        print(f"saved ONNX => {ONNX_OUT}")
    except Exception as e:
        print("[WARN] 未导出ONNX：", e)
        print("请先安装: pip install onnx，然后重新运行该脚本以导出ONNX")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--acc-thr", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train_and_export(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        acc_thr=args.acc_thr,
        resume=args.resume,
    )


