import sys
from pathlib import Path

# Add the directory containing resnet50_model.py to the Python path
# This assumes resnet50_model.py is in the same directory as processed_data_np224
# Adjust the path if necessary
module_path = Path("/content/drive/MyDrive/Adversarial_AI") # Assuming it's here based on the user's comment
if str(module_path) not in sys.path:
    sys.path.append(str(module_path))
    print(f"[INFO] Added {module_path} to sys.path")
else:
    print(f"[INFO] {module_path} is already in sys.path")

import torch
from resnet50_model import load_trained_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = "/content/drive/MyDrive/Adversarial_AI/models/resnet50_binary_best.pth"
model = load_trained_model(checkpoint_path, device=device)

import time
import math
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torchvision.transforms import ToPILImage
from PIL import Image
import matplotlib.pyplot as plt

# Square-Attack L2 하이퍼파라미터
eps = 2
n_iters = 10000
p_init = 0.1
targeted = False          # True면 표적 공격
target_label_idx = 2

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import json

class BrainTumorDatasetWithROI(Dataset):
    """
    뇌종양 이미지 데이터셋 (+ ROI mask 포함)
    - images.npy : (N,224,224,1) [0,1]
    - labels.npy : (N,) int64
    - masks.npy  : (N,224,224) bool or {0,1}
    - class_names.txt : 클래스명
    - meta.json  : (옵션) 메타데이터
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # 이미지, 라벨, 마스크 로드
        self.images = np.load(self.data_dir / "images.npy")   # (N,224,224,1)
        self.labels = np.load(self.data_dir / "labels.npy")   # (N,)
        self.masks  = np.load(self.data_dir / "masks.npy")    # (N,224,224)

        # 클래스명
        with open(self.data_dir / "class_names.txt", "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f]

        # 메타데이터 (있으면)
        meta_path = self.data_dir / "meta.json"
        self.meta = None
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

        print(f"[INFO] Loaded {len(self.images)} images with ROI from {self.data_dir}")
        print(f"[INFO] Classes: {self.class_names}")
        if self.meta is not None:
            print(f"[INFO] Meta: {self.meta}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1채널 → 3채널 복제
        img = self.images[idx]              # (224,224,1)
        img = np.repeat(img, 3, axis=2)    # (224,224,3)

        label = int(self.labels[idx])
        mask  = self.masks[idx]            # (224,224)

        if self.transform:
            img = self.transform(img)      # (3,224,224)

        # mask는 torch.Tensor로 변환
        mask = torch.from_numpy(mask.astype(np.float32))  # (224,224)

        return img, label, mask

def to_norm(x):
    return x

@torch.no_grad()
def predict_logits(x_pix):
    model.eval()
    return model(to_norm(x_pix))

@torch.no_grad()
def predict_logits(x_pix):
    return model(to_norm(x_pix))

def margin_loss_untargeted(logits, y):
    bs = logits.size(0)
    correct_logit = logits[torch.arange(bs, device=logits.device), y]
    tmp = logits.clone()
    tmp[torch.arange(bs, device=logits.device), y] = -1e9
    max_other = tmp.max(dim=1).values
    return correct_logit - max_other   # <=0 success

def margin_loss_targeted(logits, y_tgt):
    bs = logits.size(0)
    target_logit = logits[torch.arange(bs, device=logits.device), y_tgt]
    tmp = logits.clone()
    tmp[torch.arange(bs, device=logits.device), y_tgt] = -1e9
    max_other = tmp.max(dim=1).values
    return max_other - target_logit    # <=0 success

def ce_loss_targeted(logits, y_tgt):
    return nn.CrossEntropyLoss(reduction='none')(logits, y_tgt)

def p_selection(p_init, it, n_iters):
    it = int(it / n_iters * 10000)
    if   10 < it <=  50: p = p_init / 2
    elif 50 < it <= 200: p = p_init / 4
    elif 200 < it <= 500: p = p_init / 8
    elif 500 < it <= 1000: p = p_init / 16
    elif 1000 < it <= 2000: p = p_init / 32
    elif 2000 < it <= 4000: p = p_init / 64
    elif 4000 < it <= 6000: p = p_init / 128
    elif 6000 < it <= 8000: p = p_init / 256
    elif 8000 < it <= 10000: p = p_init / 512
    else: p = p_init
    return p

def pseudo_gaussian_pert_rectangles(x, y, device):
    delta = torch.zeros(x, y, device=device)
    x_c, y_c = x // 2 + 1, y // 2 + 1
    cx, cy = x_c - 1, y_c - 1
    for counter in range(0, max(x_c, y_c)):
        x0 = max(cx, 0); x1 = min(cx + (2*counter + 1), x)
        y0 = max(cy, 0); y1 = min(cy + (2*counter + 1), y)
        delta[x0:x1, y0:y1] += 1.0 / (counter + 1) ** 2
        cx -= 1; cy -= 1
    delta = delta / (delta.pow(2).sum().sqrt() + 1e-12)
    return delta

def meta_pseudo_gaussian_pert_square(s, device):
    delta = torch.zeros(s, s, device=device)
    delta[:s//2] = pseudo_gaussian_pert_rectangles(s//2, s, device)
    delta[s//2:] = -pseudo_gaussian_pert_rectangles(s - s//2, s, device)
    delta = delta / (delta.pow(2).sum().sqrt() + 1e-12)
    if torch.rand(1, device=device).item() > 0.5:
        delta = delta.t()
    return delta

def _ensure_mask_tensor(mask_2d, device, shape_c_h_w):
    """
    mask_2d: (H, W) numpy or torch, {0,1} or bool
    return: (1, C, H, W) float tensor on device in {0,1}
    """
    if isinstance(mask_2d, np.ndarray):
        m = torch.from_numpy(mask_2d)
    else:
        m = mask_2d
    m = m.float()
    m = (m > 0.5).float()  # 이진화
    # (H,W) -> (1,1,H,W) -> (1,C,H,W)
    _, C, H, W = shape_c_h_w
    m = m.view(1, 1, H, W).repeat(1, C, 1, 1).to(device)
    # ROI가 전혀 없으면(합이 0) 전체 이미지로 대체
    if m.sum().item() == 0:
        m = torch.ones_like(m)
    return m

def _l2_norm_over_mask(t, m):
    # t, m: (1,C,H,W). m in {0,1}. 마스크 내부만 계산
    v = (t * m).view(1, -1)
    return v.norm(p=2, dim=1, keepdim=True).view(1,1,1,1).clamp(min=1e-12)

def _l2_norm_over_mask_region(t, m, region_mask):
    # region_mask: (1,C,H,W) with {0,1} selecting region; use intersection with m
    mm = torch.maximum((m>0).float(), torch.zeros_like(m))  # ensure 0/1
    r = torch.maximum((region_mask>0).float(), torch.zeros_like(region_mask))
    v = (t * mm * r).view(1, -1)
    return v.norm(p=2, dim=1, keepdim=True).view(1,1,1,1).clamp(min=1e-12)

def _apply_mask(delta, m):
    # 항상 ROI 밖은 0
    delta.mul_(m)
    return delta

def square_attack_l2_single_roi(
    x_pix, y_true, mask_2d, y_target=None,
    eps=0.5, n_iters=2000, p_init=0.1,
    targeted=False, use_ce_for_targeted=True, seed=0
):
    """
    x_pix: (1,3,H,W), [0,1]
    mask_2d: (H,W) torch/numpy, {0,1}/bool  → ROI만 공격
    L2 제약(ε)도 ROI 내 픽셀에 대해서만 적용
    """
    torch.manual_seed(seed)
    device = x_pix.device
    _, c, h, w = x_pix.shape
    n_features = c * h * w

    # ROI 마스크 준비
    m = _ensure_mask_tensor(mask_2d, device, (1, c, h, w))  # (1,C,H,W)

    # ----- init: 가우시안 타일 초기화 후 ROI로 마스킹 -----
    delta_init = torch.zeros_like(x_pix)
    s = max(min(h, w) // 5, 3)
    if s % 2 == 0: s += 1
    for ch in range(0, h, s):
        for cw in range(0, w, s):
            hs = min(s, h - ch); ws = min(s, w - cw)
            if hs < 3 or ws < 3: continue
            rect = pseudo_gaussian_pert_rectangles(hs, ws, device).view(1,1,hs,ws).repeat(1, c, 1, 1)
            sign = torch.randint(0, 2, (1, c, 1, 1), device=device).float() * 2 - 1
            delta_init[:, :, ch:ch+hs, cw:cw+ws] += rect * sign

    # ROI 밖은 0으로
    delta_init = _apply_mask(delta_init, m)

    # ROI 내부 L2로 정규화 후 eps로 스케일
    norm_roi = _l2_norm_over_mask(delta_init, m)
    x_best = torch.clamp(x_pix + delta_init * (eps / norm_roi), 0.0, 1.0)

    logits = predict_logits(x_best)
    if targeted:
        margin_min = margin_loss_targeted(logits, y_target)
        loss_min = ce_loss_targeted(logits, y_target) if use_ce_for_targeted else margin_min
    else:
        margin_min = margin_loss_untargeted(logits, y_true)
        loss_min = margin_min

    n_queries = 1
    time_start = time.time()

    for i in range(n_iters):
        if (margin_min <= 0).all():
            break

        x_curr  = x_pix.clone()
        x_best_ = x_best.clone()
        delta   = (x_best_ - x_curr)              # (1,C,H,W)
        delta   = _apply_mask(delta, m)           # ROI 밖 제거

        p = p_selection(p_init, i, n_iters)
        s = max(int(round(math.sqrt(p * n_features / c))), 3)
        s = min(s, min(h, w) - 1)
        if s % 2 == 0: s += 1
        s2 = s
        if h - s <= 0 or w - s <= 0: break

        ch = torch.randint(0, h - s + 1, (1,), device=device).item()
        cw = torch.randint(0, w - s + 1, (1,), device=device).item()
        ch2 = torch.randint(0, h - s2 + 1, (1,), device=device).item()
        cw2 = torch.randint(0, w - s2 + 1, (1,), device=device).item()

        # 지역 마스크(정사각형 두 개의 합집합)
        mask1 = torch.zeros_like(x_curr); mask1[:, :, ch:ch+s,  cw:cw+s ] = 1.0
        mask2 = torch.zeros_like(x_curr); mask2[:, :, ch2:ch2+s2, cw2:cw2+s2] = 1.0
        mask_union = torch.maximum(mask1, mask2)

        # ROI와의 교집합 영역만 실제 유효
        mask_union_roi = mask_union * m

        curr_norm_window = _l2_norm_over_mask_region((x_best_ - x_curr), m, mask1)
        curr_norm_img    = _l2_norm_over_mask((x_best_ - x_curr), m)
        norms_windows    = _l2_norm_over_mask((delta * mask_union), m)

        new_deltas = meta_pseudo_gaussian_pert_square(s, device).view(1,1,s,s).repeat(1, c, 1, 1)
        sign = torch.randint(0, 2, (1, c, 1, 1), device=device).float() * 2 - 1
        new_deltas = new_deltas * sign

        old_deltas = (delta[:, :, ch:ch+s, cw:cw+s] / (curr_norm_window + 1e-10))
        new_deltas = new_deltas + old_deltas

        # ROI 내 L2-ε 유지 위한 스케일
        # scale^2 = max(eps^2 - ||delta||_ROI^2, 0)/C + ||(delta * mask_union)||_ROI^2
        scale = ((torch.clamp(eps**2 - curr_norm_img**2, min=0.0) / c) + norms_windows**2).sqrt()

        # 새 패치 정규화 후 스케일
        nd_norm = _l2_norm_over_mask(new_deltas, torch.ones_like(new_deltas))  # 패치 자체 norm
        new_deltas = new_deltas / (nd_norm + 1e-12) * scale

        # 바깥 사각형은 0으로
        delta[:, :, ch2:ch2+s2, cw2:cw2+s2] = 0.0
        # 새 패치 적용
        delta[:, :, ch:ch+s, cw:cw+s] = new_deltas
        # ROI 밖 제거
        delta = _apply_mask(delta, m)

        # 전체 ROI L2를 eps로 맞춤
        new_norm = _l2_norm_over_mask(delta, m)
        x_new = torch.clamp(x_curr + delta * (eps / new_norm), 0.0, 1.0)

        logits_new = predict_logits(x_new)
        if targeted:
            margin_new = margin_loss_targeted(logits_new, y_target)
            loss_new   = ce_loss_targeted(logits_new, y_target) if use_ce_for_targeted else margin_new
        else:
            margin_new = margin_loss_untargeted(logits_new, y_true)
            loss_new   = margin_new

        if (loss_new < loss_min).item():
            x_best = x_new
            loss_min = loss_new
            margin_min = margin_new

        n_queries += 1

        if (i+1) % 50 == 0 or margin_min.item() <= 0:
            status = "OK" if margin_min.item() <= 0 else "..."
            print(f"[{i+1:4d}/{n_iters}] margin={margin_min.item():.4f}  queries={n_queries}  s={s}  {status}")

    elapsed = time.time() - time_start
    return x_best, n_queries, elapsed, margin_min.item()

@torch.no_grad()
def stats_table_like(x_pix_clean, x_adv_pix, elapsed_sec, y_true, count_mode='element', tau_255=1.0):
    delta = (x_adv_pix - x_pix_clean)
    l2_255 = (delta * 255.0).view(1, -1).norm(p=2).item()

    if count_mode == 'element':   # H*W*C 기준
        changed = ((delta.abs() * 255.0) > tau_255).sum().item()
        total   = int(delta.numel())
    else:                         # H*W 기준
        per_spatial = delta.abs().max(dim=1)[0].squeeze(0)
        changed = ((per_spatial * 255.0) > tau_255).sum().item()
        total   = int(per_spatial.numel())

    # --- ΔMargin 계산 ---
    logits_clean = predict_logits(x_pix_clean)
    logits_adv   = predict_logits(x_adv_pix)

    # y_true: int 인덱스
    y = int(y_true)
    correct_logit_clean = logits_clean[0, y].item()
    max_other_clean     = logits_clean[0, torch.arange(logits_clean.size(1)) != y].max().item()
    margin_clean = correct_logit_clean - max_other_clean

    correct_logit_adv = logits_adv[0, y].item()
    max_other_adv     = logits_adv[0, torch.arange(logits_adv.size(1)) != y].max().item()
    margin_adv = correct_logit_adv - max_other_adv

    delta_margin = margin_clean - margin_adv  # 양수면 공격이 margin을 줄였다는 뜻

    return {
        "L2_255": float(l2_255),
        "changed": int(changed),
        "total": int(total),
        "time": float(elapsed_sec),
        "dmargin": float(delta_margin)
    }

from torch.utils.data import DataLoader

# Dataset & Loader
test_set = BrainTumorDatasetWithROI(
    "/content/drive/MyDrive/Adversarial_AI/processed_data_np224/Testing",
    transform=transforms.ToTensor()
)
test_loader_100 = DataLoader(test_set, batch_size=100, shuffle=False)

# class_names 가져오기
class_names = test_set.class_names

x_pix_clean, y_true, masks_2d = next(iter(test_loader_100))  # (100,3,224,224), (100,), (100,224,224)
x_pix_clean = x_pix_clean.to(device)
y_true = y_true.to(device)
B = x_pix_clean.size(0)

import os, time
import numpy as np
import torch
import matplotlib.pyplot as plt

def _tensor_img_to_numpy(x):  # (1,3,H,W) -> (H,W,3)
    if x.dim() == 4: x = x.squeeze(0)
    return x.detach().clamp(0,1).permute(1,2,0).cpu().numpy()

def _pred_and_conf(logits):
    probs = torch.softmax(logits, dim=1)
    pred  = probs.argmax(1).item()
    conf  = probs[0, pred].item()
    return pred, conf

def visualize_attack_quick(
    model, x_one, x_adv, class_names=None, mask_2d=None,
    amp_heat=3.0, out_dir=".", filename_prefix="attack_vis", display=True
):
    """
    x_one: (1,3,H,W) 원본
    x_adv: (1,3,H,W) 이미 생성된 적대 이미지 (재공격 안 함)
    """
    os.makedirs(out_dir, exist_ok=True)
    device = x_one.device
    print("[VIS] visualize_attack_quick start")

    # 예측/신뢰도
    with torch.no_grad():
        logits_before = predict_logits(x_one)
        logits_after  = predict_logits(x_adv)
    pred_b, conf_b = _pred_and_conf(logits_before)
    pred_a, conf_a = _pred_and_conf(logits_after)

    # 시각화 데이터
    img_b = _tensor_img_to_numpy(x_one)
    img_a = _tensor_img_to_numpy(x_adv)
    delta = (x_adv - x_one).squeeze(0).norm(p=2, dim=0).cpu().numpy()
    pert_map_vis = delta * amp_heat

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot(1,3,1); ax.imshow(img_b); ax.set_title("Original"); ax.axis('off')
    if mask_2d is not None:
        m = (mask_2d.detach().cpu().numpy() if torch.is_tensor(mask_2d) else mask_2d).astype(float)
        ax.contour(m, levels=[0.5], linewidths=1.5, colors='cyan')

    ax = plt.subplot(1,3,2); ax.imshow(img_a); ax.set_title("Adversarial"); ax.axis('off')
    if mask_2d is not None:
        ax.contour(m, levels=[0.5], linewidths=1.5, colors='cyan')

    ax = plt.subplot(1,3,3); im = ax.imshow(pert_map_vis, cmap='magma'); plt.colorbar(im)
    ax.set_title(f"|δ| (x{amp_heat:g})"); ax.axis('off')

    name = (lambda i: class_names[i] if (class_names and 0 <= i < len(class_names)) else str(i))
    title = f"Before: {name(pred_b)}({conf_b:.3f}) → After: {name(pred_a)}({conf_a:.3f})"
    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()

    ts = int(time.time())
    out_path = os.path.join(out_dir, f"{filename_prefix}_{ts}.png")
    #plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"[VIS] Saved to: {out_path}")

    if display:
        plt.show()
    else:
        plt.close(fig)

    return out_path

# === ROI 사용 여부 토글 ===
USE_ROI = False   # True = ROI 내에서만 공격 / False = 전체 이미지 공격

def _full_ones_mask_like(x1):
    # x1: (1,3,H,W)
    _, _, H, W = x1.shape
    return torch.ones(H, W, dtype=torch.float32, device=x1.device)

x_adv_list = []
elapsed_list = []
success_list = []
l2_list = []
changed_list = []
total_list = []
dmargin_list = []
conf_drop_list = []

for i in range(B):
    xi = x_pix_clean[i:i+1]        # (1,3,H,W)
    mi = masks_2d[i]               # (H,W), torch.bool/float

    if USE_ROI:
          mi_eff = mi          # ROI 공격 (데이터셋 마스크 사용)
          mask_for_vis = mi
    else:
          mi_eff = _full_ones_mask_like(xi)  # 전체 이미지 공격 (올-원 마스크)
          mask_for_vis = None

    with torch.no_grad():
        src_idx = predict_logits(xi).argmax(1)  # untargeted: 현재 예측을 소스 라벨로

    # ROI 공격 실행
    x_adv_pix, n_queries, elapsed, final_margin = square_attack_l2_single_roi(
        xi, y_true=src_idx, mask_2d=mi_eff,
        eps=eps, n_iters=n_iters, p_init=p_init,
        targeted=False, use_ce_for_targeted=True, seed=0
    )

    # 단일 샘플 통계 (기존 함수 재사용)
    S = stats_table_like(xi, x_adv_pix, elapsed, y_true=int(src_idx.item()), count_mode='element', tau_255=1.0)

    with torch.no_grad():
        # src_idx: 텐서([k]) 형태일 수 있으니 int로
        y_idx = int(src_idx.item())
        clean_probs = torch.softmax(predict_logits(xi), dim=1)
        adv_probs   = torch.softmax(predict_logits(x_adv_pix), dim=1)
        clean_conf  = clean_probs[0, y_idx].item()
        adv_conf    = adv_probs[0, y_idx].item()
    conf_drop = clean_conf - adv_conf
    conf_drop_list.append(conf_drop)

    # 샘플별 성공 여부 출력
    pred_orig = predict_logits(xi).argmax(1).item()
    pred_adv  = predict_logits(x_adv_pix).argmax(1).item()
    success   = (pred_orig != pred_adv)
    print(f"[{i+1}/{B}] 공격 {'성공 ✅' if success else '실패 ❌'} ")
    img_path = visualize_attack_quick(
        model, xi, x_adv_pix,
        class_names=class_names, mask_2d=mask_for_vis,
        amp_heat=4.0, out_dir="/content/drive/MyDrive/Adversarial_AI/attack_vis_eps2_noroi", filename_prefix=f"sample{i}", display=True
    )
    print("저장 경로:", img_path)

    # 누적
    x_adv_list.append(x_adv_pix)
    elapsed_list.append(S['time'])
    success_list.append(success)
    l2_list.append(S['L2_255'])
    changed_list.append(S['changed'])
    total_list.append(S['total'])
    dmargin_list.append(S['dmargin'])

def _safe_mean(arr):
    return float(np.mean(arr)) if len(arr) > 0 else float('nan')

ASR = float(np.mean(success_list) * 100.0)
success_idx = [i for i, s in enumerate(success_list) if s]

# --- 성공 케이스 평균 ---
mean_L2_success = _safe_mean([l2_list[i] for i in success_idx])
mean_changed_success = _safe_mean([changed_list[i] / total_list[i] for i in success_idx]) * 100.0
mean_time_success = _safe_mean([elapsed_list[i] for i in success_idx])
mean_dmargin_success = _safe_mean([dmargin_list[i] for i in success_idx])
mean_conf_drop_success = _safe_mean([conf_drop_list[i] for i in success_idx])

# --- 전체 평균 ---
mean_L2_all = _safe_mean(l2_list)
mean_changed_all = _safe_mean(np.array(changed_list) / np.array(total_list)) * 100.0
mean_time_all = _safe_mean(elapsed_list)
mean_dmargin_all = _safe_mean(dmargin_list)
mean_conf_drop_all = _safe_mean(conf_drop_list)

print("\n===== ROI Square L2 (Batch 100) 평균 통계 =====")
print(f"Attack Success Rate (ASR)                : {ASR:.2f}%")

print("\n--- 성공 케이스 평균 ---")
print(f"Mean L2 (0~255 scale)                    : {mean_L2_success:.3f}")
print(f"Mean Changed Pixels (elem)               : {mean_changed_success:.2f}%")
print(f"Mean Generation Time (sec)               : {mean_time_success:.2f}")
print(f"Mean ΔMargin (confidence drop)           : {mean_dmargin_success:.3f}")
print(f"Mean ΔConfidence (softmax)            : {mean_conf_drop_success:.3f}")

print("\n--- 전체 평균 ---")
print(f"Mean L2 (0~255 scale)                    : {mean_L2_all:.3f}")
print(f"Mean Changed Pixels (elem)               : {mean_changed_all:.2f}%")
print(f"Mean Generation Time (sec)               : {mean_time_all:.2f}")
print(f"Mean ΔMargin (confidence drop)           : {mean_dmargin_all:.3f}")
print(f"Mean ΔConfidence (softmax)            : {mean_conf_drop_all:.3f}")

def fgsm_attack_full(model, image, label, eps=8/255.0):
    """
    FGSM 공격 (전체 이미지에 적용, 픽셀 공간)
    Args:
        model: torch.nn.Module
        image: (1,C,H,W) tensor, [0,1] 범위
        label: (1,) tensor
        eps: 0~1 scale (픽셀 단위 섭동 크기)
    """
    device = next(model.parameters()).device
    image = image.clone().detach().to(device).requires_grad_(True)
    label = label.to(device)

    # 모델 입력은 정규화 버전
    output = model(to_norm(image))
    loss = nn.CrossEntropyLoss()(output, label)

    model.zero_grad()
    loss.backward()

    # gradient sign
    grad_sign = image.grad.data.sign()

    # 전체 이미지에 적용 (ROI 없음)
    adv_image = image + eps * grad_sign
    adv_image = torch.clamp(adv_image, 0, 1)

    perturb = adv_image - image
    return adv_image.detach(), perturb.detach()

def visualize_fgsm_full(model, x, x_adv, pert, class_names):
    import matplotlib.pyplot as plt

    # numpy 변환
    orig = x.squeeze().permute(1,2,0).cpu().numpy()
    adv  = x_adv.squeeze().permute(1,2,0).cpu().numpy()
    pert_vis = pert.squeeze().permute(1,2,0).cpu().numpy()

    # 픽셀 범위를 0~1로 클램핑
    orig = np.clip(orig, 0, 1)
    adv  = np.clip(adv, 0, 1)

    # 섭동 시각화 (확대 & 중심을 0.5로 맞추기)
    amp = 10.0
    pert_vis = (pert_vis * amp + 1) / 2
    pert_vis = np.clip(pert_vis, 0, 1)

    # 모델 예측
    with torch.no_grad():
        pred_orig = model(to_norm(x)).argmax(1).item()
        pred_adv  = model(to_norm(x_adv)).argmax(1).item()

    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1)
    plt.title(f"Original\n{class_names[pred_orig]}")
    plt.imshow(orig)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Perturbation (x10)")
    plt.imshow(pert_vis)
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title(f"Adversarial\n{class_names[pred_adv]}")
    plt.imshow(adv)
    plt.axis("off")

    plt.show()

def run_fgsm_full_attack(model, test_set, class_names, eps=8/255.0, n_samples=100):
    """
    테스트셋 앞 n_samples 장에 대해 FGSM 전체 이미지 공격 실행
    (ROI 무시, 전체 이미지에 FGSM 적용)
    """
    device = next(model.parameters()).device
    success_list, l2_list, changed_list, total_list, elapsed_list = [], [], [], [], []
    dmargin_list, conf_drop_list = [], []  # ΔMargin, ΔConfidence 기록

    def _margin_from_logits(logits, y_idx: int):
        # margin = f_y - max_{j != y} f_j
        correct = logits[0, y_idx]
        # mask를 써서 y 제외 최댓값 구함
        mask = torch.ones_like(logits[0], dtype=torch.bool)
        mask[y_idx] = False
        max_other = logits[0][mask].max()
        return (correct - max_other).item()

    for i in range(min(n_samples, len(test_set))):
        sample = test_set[i]

        # Dataset 클래스에 따라 (img, label) 또는 (img, label, mask)
        if len(sample) == 2:
            img, label = sample
        else:
            img, label, _ = sample  # mask는 무시

        x = img.unsqueeze(0).to(device)      # (1,3,224,224)
        y = torch.tensor([label], device=device)
        y_idx = int(label)

        # 공격 실행
        start = time.time()
        x_adv, pert = fgsm_attack_full(model, x, y, eps=eps)
        elapsed = time.time() - start

        # 성능 측정
        with torch.no_grad():
            logits_clean = model(to_norm(x))
            logits_adv   = model(to_norm(x_adv))
            pred_orig = logits_clean.argmax(1).item()
            pred_adv  = logits_adv.argmax(1).item()

        success = (pred_adv != pred_orig)
        success_list.append(success)
        elapsed_list.append(elapsed)

        # L2 & 변경 픽셀 계산
        delta = (x_adv - x).cpu()
        l2_255 = (delta.view(1,-1)*255).norm(p=2).item()
        changed = ((delta.abs()*255) > 1.0).sum().item()
        total = delta.numel()
        l2_list.append(l2_255)
        changed_list.append(changed)
        total_list.append(total)

        # ----- ΔMargin / ΔConfidence 계산 -----
        margin_clean = _margin_from_logits(logits_clean, y_idx)
        margin_adv   = _margin_from_logits(logits_adv,   y_idx)
        dmargin = margin_clean - margin_adv
        dmargin_list.append(float(dmargin))

        probs_clean = torch.softmax(logits_clean, dim=1)[0]
        probs_adv   = torch.softmax(logits_adv,   dim=1)[0]
        conf_drop = (probs_clean[y_idx] - probs_adv[y_idx]).item()
        conf_drop_list.append(float(conf_drop))
        # -----------------------------------

        # 한 장마다 결과 출력
        print(f"[{i+1}/{n_samples}] 공격 {'성공 ✅' if success else '실패 ❌'} "
              f"(원래: {class_names[pred_orig]} → 적대: {class_names[pred_adv]})")

        # 첫 장만 시각화
        if i == 0:
            visualize_fgsm_full(model, x, x_adv, pert, class_names)

    # 안전 평균
    def _safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else float('nan')

    # 전체 평균 통계
    ASR = float(np.mean(success_list) * 100.0)
    mean_L2_255 = float(np.mean(l2_list))
    mean_changed_pct = float(np.mean(np.array(changed_list) / np.array(total_list)) * 100.0)
    mean_time = float(np.mean(elapsed_list))

    # 성공/전체 Δ 지표 평균
    success_idx = [i for i, s in enumerate(success_list) if s]
    mean_dmargin_success = _safe_mean([dmargin_list[i]    for i in success_idx])
    mean_confdrop_success = _safe_mean([conf_drop_list[i] for i in success_idx])

    mean_dmargin_all = _safe_mean(dmargin_list)
    mean_confdrop_all = _safe_mean(conf_drop_list)

    print("\n===== FGSM Full Image (Batch 100) 평균 통계 =====")
    print(f"Attack Success Rate (ASR)            : {ASR:.2f}%")
    print(f"Mean L2 (0~255 scale)                : {mean_L2_255:.3f}")
    print(f"Mean Changed Pixels (elem)           : {mean_changed_pct:.2f}%")
    print(f"Mean Generation Time (sec)           : {mean_time:.2f}")

    print("\n--- 성공 케이스 평균 ---")
    print(f"Mean ΔMargin (confidence gap drop)   : {mean_dmargin_success:.3f}")
    print(f"Mean ΔConfidence (softmax)           : {mean_confdrop_success:.3f}")

    print("\n--- 전체 평균 ---")
    print(f"Mean ΔMargin (confidence gap drop)   : {mean_dmargin_all:.3f}")
    print(f"Mean ΔConfidence (softmax)           : {mean_confdrop_all:.3f}")

    return ASR, mean_L2_255, mean_changed_pct, mean_time, success_list

# test_set 준비 (ROI 포함되어 있지만 무시됨)
test_set = BrainTumorDatasetWithROI(
    "/content/drive/MyDrive/Adversarial_AI/processed_data_np224/Testing",
    transform=transforms.ToTensor()
)

# 공격 실행 (앞에서 정의한 to_norm, fgsm_attack_full, visualize_fgsm_full 필요)
ASR, mean_L2, mean_changed_pct, mean_time, success_list = run_fgsm_full_attack(
    model, test_set, class_names, eps=8/255.0, n_samples=100
)

"""
JSMA (Jacobian-based Saliency Map Attack) 구현 - Flask 웹 애플리케이션용
- 마진 기반 공격 (margin = logit[y] - max_{j≠y} logit[j])
- Adam 스타일 모멘텀 적용
- 동적 k 조절 (정체 시 자동 확대)
- ROI(Region of Interest) 마스크 지원
"""

import torch
import torch.nn.functional as F
import numpy as np


# ==============================================================================
# 유틸리티 함수
# ==============================================================================

def to_pixel(x_norm, mean, std):
    """정규화된 이미지를 픽셀 공간 [0,1]로 변환"""
    if isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean, device=x_norm.device).view(1, -1, 1, 1)
    if isinstance(std, (list, tuple)):
        std = torch.tensor(std, device=x_norm.device).view(1, -1, 1, 1)
    return torch.clamp(x_norm * std + mean, 0.0, 1.0)


def to_norm(x_pix, mean, std):
    """픽셀 공간 이미지를 정규화"""
    if isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean, device=x_pix.device).view(1, -1, 1, 1)
    if isinstance(std, (list, tuple)):
        std = torch.tensor(std, device=x_pix.device).view(1, -1, 1, 1)
    return (torch.clamp(x_pix, 0.0, 1.0) - mean) / std


@torch.no_grad()
def _predict_logits(model, x):
    """로짓 예측 (gradient 없이)"""
    return model(x)


# ==============================================================================
# 마진 및 그래디언트 계산
# ==============================================================================

def _margin_and_grad(model, x, y_idx):
    """
    margin(x) = logit[y] - max_{j≠y} logit[j]

    Args:
        model: 타겟 모델
        x: 입력 이미지 (1,C,H,W)
        y_idx: 타겟 클래스 인덱스

    Returns:
        margin: 마진 값 (float)
        grad: 마진에 대한 그래디언트 (1,C,H,W)
        pred_lbl: 예측된 레이블 (int)
    """
    with torch.enable_grad():
        x = x.clone().detach().requires_grad_(True)
        logits = model(x)
        y = int(y_idx.item() if torch.is_tensor(y_idx) else y_idx)

        c = logits.size(1)
        if c == 2:
            # 이진 분류: 다른 클래스는 1-y
            j_star = 1 - y
        else:
            # 다중 분류: y를 제외한 최대 로짓 찾기
            tmp = logits[0].clone()
            tmp[y] = -1e9
            j_star = int(tmp.argmax().item())

        # 마진 = correct_logit - max_other_logit
        margin_t = logits[0, y] - logits[0, j_star]

        # 그래디언트 계산
        grad = torch.autograd.grad(
            margin_t, x,
            retain_graph=False,
            create_graph=False,
            allow_unused=False
        )[0]

    pred_lbl = int(logits.detach().argmax(dim=1).item())
    return float(margin_t.item()), grad.detach(), pred_lbl


# ==============================================================================
# JSMA 공격 (핵심 알고리즘)
# ==============================================================================

def jsma_attack_margin_mom(
    model,
    x,                          # (1,C,H,W) in [0,1]
    y_true,                     # (1,) or int
    theta=0.08,                 # 기본 스텝 크기
    max_pixels_percentage=0.05, # 변경 허용 픽셀 비율 (5%)
    k_small=2,                  # 초기 k (동시 변경 픽셀 수)
    k_big=12,                   # 정체 시 k 확대 상한
    patience=3,                 # margin 개선 정체 허용 스텝
    momentum=0.75,              # gradient EMA 모멘텀
    restarts=4,                 # 재시도 횟수
    topk_pool=5000,             # 후보 풀 크기
    allowed_masks=None,         # (H,W) bool ROI 마스크
    clamp=(0.0, 1.0),          # 픽셀 값 범위
    early_success=True          # 조기 성공 체크
):
    """
    JSMA 공격 (마진 기반 + 모멘텀 + 동적 k 조절)

    Args:
        model: 타겟 모델
        x: 원본 이미지 (1,C,H,W) [0,1]
        y_true: 실제 레이블 (1,) 또는 int
        theta: 픽셀 변경 스텝 크기
        max_pixels_percentage: 최대 변경 픽셀 비율
        k_small: 초기 동시 변경 픽셀 수
        k_big: 정체 시 k 확대 상한
        patience: margin 개선 정체 허용 스텝 수
        momentum: gradient 모멘텀 (EMA)
        restarts: 재시도 횟수
        topk_pool: 후보 픽셀 풀 크기
        allowed_masks: ROI 마스크 (H,W) bool
        clamp: 픽셀 값 범위
        early_success: 조기 성공 체크

    Returns:
        x_adv: 공격된 이미지 (1,C,H,W)
        changed_spatial: 변경된 픽셀 수 (tensor)
        l1_total: L1 섭동 총합 (tensor)
        success: 공격 성공 여부 (tensor 0/1)
    """
    device = x.device
    _, C, H, W = x.shape
    budget = int(max_pixels_percentage * H * W)
    best = None

    # y_true를 정수로 변환
    if torch.is_tensor(y_true):
        if y_true.dim() == 0:
            y_true = y_true.unsqueeze(0)
    else:
        y_true = torch.tensor([y_true], device=device)

    # ROI 마스크 설정
    if allowed_masks is not None:
        roi = allowed_masks.bool().to(device)
        if roi.dim() == 3:
            roi = roi.any(dim=0)
    else:
        roi = torch.ones((H, W), dtype=torch.bool, device=device)

    # 재시도 루프
    for restart_idx in range(restarts):
        x_adv = x.clone().detach()
        v = torch.zeros_like(x_adv)
        changed_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        last_margins = []
        changed_spatial = 0

        step_theta = theta
        k = k_small
        success = False

        # 공격 루프
        while changed_spatial < budget:
            # 현재 margin / grad / pred 계산
            margin, g, pred_lbl = _margin_and_grad(model, x_adv, y_true)
            last_margins.append(margin)
            if len(last_margins) > patience + 1:
                last_margins.pop(0)

            # 이미 성공했는가?
            if early_success and pred_lbl != int(y_true.item()):
                success = True
                break

            # 모멘텀 업데이트 (EMA)
            v = momentum * v + (1.0 - momentum) * g

            # Saliency map 계산 (공간 단위)
            sal = v.abs().sum(dim=1, keepdim=False)[0]

            # 업데이트 불가 위치 마스킹
            eligible = roi & (~changed_mask)

            with torch.no_grad():
                sign_v = v.sign()[0]
                up_block = (sign_v > 0) & (x_adv[0] >= clamp[1] - 1e-6)
                down_block = (sign_v < 0) & (x_adv[0] <= clamp[0] + 1e-6)
                blocked = up_block.any(dim=0) | down_block.any(dim=0)
                eligible = eligible & (~blocked)

            if not eligible.any():
                break

            # Top-k 픽셀 선택
            sal_masked = sal.masked_fill(~eligible, float('-inf'))
            flat = sal_masked.view(-1)
            k_pool = min(topk_pool, flat.numel())
            idx_pool = flat.topk(k_pool, largest=True).indices

            if idx_pool.numel() == 0 or torch.isneginf(flat[idx_pool[0]]):
                break

            k_eff = min(k, idx_pool.numel(), budget - changed_spatial)
            idx_pick = idx_pool[:k_eff]
            ys = (idx_pick // W).long()
            xs = (idx_pick % W).long()

            # 벡터화 업데이트
            with torch.no_grad():
                upd = step_theta * v[0, :, ys, xs].sign()
                x_adv[0, :, ys, xs] = (x_adv[0, :, ys, xs] - upd).clamp_(clamp[0], clamp[1])
                changed_mask[ys, xs] = True
                changed_spatial += k_eff

            # Margin 정체 시 k/θ 동적 조절
            if len(last_margins) >= patience + 1:
                improvement = last_margins[0] - last_margins[-1]
                if improvement < 1e-4:
                    k = min(k_big, k * 2)
                    step_theta = min(step_theta * 1.25, 0.2)
                else:
                    k = max(k_small, k // 2)
                    step_theta = max(step_theta * 0.95, theta * 0.5)

        # 최종 결과 집계
        with torch.no_grad():
            diff = (x_adv - x).abs()
            l1_total = float(diff.sum().item())
            pred_final = _predict_logits(model, x_adv).argmax(dim=1)
            success = success or (int(pred_final.item()) != int(y_true.item()))
            fin_margin, _, _ = _margin_and_grad(model, x_adv, y_true)

        cand = (success, changed_spatial, -fin_margin, x_adv.detach(), l1_total)

        if best is None:
            best = cand
        else:
            if (cand[0] and not best[0]) \
               or (cand[0] and best[0] and cand[1] < best[1]) \
               or (cand[0] == best[0] and cand[1] == best[1] and cand[2] > best[2]):
                best = cand

    # 최종 결과 반환
    success, changed_spatial, _neg_margin, x_best, l1_total = best
    return (
        x_best,
        torch.tensor(changed_spatial, dtype=torch.int32),
        torch.tensor(l1_total, dtype=torch.float32),
        torch.tensor(1 if success else 0, dtype=torch.int32)
    )


# ==============================================================================
# Flask 웹 애플리케이션용 인터페이스
# ==============================================================================

def jsma_attack(
    model,
    x_pix,              # (1,C,H,W) in [0,1]
    y_true,             # int or tensor
    mean,               # 정규화 평균 (호환성 유지)
    std,                # 정규화 표준편차 (호환성 유지)
    theta=0.08,         # 스텝 크기
    max_pixels_pct=0.05, # 최대 변경 픽셀 비율
    k_small=2,          # 초기 k
    restarts=4,         # 재시도 횟수
    topk_pool=5000,     # 후보 풀 크기
    roi_mask=None       # ROI 마스크
):
    """
    JSMA 공격 - Flask 애플리케이션용 인터페이스

    Args:
        model: 타겟 모델
        x_pix: 픽셀 공간 이미지 (1,C,H,W) [0,1]
        y_true: 실제 레이블
        mean: 정규화 평균값 (호환성 유지용)
        std: 정규화 표준편차 (호환성 유지용)
        theta: 스텝 크기
        max_pixels_pct: 최대 변경 픽셀 비율
        k_small: 초기 동시 변경 픽셀 수
        restarts: 재시도 횟수
        topk_pool: 후보 풀 크기
        roi_mask: ROI 마스크 (H,W) bool

    Returns:
        x_adv_pix: 공격된 픽셀 공간 이미지
        changed_spatial: 변경된 픽셀 수
        l1_total: L1 섭동 총합
        success: 공격 성공 여부
    """
    # y_true를 텐서로 변환
    if not torch.is_tensor(y_true):
        y_true = torch.tensor([y_true], device=x_pix.device)
    elif y_true.dim() == 0:
        y_true = y_true.unsqueeze(0)

    # ROI 마스크 처리
    if roi_mask is not None:
        if torch.is_tensor(roi_mask):
            if roi_mask.dim() == 3:
                roi_mask = roi_mask.any(dim=0)
        else:
            roi_mask = torch.as_tensor(roi_mask, device=x_pix.device)

    # JSMA 공격 실행
    x_adv, changed, l1, success = jsma_attack_margin_mom(
        model=model,
        x=x_pix,
        y_true=y_true,
        theta=theta,
        max_pixels_percentage=max_pixels_pct,
        k_small=k_small,
        k_big=max(12, k_small * 2),
        patience=3,
        momentum=0.75,
        restarts=restarts,
        topk_pool=topk_pool,
        allowed_masks=roi_mask,
        clamp=(0.0, 1.0),
        early_success=True
    )

    return x_adv, changed, l1, success


# ==============================================================================
# 메트릭 계산 함수
# ==============================================================================

@torch.no_grad()
def calculate_jsma_metrics(x_original, x_adv, y_true, model, threshold=1.0):
    """
    JSMA 공격 결과 메트릭 계산

    Args:
        x_original: 원본 이미지 (1,C,H,W)
        x_adv: 공격된 이미지 (1,C,H,W)
        y_true: 실제 레이블
        model: 타겟 모델
        threshold: 변경 픽셀 판정 임계값 (0-255 스케일)

    Returns:
        metrics: 메트릭 딕셔너리
    """
    # 섭동 계산
    delta = (x_adv - x_original).abs()

    # L1, L2, Linf 노름
    l1_norm = delta.sum().item()
    l2_norm = delta.pow(2).sum().sqrt().item()
    linf_norm = delta.max().item()

    # 변경된 픽셀 수 (0-255 스케일)
    delta_255 = delta * 255.0
    changed_pixels = (delta_255 > threshold).sum().item()
    total_pixels = delta.numel()
    changed_ratio = changed_pixels / total_pixels * 100.0

    # 예측 결과
    logits_original = model(x_original)
    logits_adv = model(x_adv)

    pred_original = logits_original.argmax(1).item()
    pred_adv = logits_adv.argmax(1).item()

    # 신뢰도 변화
    prob_original = torch.softmax(logits_original, dim=1)
    prob_adv = torch.softmax(logits_adv, dim=1)

    if torch.is_tensor(y_true):
        y_idx = int(y_true.item())
    else:
        y_idx = int(y_true)

    conf_original = prob_original[0, y_idx].item()
    conf_adv = prob_adv[0, y_idx].item()
    conf_drop = conf_original - conf_adv

    # 마진 변화
    margin_original, _, _ = _margin_and_grad(model, x_original, y_true)
    margin_adv, _, _ = _margin_and_grad(model, x_adv, y_true)
    margin_drop = margin_original - margin_adv

    # 성공 여부
    success = (pred_adv != y_idx)

    return {
        'l1_norm': l1_norm,
        'l2_norm': l2_norm,
        'linf_norm': linf_norm,
        'changed_pixels': changed_pixels,
        'total_pixels': total_pixels,
        'changed_ratio': changed_ratio,
        'pred_original': pred_original,
        'pred_adv': pred_adv,
        'conf_original': conf_original,
        'conf_adv': conf_adv,
        'conf_drop': conf_drop,
        'margin_original': margin_original,
        'margin_adv': margin_adv,
        'margin_drop': margin_drop,
        'success': success
    }

import time
import numpy as np
import torch

def run_jsma_full_attack(
    model,
    test_set,
    class_names,
    mean, std,                     # jsma_attack 인터페이스 요구
    theta=0.08,
    max_pixels_pct=0.05,
    k_small=2,
    restarts=4,
    topk_pool=5000,
    n_samples=100,
    visualize_first=True,
    viz_func=None                  # 시각화 함수(옵션). 예: visualize_attack_quick
):
    """
    Full-image JSMA 공격을 테스트셋 앞 n_samples 장에 대해 실행 (ROI 무시).
    Returns:
        ASR, mean_L1, mean_changed_count, mean_time, success_list
    """
    device = next(model.parameters()).device
    success_list = []
    l1_list = []
    changed_count_list = []
    total_list = []
    elapsed_list = []
    dmargin_list = []
    conf_drop_list = []

    for i in range(min(n_samples, len(test_set))):
        sample = test_set[i]
        # dataset 형식 (img, label) 또는 (img, label, mask)
        if len(sample) == 2:
            img, label = sample
        else:
            img, label, _ = sample

        # 준비
        x = img.unsqueeze(0).to(device)    # (1,C,H,W)
        y = int(label)

        # 공격 실행
        t0 = time.time()
        x_adv, changed_spatial, l1_total, success_flag = jsma_attack(
            model=model,
            x_pix=x,
            y_true=y,
            mean=mean, std=std,
            theta=theta,
            max_pixels_pct=max_pixels_pct,
            k_small=k_small,
            restarts=restarts,
            topk_pool=topk_pool,
            roi_mask=None
        )
        elapsed = time.time() - t0

        # 메트릭 계산
        metrics = calculate_jsma_metrics(x, x_adv, torch.tensor([y], device=device), model, threshold=1.0)

        # 수집
        success = bool(metrics['success'])
        success_list.append(success)
        elapsed_list.append(elapsed)
        l1_list.append(metrics['l1_norm'])
        changed_count_list.append(metrics['changed_pixels'])
        total_list.append(metrics['total_pixels'])
        dmargin_list.append(metrics['margin_drop'])
        conf_drop_list.append(metrics['conf_drop'])

        # 출력 (한 장마다)
        pred_orig = metrics['pred_original']
        pred_adv  = metrics['pred_adv']
        print(f"[{i+1}/{n_samples}] 공격 {'성공 ✅' if success else '실패 ❌'} "
              f"(원래: {class_names[pred_orig]} → 적대: {class_names[pred_adv]}) "
              f"L1={metrics['l1_norm']:.1f} changed={metrics['changed_pixels']}/{metrics['total_pixels']} "
              f"ΔConf={metrics['conf_drop']:.3f} ΔMargin={metrics['margin_drop']:.3f} time={elapsed:.2f}s")

        # 첫 장만 시각화 (옵션)
        if i == 0 and visualize_first:
            # 안전하게 시각화 함수 호출 (사용자 환경에 따라 함수명이 다를 수 있으므로 try/except)
            if viz_func is not None:
                try:
                    viz_func(model, x, x_adv, class_names=class_names)
                except Exception as e:
                    print("[VIS] visualization failed:", e)
            else:
                # 기본적으로 존재하는 visualize_attack_quick 같은 함수가 있으면 사용
                try:
                    visualize_attack_quick(model, x, x_adv, class_names=class_names)
                except Exception:
                    pass

    # 집계(안전 평균)
    def _safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else float('nan')

    ASR = float(np.mean(success_list) * 100.0)
    mean_L1 = _safe_mean(l1_list)
    mean_changed_count = _safe_mean(changed_count_list)
    mean_time = _safe_mean(elapsed_list)

    # 성공 케이스 평균 (선택)
    success_idx = [idx for idx, s in enumerate(success_list) if s]
    mean_changed_count_success = _safe_mean([changed_count_list[i] for i in success_idx])
    mean_dmargin_success = _safe_mean([dmargin_list[i] for i in success_idx])
    mean_confdrop_success = _safe_mean([conf_drop_list[i] for i in success_idx])

    # 출력 (요약)
    print("\n===== JSMA Full Image (Batch) 평균 통계 =====")
    print(f"Attack Success Rate (ASR)            : {ASR:.2f}%")
    print(f"Mean L1 (pixel space, 0~1)           : {mean_L1:.3f}")
    print(f"Mean Changed Pixels (count, all)     : {mean_changed_count:.1f}")
    if len(total_list) > 0:
        print(f"Per-image pixel count (one image)    : {int(total_list[0])}")
    print(f"Mean Generation Time (sec)           : {mean_time:.2f}")

    print("\n--- 성공 케이스 평균 ---")
    print(f"Mean Changed Pixels (count, success) : {mean_changed_count_success:.1f}")
    print(f"Mean ΔMargin (success)               : {mean_dmargin_success:.3f}")
    print(f"Mean ΔConfidence (success)           : {mean_confdrop_success:.3f}")

    return ASR, mean_L1, mean_changed_count, mean_time, success_list

import os, time
import torch
import matplotlib.pyplot as plt

def _tensor_img_to_numpy(x):  # (1,3,H,W) -> (H,W,3)
    if x.dim() == 4: x = x.squeeze(0)
    return x.detach().clamp(0,1).permute(1,2,0).cpu().numpy()

def _pred_and_conf_from_logits(logits):
    probs = torch.softmax(logits, dim=1)
    pred  = probs.argmax(1).item()
    conf  = probs[0, pred].item()
    return pred, conf

def visualize_attack_quick_fixed(
    model,
    x_one,          # (1,C,H,W) pixel space [0,1]
    x_adv,          # (1,C,H,W) pixel space [0,1]
    class_names=None,
    mask_2d=None,
    amp_heat=3.0,
    out_dir=".",
    filename_prefix="attack_vis",
    display=True,
    mean=None,      # <-- 추가: 정규화 평균 (list or tensor)
    std=None        # <-- 추가: 정규화 std
):
    """
    수정된 시각화: 모델에 넣기 전에 to_norm(x_pix, mean, std)로 정규화해서 사용합니다.
    mean, std는 list/tuple or tensor (channel-wise) 형태여야 합니다.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = x_one.device
    print("[VIS] visualize_attack_quick_fixed start")

    # 입력 유효성
    if mean is None or std is None:
        raise ValueError("visualize_attack_quick_fixed requires mean and std to normalize inputs for model.")

    # 정규화 함수 (간단 구현: mean/std can be list/tuple or tensor)
    def _to_norm_local(x_pix):
        if isinstance(mean, (list, tuple)):
            m = torch.tensor(mean, device=x_pix.device).view(1, -1, 1, 1)
        else:
            m = mean.to(x_pix.device).view(1, -1, 1, 1)
        if isinstance(std, (list, tuple)):
            s = torch.tensor(std, device=x_pix.device).view(1, -1, 1, 1)
        else:
            s = std.to(x_pix.device).view(1, -1, 1, 1)
        return (torch.clamp(x_pix, 0.0, 1.0) - m) / s

    # 예측/신뢰도 (정규화해서 모델에 넣음)
    with torch.no_grad():
        logits_before = model(_to_norm_local(x_one))
        logits_after  = model(_to_norm_local(x_adv))
    pred_b, conf_b = _pred_and_conf_from_logits(logits_before)
    pred_a, conf_a = _pred_and_conf_from_logits(logits_after)

    # 시각화 데이터
    img_b = _tensor_img_to_numpy(x_one)
    img_a = _tensor_img_to_numpy(x_adv)
    # perturbation magnitude (per-spatial L2 across channels)
    delta = (x_adv - x_one).squeeze(0).norm(p=2, dim=0).cpu().numpy()
    pert_map_vis = delta * amp_heat

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot(1,3,1); ax.imshow(img_b); ax.set_title("Original"); ax.axis('off')
    if mask_2d is not None:
        m = (mask_2d.detach().cpu().numpy() if torch.is_tensor(mask_2d) else mask_2d).astype(float)
        ax.contour(m, levels=[0.5], linewidths=1.5, colors='cyan')

    ax = plt.subplot(1,3,2); ax.imshow(img_a); ax.set_title("Adversarial"); ax.axis('off')
    if mask_2d is not None:
        ax.contour(m, levels=[0.5], linewidths=1.5, colors='cyan')

    ax = plt.subplot(1,3,3); im = ax.imshow(pert_map_vis, cmap='magma'); plt.colorbar(im)
    ax.set_title(f"|δ| (x{amp_heat:g})"); ax.axis('off')

    name = (lambda i: class_names[i] if (class_names and 0 <= i < len(class_names)) else str(i))
    title = f"Before: {name(pred_b)}({conf_b:.3f}) → After: {name(pred_a)}({conf_a:.3f})"
    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()

    ts = int(time.time())
    out_path = os.path.join(out_dir, f"{filename_prefix}_{ts}.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"[VIS] Saved to: {out_path}")

    if display:
        plt.show()
    else:
        plt.close(fig)

    return out_path

# 모델이 내부에서 이미 정규화하는 경우:
mean_vis = [0.0, 0.0, 0.0]
std_vis  = [1.0, 1.0, 1.0]

# viz_wrapper 만들 때 이렇게 전달
def viz_wrapper(model, x, x_adv, class_names=None):
    return visualize_attack_quick_fixed(
        model, x, x_adv, class_names=class_names,
        amp_heat=4.0,
        out_dir="/content/drive/MyDrive/Adversarial_AI/attack_vis_jsma",
        filename_prefix="jsma_sample",
        display=True,
        mean=mean_vis, std=std_vis
    )

# 호출 (예시)
ASR, mean_L1, mean_changed_count, mean_time, success_list = run_jsma_full_attack(
    model=model,
    test_set=test_set,
    class_names=class_names,
    mean=mean_vis,
    std=std_vis,
    theta=0.08,
    max_pixels_pct=0.05,
    k_small=2,
    restarts=4,
    topk_pool=5000,
    n_samples=10,
    visualize_first=True,
    viz_func=viz_wrapper
)

# 필요한 라이브러리
import io, time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# ------------------------------
# 1) JPEG 압축 유틸 (PIL 사용)
# ------------------------------
def jpeg_compress_tensor_pil(x_tensor, quality=75):
    """
    x_tensor: torch.Tensor (1,C,H,W) or (C,H,W) in [0,1] (float)
    returns: same-shape tensor in [0,1] on same device as input (moved to cpu during PIL ops)
    """
    single = False
    if x_tensor.dim() == 3:  # (C,H,W) -> (1,C,H,W)
        x_tensor = x_tensor.unsqueeze(0)
        single = True

    device = x_tensor.device
    x = x_tensor.detach().cpu().clamp(0,1)
    bs, c, h, w = x.shape
    out = torch.empty_like(x)

    for i in range(bs):
        arr = (x[i].permute(1,2,0).numpy() * 255.0).astype(np.uint8)  # H,W,C
        pil = Image.fromarray(arr)
        bio = io.BytesIO()
        pil.save(bio, format='JPEG', quality=int(quality))
        bio.seek(0)
        pil2 = Image.open(bio).convert('RGB')
        arr2 = np.asarray(pil2).astype(np.float32) / 255.0
        out[i] = torch.from_numpy(arr2).permute(2,0,1)

    out = out.to(device)
    return out.squeeze(0) if single else out

# ------------------------------
# 2) margin 계산 유틸
# ------------------------------
def margin_from_logits(logits, y_idx):
    """
    logits: tensor shape (1, num_classes)
    y_idx: int
    returns margin = logit[y] - max_{j!=y} logit[j] (float)
    """
    with torch.no_grad():
        logits0 = logits[0]
        tmp = logits0.clone()
        tmp[y_idx] = -1e9
        max_other = tmp.max().item()
        margin = float((logits0[y_idx].item() - max_other))
    return margin

# ------------------------------
# 3) per-sample evaluation helper
# ------------------------------
@torch.no_grad()
def eval_example_with_jpeg(model, x_orig_pix, x_adv_pix, mean, std, quality=75, y_true=None):
    """
    Evaluates one example before/after JPEG.
    x_orig_pix, x_adv_pix: (1,C,H,W) tensors in pixel space [0,1]
    mean/std: list/tuple or tensor for to_norm
    quality: JPEG quality for defense
    y_true: int or None (if provided used for margin/confidence drop computation)
    Returns: dict of metrics
    """
    device = next(model.parameters()).device

    # Normalize helper (accept list or tensor)
    def _to_norm_local(x_pix):
        if isinstance(mean, (list, tuple)):
            m = torch.tensor(mean, device=x_pix.device).view(1, -1, 1, 1)
        else:
            m = mean.to(x_pix.device).view(1, -1, 1, 1)
        if isinstance(std, (list, tuple)):
            s = torch.tensor(std, device=x_pix.device).view(1, -1, 1, 1)
        else:
            s = std.to(x_pix.device).view(1, -1, 1, 1)
        return (torch.clamp(x_pix, 0.0, 1.0) - m) / s

    # model inputs: normalized
    x_norm = _to_norm_local(x_orig_pix.to(device))
    xadv_norm = _to_norm_local(x_adv_pix.to(device))

    logits_clean = model(x_norm)
    logits_adv   = model(xadv_norm)

    probs_clean = torch.softmax(logits_clean, dim=1)[0]
    probs_adv   = torch.softmax(logits_adv, dim=1)[0]
    pred_clean = int(probs_clean.argmax().item())
    pred_adv   = int(probs_adv.argmax().item())
    conf_clean = float(probs_clean[pred_clean].item())
    conf_adv   = float(probs_adv[pred_adv].item())

    # margin/confidence wrt true label if provided; else use orig predicted label
    if y_true is None:
        y_idx = pred_clean
    else:
        y_idx = int(y_true)

    margin_clean = margin_from_logits(logits_clean, y_idx)
    margin_adv   = margin_from_logits(logits_adv, y_idx)
    margin_drop = margin_clean - margin_adv
    conf_drop = float((torch.softmax(logits_clean, dim=1)[0, y_idx] - torch.softmax(logits_adv, dim=1)[0, y_idx]).item())

    # Now apply JPEG compression to adversarial pixel image
    x_adv_jpeg_pix = jpeg_compress_tensor_pil(x_adv_pix, quality=quality).to(device)
    xadv_jpeg_norm = _to_norm_local(x_adv_jpeg_pix)

    logits_adv_jpeg = model(xadv_jpeg_norm)
    probs_adv_jpeg = torch.softmax(logits_adv_jpeg, dim=1)[0]
    pred_adv_jpeg = int(probs_adv_jpeg.argmax().item())
    conf_adv_jpeg = float(probs_adv_jpeg[pred_adv_jpeg].item())

    margin_adv_jpeg = margin_from_logits(logits_adv_jpeg, y_idx)
    margin_drop_after = margin_clean - margin_adv_jpeg
    conf_drop_after = float((torch.softmax(logits_clean, dim=1)[0, y_idx] - probs_adv_jpeg[y_idx]).item())

    # perturbation norms (in pixel space)
    delta_adv = (x_adv_pix - x_orig_pix).detach().cpu()
    l2_255 = (delta_adv.view(1, -1) * 255.0).norm(p=2).item()
    l1 = delta_adv.abs().sum().item()
    linf = delta_adv.abs().max().item()

    # changed pixels count (0-255 threshold)
    changed_pixels = int(((delta_adv * 255.0) > 1.0).sum().item())
    total_pixels = int(delta_adv.numel())

    return {
        'pred_clean': pred_clean,
        'pred_adv': pred_adv,
        'pred_adv_jpeg': pred_adv_jpeg,
        'conf_clean': conf_clean,
        'conf_adv': conf_adv,
        'conf_adv_jpeg': conf_adv_jpeg,
        'margin_clean': margin_clean,
        'margin_adv': margin_adv,
        'margin_adv_jpeg': margin_adv_jpeg,
        'margin_drop_before': margin_drop,
        'margin_drop_after': margin_drop_after,
        'conf_drop_before': conf_drop,
        'conf_drop_after': conf_drop_after,
        'l2_255': l2_255,
        'l1': l1,
        'linf': linf,
        'changed_pixels': changed_pixels,
        'total_pixels': total_pixels,
        'success_before': (pred_adv != pred_clean),
        'success_after_jpeg': (pred_adv_jpeg != pred_clean),
        'x_adv_jpeg_pix': x_adv_jpeg_pix  # returned for possible visualization
    }

# ------------------------------
# 4) 전체 실험 배치 실행 함수
# ------------------------------
def run_jsma_and_eval_jpeg_defense(
    model,
    test_set,
    class_names,
    mean, std,
    n_samples=100,
    jsma_params=None,
    jpeg_quality=75,
    visualize_first=True,
    viz_func=None
):
    """
    1) 각 샘플에 대해 jsma_attack -> x_adv
    2) eval_example_with_jpeg: 평가 전/후
    3) 집계 출력 및 결과 반환
    """
    device = next(model.parameters()).device

    # 결과 수집
    success_before_list = []
    success_after_list = []
    l2_list = []
    changed_list = []
    total_list = []
    margin_drop_before_list = []
    margin_drop_after_list = []
    conf_drop_before_list = []
    conf_drop_after_list = []
    times = []

    # default jsma params wrapper
    def _call_jsma(x, y):
        # jsma_attack signature: jsma_attack(model, x_pix, y_true, mean, std, ...)
        params = jsma_params or {}
        return jsma_attack(model=model, x_pix=x, y_true=y, mean=mean, std=std, **params)

    n_run = min(n_samples, len(test_set))
    for i in range(n_run):
        sample = test_set[i]
        if len(sample) == 2:
            img, label = sample
        else:
            img, label, _ = sample

        x = img.unsqueeze(0).to(device)
        y = int(label)

        # JSMA 공격
        t0 = time.time()
        x_adv, changed_spatial, l1_total, success_flag = _call_jsma(x, y)
        elapsed = time.time() - t0

        # Ensure x_adv is tensor on same device and shape (1,C,H,W)
        x_adv = x_adv.to(device)
        times.append(elapsed)

        # Evaluate before/after JPEG
        metrics = eval_example_with_jpeg(model, x, x_adv, mean, std, quality=jpeg_quality, y_true=y)

        # collect
        success_before_list.append(int(metrics['success_before']))
        success_after_list.append(int(metrics['success_after_jpeg']))
        l2_list.append(metrics['l2_255'])
        changed_list.append(metrics['changed_pixels'])
        total_list.append(metrics['total_pixels'])
        margin_drop_before_list.append(metrics['margin_drop_before'])
        margin_drop_after_list.append(metrics['margin_drop_after'])
        conf_drop_before_list.append(metrics['conf_drop_before'])
        conf_drop_after_list.append(metrics['conf_drop_after'])

        # print per-sample
        print(f"[{i+1}/{n_run}] JSMA success_before={metrics['success_before']} "
              f"-> after JPEG success={metrics['success_after_jpeg']} | "
              f"pred: {class_names[metrics['pred_clean']]} -> adv:{class_names[metrics['pred_adv']]} -> adv+jpeg:{class_names[metrics['pred_adv_jpeg']]} "
              f"| L2_255={metrics['l2_255']:.3f} changed={metrics['changed_pixels']}/{metrics['total_pixels']} time={elapsed:.2f}s")

        # inside the loop, after metrics computed
        if i == 0 and visualize_first:
            try:
                visualize_attack_with_jpeg(
                    model,
                    x_orig_pix=x,
                    x_adv_pix=x_adv,
                    x_adv_jpeg_pix=metrics['x_adv_jpeg_pix'],
                    class_names=class_names,
                    mean=mean, std=std,
                    amp_delta=3.0,
                    out_dir="./attack_vis_jpeg",
                    filename_prefix="sample"
                )
            except Exception as e:
                print("[VIS] jpeg visualization failed:", e)


    # Aggregates
    ASR_before = float(np.mean(success_before_list) * 100.0)
    ASR_after  = float(np.mean(success_after_list) * 100.0)
    mean_L2_255 = float(np.mean(l2_list))
    mean_changed_count = float(np.mean(changed_list))
    mean_changed_pct = float(np.mean(np.array(changed_list) / np.array(total_list)) * 100.0)
    mean_time = float(np.mean(times))
    mean_margin_drop_before = float(np.mean(margin_drop_before_list))
    mean_margin_drop_after = float(np.mean(margin_drop_after_list))
    mean_conf_drop_before = float(np.mean(conf_drop_before_list))
    mean_conf_drop_after = float(np.mean(conf_drop_after_list))

    print("\n===== JSMA -> JPEG Defense Summary =====")
    print(f"Samples evaluated               : {n_run}")
    print(f"ASR before (JSMA)              : {ASR_before:.2f}%")
    print(f"ASR after  (JSMA -> JPEG q={jpeg_quality}) : {ASR_after:.2f}%")
    print(f"Mean L2 (0~255)                : {mean_L2_255:.3f}")
    print(f"Mean Changed Pixels (count)    : {mean_changed_count:.1f} ({mean_changed_pct:.2f}%)")
    print(f"Mean Generation Time (sec)     : {mean_time:.2f}")
    print(f"Mean ΔMargin before            : {mean_margin_drop_before:.3f}")
    print(f"Mean ΔMargin after (post-JPEG) : {mean_margin_drop_after:.3f}")
    print(f"Mean ΔConfidence before        : {mean_conf_drop_before:.3f}")
    print(f"Mean ΔConfidence after (post)  : {mean_conf_drop_after:.3f}")

    results = {
        'ASR_before': ASR_before,
        'ASR_after': ASR_after,
        'mean_L2_255': mean_L2_255,
        'mean_changed_count': mean_changed_count,
        'mean_changed_pct': mean_changed_pct,
        'mean_time': mean_time,
        'mean_margin_drop_before': mean_margin_drop_before,
        'mean_margin_drop_after': mean_margin_drop_after,
        'mean_conf_drop_before': mean_conf_drop_before,
        'mean_conf_drop_after': mean_conf_drop_after,
        'per_sample': {
            'success_before_list': success_before_list,
            'success_after_list': success_after_list,
            'l2_list': l2_list,
            'changed_list': changed_list,
            'total_list': total_list,
            'margin_drop_before': margin_drop_before_list,
            'margin_drop_after': margin_drop_after_list,
            'conf_drop_before': conf_drop_before_list,
            'conf_drop_after': conf_drop_after_list,
            'times': times
        }
    }

    return results

import os, time
import torch
import numpy as np
import matplotlib.pyplot as plt

def _to_norm_local(x_pix, mean, std):
    if isinstance(mean, (list, tuple)):
        m = torch.tensor(mean, device=x_pix.device).view(1, -1, 1, 1)
    else:
        m = mean.to(x_pix.device).view(1, -1, 1, 1)
    if isinstance(std, (list, tuple)):
        s = torch.tensor(std, device=x_pix.device).view(1, -1, 1, 1)
    else:
        s = std.to(x_pix.device).view(1, -1, 1, 1)
    return (torch.clamp(x_pix, 0.0, 1.0) - m) / s

def _pred_and_conf(model, x_pix, mean, std):
    with torch.no_grad():
        logits = model(_to_norm_local(x_pix, mean, std))
        probs = torch.softmax(logits, dim=1)
        pred  = int(probs.argmax(1).item())
        conf  = float(probs[0, pred].item())
    return pred, conf

def _to_numpy_img(x):
    if x.dim()==4: x = x.squeeze(0)
    return x.detach().clamp(0,1).permute(1,2,0).cpu().numpy()

def visualize_attack_with_jpeg(
    model,
    x_orig_pix,                 # (1,C,H,W) [0,1]
    x_adv_pix,                  # (1,C,H,W) [0,1]
    x_adv_jpeg_pix,             # (1,C,H,W) [0,1]  ← JPEG 적용 결과
    class_names=None,
    mean=None, std=None,        # 모델 입력 정규화
    amp_delta=3.0,
    out_dir=".", filename_prefix="attack_vis_jpeg", display=True
):
    os.makedirs(out_dir, exist_ok=True)

    # 예측/신뢰도
    p0,c0 = _pred_and_conf(model, x_orig_pix,      mean, std)
    p1,c1 = _pred_and_conf(model, x_adv_pix,       mean, std)
    p2,c2 = _pred_and_conf(model, x_adv_jpeg_pix,  mean, std)

    # 이미지/맵 준비
    img0 = _to_numpy_img(x_orig_pix)
    img1 = _to_numpy_img(x_adv_pix)
    img2 = _to_numpy_img(x_adv_jpeg_pix)

    # 제거된 섭동(= JPEG가 없앤 성분): δ_removed = x_adv - x_adv_jpeg
    delta_removed = (x_adv_pix - x_adv_jpeg_pix).squeeze(0).norm(p=2, dim=0).cpu().numpy()
    delta_removed_vis = delta_removed * amp_delta

    # 그리기 (1x4)
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,4,1); ax.imshow(img0); ax.set_title(f"Original\n{class_names[p0]}({c0:.3f})"); ax.axis('off')
    ax = plt.subplot(1,4,2); ax.imshow(img1); ax.set_title(f"Adversarial\n{class_names[p1]}({c1:.3f})"); ax.axis('off')
    ax = plt.subplot(1,4,3); ax.imshow(img2); ax.set_title(f"Adv + JPEG\n{class_names[p2]}({c2:.3f})"); ax.axis('off')
    ax = plt.subplot(1,4,4); im = ax.imshow(delta_removed_vis, cmap='magma'); plt.colorbar(im)
    ax.set_title(f"|δ_removed| (x{amp_delta:g})"); ax.axis('off')

    title = f"Before→After: {class_names[p0]}({c0:.3f}) → {class_names[p1]}({c1:.3f}) → {class_names[p2]}({c2:.3f})"
    plt.suptitle(title, y=1.03)
    plt.tight_layout()

    ts = int(time.time())
    save_path = os.path.join(out_dir, f"{filename_prefix}_{ts}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if display: plt.show()
    else: plt.close(fig)
    print(f"[VIS] Saved to: {save_path}")
    return save_path

# model: 이미 로드되어 있고 model.eval() 되어 있어야 함
# test_set: BrainTumorDatasetWithROI 혹은 similar (returns img,label[,mask]) with transforms.ToTensor()
# class_names: list of names
mean = [0,0,0]   # 또는 model에 맞게 (혹은 [0,0,0] if model does internal norm)
std  = [1,1,1]

jsma_params = {
    'theta': 0.12,
    'max_pixels_pct': 0.10,
    'k_small': 4,
    'restarts': 4,
    'topk_pool': 5000
}

results = run_jsma_and_eval_jpeg_defense(
    model=model,
    test_set=test_set,
    class_names=class_names,
    mean=mean, std=std,
    n_samples=50,
    jsma_params=jsma_params,
    jpeg_quality=60,
    visualize_first=True,
    viz_func=visualize_attack_quick_fixed  # or your preferred viz wrapper
)



def fgsm_attack_roi_pixelspace(model, image, label, roi_mask, eps=8/255.0):
    """
    FGSM ROI 공격 (ROI 영역에만 섭동 추가, 픽셀 공간 기준)
    Args:
        model: torch.nn.Module
        image: (1,C,H,W) tensor, [0,1] 범위
        label: (1,) tensor
        roi_mask: (H,W) tensor {0,1}
        eps: 0~1 scale (픽셀 단위 섭동 크기)
    """
    device = next(model.parameters()).device
    image = image.clone().detach().to(device).requires_grad_(True)
    label = label.to(device)

    # 모델 입력은 정규화된 이미지
    output = model(to_norm(image))
    loss = nn.CrossEntropyLoss()(output, label)

    model.zero_grad()
    loss.backward()

    # gradient sign
    grad_sign = image.grad.data.sign()

    # ROI 마스크 적용
    roi_mask = roi_mask.to(device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    roi_mask = roi_mask.expand_as(grad_sign)                  # (1,C,H,W)
    grad_sign = grad_sign * roi_mask

    # adversarial example 생성
    adv_image = image + eps * grad_sign
    adv_image = torch.clamp(adv_image, 0, 1)

    perturb = adv_image - image
    return adv_image.detach(), perturb.detach()

def visualize_fgsm_roi(model, x, x_adv, pert, roi_mask, class_names):
    import matplotlib.pyplot as plt
    import numpy as np

    # numpy 변환
    orig = x.squeeze().permute(1,2,0).cpu().numpy()
    adv  = x_adv.squeeze().permute(1,2,0).cpu().numpy()
    pert_vis = pert.squeeze().permute(1,2,0).cpu().numpy()
    roi_vis = roi_mask.cpu().numpy()

    orig = np.clip(orig, 0, 1)
    adv  = np.clip(adv, 0, 1)

    # 섭동 강조 (시각화용)
    amp = 10.0
    pert_vis = (pert_vis * amp + 1) / 2
    pert_vis = np.clip(pert_vis, 0, 1)

    # 모델 예측
    with torch.no_grad():
        pred_orig = model(to_norm(x)).argmax(1).item()
        pred_adv  = model(to_norm(x_adv)).argmax(1).item()

    plt.figure(figsize=(12,3))
    plt.subplot(1,4,1)
    plt.title(f"Original\n{class_names[pred_orig]}")
    plt.imshow(orig)
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.title("Perturbation (x10)")
    plt.imshow(pert_vis)
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.title("Adversarial\n"+class_names[pred_adv])
    plt.imshow(adv)
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.title("ROI Mask")
    plt.imshow(roi_vis, cmap="gray")
    plt.axis("off")

    plt.show()

def run_fgsm_roi_attack(model, test_set, class_names, eps=8/255.0, n_samples=100):
    """
    테스트셋 앞 n_samples 장에 대해 FGSM ROI 공격 실행
    + ΔMargin / ΔConfidence 통계 출력
    """
    device = next(model.parameters()).device
    success_list, l2_list, changed_list, total_list, elapsed_list = [], [], [], [], []
    dmargin_list, conf_drop_list = [], []  # ΔMargin, ΔConfidence 기록

    def _margin_from_logits(logits, y_idx: int):
        # margin = f_y - max_{j != y} f_j
        correct = logits[0, y_idx]
        mask = torch.ones_like(logits[0], dtype=torch.bool)
        mask[y_idx] = False
        max_other = logits[0][mask].max()
        return (correct - max_other).item()

    for i in range(min(n_samples, len(test_set))):
        img, label, roi_mask = test_set[i]

        x = img.unsqueeze(0).to(device)       # (1,3,H,W)
        y = torch.tensor([label], device=device)
        y_idx = int(label)
        roi = torch.as_tensor(roi_mask, dtype=torch.float32)  # (H,W)

        # 공격 실행
        start = time.time()
        x_adv, pert = fgsm_attack_roi_pixelspace(model, x, y, roi, eps=eps)
        elapsed = time.time() - start

        # 성능 측정
        with torch.no_grad():
            logits_clean = model(to_norm(x))
            logits_adv   = model(to_norm(x_adv))
            pred_orig = logits_clean.argmax(1).item()
            pred_adv  = logits_adv.argmax(1).item()

        success = (pred_adv != pred_orig)
        success_list.append(success)
        elapsed_list.append(elapsed)

        # L2 & 변경 픽셀 계산 (분모는 ROI 픽셀 수)
        delta = (x_adv - x).cpu()
        l2_255 = (delta.view(1,-1)*255).norm(p=2).item()
        changed = ((delta.abs()*255) > 1.0).sum().item()
        total = int(roi.numel())
        l2_list.append(l2_255)
        changed_list.append(changed)
        total_list.append(total)

        # ----- ΔMargin / ΔConfidence -----
        margin_clean = _margin_from_logits(logits_clean, y_idx)
        margin_adv   = _margin_from_logits(logits_adv,   y_idx)
        dmargin_list.append(float(margin_clean - margin_adv))

        probs_clean = torch.softmax(logits_clean, dim=1)[0]
        probs_adv   = torch.softmax(logits_adv,   dim=1)[0]
        conf_drop_list.append(float(probs_clean[y_idx] - probs_adv[y_idx]))
        # ---------------------------------

        # 한 장마다 결과 출력
        print(f"[{i+1}/{n_samples}] 공격 {'성공 ✅' if success else '실패 ❌'} "
              f"(원래: {class_names[pred_orig]} → 적대: {class_names[pred_adv]})")

        # 첫 장만 시각화
        if i == 0:
            visualize_fgsm_roi(model, x, x_adv, pert, roi, class_names)

    # 안전 평균 함수
    def _safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else float('nan')

    # 전체 평균 통계 (기존 반환용 4개)
    ASR = float(np.mean(success_list) * 100.0)
    mean_L2_255 = float(np.mean(l2_list))
    mean_changed_pct = float(np.mean(np.array(changed_list) / np.array(total_list)) * 100.0)
    mean_time = float(np.mean(elapsed_list))

    # 성공/전체 Δ 지표 평균
    success_idx = [i for i, s in enumerate(success_list) if s]
    mean_dmargin_success   = _safe_mean([dmargin_list[i]    for i in success_idx])
    mean_confdrop_success  = _safe_mean([conf_drop_list[i] for i in success_idx])
    mean_dmargin_all       = _safe_mean(dmargin_list)
    mean_confdrop_all      = _safe_mean(conf_drop_list)

    print("\n===== FGSM ROI Attack (Batch 100) 평균 통계 =====")
    print(f"Attack Success Rate (ASR)            : {ASR:.2f}%")
    print(f"Mean L2 (0~255 scale)                : {mean_L2_255:.3f}")
    print(f"Mean Changed Pixels (elem)           : {mean_changed_pct:.2f}%")
    print(f"Mean Generation Time (sec)           : {mean_time:.2f}")

    print("\n--- 성공 케이스 평균 ---")
    print(f"Mean ΔMargin (confidence gap drop)   : {mean_dmargin_success:.3f}")
    print(f"Mean ΔConfidence (softmax)           : {mean_confdrop_success:.3f}")

    print("\n--- 전체 평균 ---")
    print(f"Mean ΔMargin (confidence gap drop)   : {mean_dmargin_all:.3f}")
    print(f"Mean ΔConfidence (softmax)           : {mean_confdrop_all:.3f}")

    return ASR, mean_L2_255, mean_changed_pct, mean_time, success_list

# 테스트셋 (ROI 포함된 버전 사용)
test_set = BrainTumorDatasetWithROI(
    "/content/drive/MyDrive/Adversarial_AI/processed_data_np224/Testing",
    transform=transforms.ToTensor()
)

# 100장 ROI FGSM 공격 실행
ASR, mean_L2, mean_changed_pct, mean_time, success_list = run_fgsm_roi_attack(
    model, test_set, class_names, eps=8/255.0, n_samples=100
)