"""
Square Attack L2 (ROI 버전)
- ROI(Region of Interest) 영역 내에서만 L2 제약 공격 수행
- Margin 기반 공격
"""

import sys
from pathlib import Path
import time
import math
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import json
import os


# ==============================================================================
# 데이터셋 클래스
# ==============================================================================

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


# ==============================================================================
# 유틸리티 함수
# ==============================================================================

def to_norm(x):
    """모델이 내부에서 정규화하는 경우 사용"""
    return x


@torch.no_grad()
def predict_logits(model, x_pix):
    """로짓 예측"""
    model.eval()
    return model(to_norm(x_pix))


def margin_loss_untargeted(logits, y):
    """비표적 공격용 마진 손실"""
    bs = logits.size(0)
    correct_logit = logits[torch.arange(bs, device=logits.device), y]
    tmp = logits.clone()
    tmp[torch.arange(bs, device=logits.device), y] = -1e9
    max_other = tmp.max(dim=1).values
    return correct_logit - max_other   # <=0 success


def margin_loss_targeted(logits, y_tgt):
    """표적 공격용 마진 손실"""
    bs = logits.size(0)
    target_logit = logits[torch.arange(bs, device=logits.device), y_tgt]
    tmp = logits.clone()
    tmp[torch.arange(bs, device=logits.device), y_tgt] = -1e9
    max_other = tmp.max(dim=1).values
    return max_other - target_logit    # <=0 success


def ce_loss_targeted(logits, y_tgt):
    """표적 공격용 Cross Entropy 손실"""
    return nn.CrossEntropyLoss(reduction='none')(logits, y_tgt)


def p_selection(p_init, it, n_iters):
    """동적 p 선택 (Square Attack 스케줄)"""
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
    """의사 가우시안 섭동 (직사각형)"""
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
    """의사 가우시안 섭동 (정사각형)"""
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
    """마스크 내부만 L2 노름 계산"""
    # t, m: (1,C,H,W). m in {0,1}. 마스크 내부만 계산
    v = (t * m).view(1, -1)
    return v.norm(p=2, dim=1, keepdim=True).view(1,1,1,1).clamp(min=1e-12)


def _l2_norm_over_mask_region(t, m, region_mask):
    """특정 영역과 마스크 교집합에서 L2 노름 계산"""
    # region_mask: (1,C,H,W) with {0,1} selecting region; use intersection with m
    mm = torch.maximum((m>0).float(), torch.zeros_like(m))
    r = torch.maximum((region_mask>0).float(), torch.zeros_like(region_mask))
    v = (t * mm * r).view(1, -1)
    return v.norm(p=2, dim=1, keepdim=True).view(1,1,1,1).clamp(min=1e-12)


def _apply_mask(delta, m):
    """마스크 적용 (ROI 밖은 0)"""
    delta.mul_(m)
    return delta


# ==============================================================================
# Square Attack L2 (ROI 버전)
# ==============================================================================

def square_attack_l2_single_roi(
    model, x_pix, y_true, mask_2d, y_target=None,
    eps=0.5, n_iters=2000, p_init=0.1,
    targeted=False, use_ce_for_targeted=True, seed=0
):
    """
    Square Attack L2 (ROI 버전)
    
    Args:
        model: 타겟 모델
        x_pix: (1,3,H,W), [0,1]
        y_true: 실제 레이블
        mask_2d: (H,W) torch/numpy, {0,1}/bool → ROI만 공격
        y_target: 표적 공격 시 타겟 레이블
        eps: L2 제약 (0~1 스케일)
        n_iters: 최대 반복 횟수
        p_init: 초기 p 값
        targeted: 표적 공격 여부
        use_ce_for_targeted: 표적 공격 시 CE 손실 사용 여부
        seed: 랜덤 시드
        
    Returns:
        x_best: 공격된 이미지
        n_queries: 쿼리 수
        elapsed: 소요 시간
        final_margin: 최종 마진 값
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

    logits = predict_logits(model, x_best)
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
        scale = ((torch.clamp(eps**2 - curr_norm_img**2, min=0.0) / c) + norms_windows**2).sqrt()

        # 새 패치 정규화 후 스케일
        nd_norm = _l2_norm_over_mask(new_deltas, torch.ones_like(new_deltas))
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

        logits_new = predict_logits(model, x_new)
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


# ==============================================================================
# 통계 및 시각화
# ==============================================================================

@torch.no_grad()
def stats_table_like(model, x_pix_clean, x_adv_pix, elapsed_sec, y_true, count_mode='element', tau_255=1.0):
    """공격 결과 통계 계산"""
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
    logits_clean = predict_logits(model, x_pix_clean)
    logits_adv   = predict_logits(model, x_adv_pix)

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


def _tensor_img_to_numpy(x):
    """텐서 이미지를 numpy로 변환"""
    if x.dim() == 4: x = x.squeeze(0)
    return x.detach().clamp(0,1).permute(1,2,0).cpu().numpy()


def _pred_and_conf(model, x):
    """예측과 신뢰도 반환"""
    logits = predict_logits(model, x)
    probs = torch.softmax(logits, dim=1)
    pred  = probs.argmax(1).item()
    conf  = probs[0, pred].item()
    return pred, conf


def visualize_attack_quick(
    model, x_one, x_adv, class_names=None, mask_2d=None,
    amp_heat=3.0, out_dir=".", filename_prefix="attack_vis", display=True
):
    """
    공격 결과 시각화
    
    Args:
        model: 타겟 모델
        x_one: (1,3,H,W) 원본
        x_adv: (1,3,H,W) 공격된 이미지
        class_names: 클래스 이름 리스트
        mask_2d: ROI 마스크
        amp_heat: 섭동 히트맵 증폭 계수
        out_dir: 저장 디렉토리
        filename_prefix: 파일명 접두사
        display: 화면 표시 여부
    """
    os.makedirs(out_dir, exist_ok=True)
    device = x_one.device
    print("[VIS] visualize_attack_quick start")

    # 예측/신뢰도
    with torch.no_grad():
        pred_b, conf_b = _pred_and_conf(model, x_one)
        pred_a, conf_a = _pred_and_conf(model, x_adv)

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
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"[VIS] Saved to: {out_path}")

    if display:
        plt.show()
    else:
        plt.close(fig)

    return out_path


# ==============================================================================
# 배치 실행 함수
# ==============================================================================

def run_square_attack_batch(
    model, test_set, device='cuda',
    eps=2.0, n_iters=10000, p_init=0.1,
    use_roi=False, n_samples=100,
    out_dir="./attack_vis", visualize=True
):
    """
    배치로 Square Attack 실행
    
    Args:
        model: 타겟 모델
        test_set: 테스트 데이터셋
        device: 디바이스
        eps: L2 제약
        n_iters: 최대 반복 횟수
        p_init: 초기 p 값
        use_roi: ROI 사용 여부
        n_samples: 샘플 수
        out_dir: 시각화 저장 디렉토리
        visualize: 시각화 여부
        
    Returns:
        results: 결과 딕셔너리
    """
    # 데이터 로드
    test_loader = DataLoader(test_set, batch_size=n_samples, shuffle=False)
    x_pix_clean, y_true, masks_2d = next(iter(test_loader))
    x_pix_clean = x_pix_clean.to(device)
    y_true = y_true.to(device)
    B = x_pix_clean.size(0)
    
    class_names = test_set.class_names

    def _full_ones_mask_like(x1):
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
        xi = x_pix_clean[i:i+1]
        mi = masks_2d[i]

        if use_roi:
            mi_eff = mi
            mask_for_vis = mi
        else:
            mi_eff = _full_ones_mask_like(xi)
            mask_for_vis = None

        with torch.no_grad():
            src_idx = predict_logits(model, xi).argmax(1)

        # 공격 실행
        x_adv_pix, n_queries, elapsed, final_margin = square_attack_l2_single_roi(
            model, xi, y_true=src_idx, mask_2d=mi_eff,
            eps=eps, n_iters=n_iters, p_init=p_init,
            targeted=False, use_ce_for_targeted=True, seed=0
        )

        # 통계 계산
        S = stats_table_like(model, xi, x_adv_pix, elapsed, y_true=int(src_idx.item()), 
                           count_mode='element', tau_255=1.0)

        with torch.no_grad():
            y_idx = int(src_idx.item())
            clean_probs = torch.softmax(predict_logits(model, xi), dim=1)
            adv_probs   = torch.softmax(predict_logits(model, x_adv_pix), dim=1)
            clean_conf  = clean_probs[0, y_idx].item()
            adv_conf    = adv_probs[0, y_idx].item()
        conf_drop = clean_conf - adv_conf
        conf_drop_list.append(conf_drop)

        # 성공 여부
        pred_orig = predict_logits(model, xi).argmax(1).item()
        pred_adv  = predict_logits(model, x_adv_pix).argmax(1).item()
        success   = (pred_orig != pred_adv)
        print(f"[{i+1}/{B}] 공격 {'성공 ✅' if success else '실패 ❌'}")
        
        if visualize:
            img_path = visualize_attack_quick(
                model, xi, x_adv_pix,
                class_names=class_names, mask_2d=mask_for_vis,
                amp_heat=4.0, out_dir=out_dir, 
                filename_prefix=f"square_sample{i}", display=False
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

    # 결과 집계
    def _safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else float('nan')

    ASR = float(np.mean(success_list) * 100.0)
    success_idx = [i for i, s in enumerate(success_list) if s]

    # 성공 케이스 평균
    mean_L2_success = _safe_mean([l2_list[i] for i in success_idx])
    mean_changed_success = _safe_mean([changed_list[i] / total_list[i] for i in success_idx]) * 100.0
    mean_time_success = _safe_mean([elapsed_list[i] for i in success_idx])
    mean_dmargin_success = _safe_mean([dmargin_list[i] for i in success_idx])
    mean_conf_drop_success = _safe_mean([conf_drop_list[i] for i in success_idx])

    # 전체 평균
    mean_L2_all = _safe_mean(l2_list)
    mean_changed_all = _safe_mean(np.array(changed_list) / np.array(total_list)) * 100.0
    mean_time_all = _safe_mean(elapsed_list)
    mean_dmargin_all = _safe_mean(dmargin_list)
    mean_conf_drop_all = _safe_mean(conf_drop_list)

    print("\n===== Square L2 Attack 평균 통계 =====")
    print(f"Attack Success Rate (ASR)                : {ASR:.2f}%")

    print("\n--- 성공 케이스 평균 ---")
    print(f"Mean L2 (0~255 scale)                    : {mean_L2_success:.3f}")
    print(f"Mean Changed Pixels (elem)               : {mean_changed_success:.2f}%")
    print(f"Mean Generation Time (sec)               : {mean_time_success:.2f}")
    print(f"Mean ΔMargin (confidence drop)           : {mean_dmargin_success:.3f}")
    print(f"Mean ΔConfidence (softmax)               : {mean_conf_drop_success:.3f}")

    print("\n--- 전체 평균 ---")
    print(f"Mean L2 (0~255 scale)                    : {mean_L2_all:.3f}")
    print(f"Mean Changed Pixels (elem)               : {mean_changed_all:.2f}%")
    print(f"Mean Generation Time (sec)               : {mean_time_all:.2f}")
    print(f"Mean ΔMargin (confidence drop)           : {mean_dmargin_all:.3f}")
    print(f"Mean ΔConfidence (softmax)               : {mean_conf_drop_all:.3f}")

    return {
        'ASR': ASR,
        'mean_L2_success': mean_L2_success,
        'mean_changed_success': mean_changed_success,
        'mean_time_success': mean_time_success,
        'mean_dmargin_success': mean_dmargin_success,
        'mean_conf_drop_success': mean_conf_drop_success,
        'mean_L2_all': mean_L2_all,
        'mean_changed_all': mean_changed_all,
        'mean_time_all': mean_time_all,
        'mean_dmargin_all': mean_dmargin_all,
        'mean_conf_drop_all': mean_conf_drop_all,
        'success_list': success_list,
        'x_adv_list': x_adv_list
    }


# ==============================================================================
# 메인 실행 예제
# ==============================================================================

if __name__ == "__main__":
    # 사용 예제
    print("Square Attack L2 (ROI) 모듈")
    print("사용법:")
    print("  from attacks.attack_square_l2 import run_square_attack_batch, BrainTumorDatasetWithROI")
    print("  results = run_square_attack_batch(model, test_set, device='cuda', eps=2.0)")

