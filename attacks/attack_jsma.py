"""
JSMA (Jacobian-based Saliency Map Attack) 구현
- 마진 기반 공격 (margin = logit[y] - max_{j≠y} logit[j])
- Adam 스타일 모멘텀 적용
- 동적 k 조절 (정체 시 자동 확대)
- ROI(Region of Interest) 마스크 지원
"""

import time
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset
import json


# ==============================================================================
# 데이터셋 클래스
# ==============================================================================

class BrainTumorDatasetWithROI(Dataset):
    """
    뇌종양 이미지 데이터셋 (+ ROI mask 포함)
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        self.images = np.load(self.data_dir / "images.npy")
        self.labels = np.load(self.data_dir / "labels.npy")
        self.masks  = np.load(self.data_dir / "masks.npy")

        with open(self.data_dir / "class_names.txt", "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f]

        meta_path = self.data_dir / "meta.json"
        self.meta = None
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

        print(f"[INFO] Loaded {len(self.images)} images with ROI from {self.data_dir}")
        print(f"[INFO] Classes: {self.class_names}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = np.repeat(img, 3, axis=2)
        label = int(self.labels[idx])
        mask  = self.masks[idx]

        if self.transform:
            img = self.transform(img)

        mask = torch.from_numpy(mask.astype(np.float32))
        return img, label, mask


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


# ==============================================================================
# 배치 실행 함수
# ==============================================================================

def run_jsma_full_attack(
    model,
    test_set,
    class_names,
    mean, std,
    theta=0.08,
    max_pixels_pct=0.05,
    k_small=2,
    restarts=4,
    topk_pool=5000,
    n_samples=100,
    visualize_first=True,
    viz_func=None
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
        if len(sample) == 2:
            img, label = sample
        else:
            img, label, _ = sample

        x = img.unsqueeze(0).to(device)
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

        # 출력
        pred_orig = metrics['pred_original']
        pred_adv  = metrics['pred_adv']
        print(f"[{i+1}/{n_samples}] 공격 {'성공 ✅' if success else '실패 ❌'} "
              f"(원래: {class_names[pred_orig]} → 적대: {class_names[pred_adv]}) "
              f"L1={metrics['l1_norm']:.1f} changed={metrics['changed_pixels']}/{metrics['total_pixels']} "
              f"ΔConf={metrics['conf_drop']:.3f} ΔMargin={metrics['margin_drop']:.3f} time={elapsed:.2f}s")

        # 첫 장만 시각화
        if i == 0 and visualize_first and viz_func is not None:
            try:
                viz_func(model, x, x_adv, class_names=class_names)
            except Exception as e:
                print("[VIS] visualization failed:", e)

    # 집계
    def _safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else float('nan')

    ASR = float(np.mean(success_list) * 100.0)
    mean_L1 = _safe_mean(l1_list)
    mean_changed_count = _safe_mean(changed_count_list)
    mean_time = _safe_mean(elapsed_list)

    success_idx = [idx for idx, s in enumerate(success_list) if s]
    mean_changed_count_success = _safe_mean([changed_count_list[i] for i in success_idx])
    mean_dmargin_success = _safe_mean([dmargin_list[i] for i in success_idx])
    mean_confdrop_success = _safe_mean([conf_drop_list[i] for i in success_idx])

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


# ==============================================================================
# 시각화 함수
# ==============================================================================

def _tensor_img_to_numpy(x):
    """텐서를 numpy 이미지로 변환"""
    if x.dim() == 4: x = x.squeeze(0)
    return x.detach().clamp(0,1).permute(1,2,0).cpu().numpy()


def _pred_and_conf_from_logits(logits):
    """로짓에서 예측과 신뢰도 추출"""
    probs = torch.softmax(logits, dim=1)
    pred  = probs.argmax(1).item()
    conf  = probs[0, pred].item()
    return pred, conf


def visualize_attack_quick_fixed(
    model,
    x_one,
    x_adv,
    class_names=None,
    mask_2d=None,
    amp_heat=3.0,
    out_dir=".",
    filename_prefix="attack_vis",
    display=True,
    mean=None,
    std=None
):
    """
    JSMA 공격 결과 시각화
    """
    os.makedirs(out_dir, exist_ok=True)
    device = x_one.device
    print("[VIS] visualize_attack_quick_fixed start")

    if mean is None or std is None:
        raise ValueError("visualize_attack_quick_fixed requires mean and std")

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

    with torch.no_grad():
        logits_before = model(_to_norm_local(x_one))
        logits_after  = model(_to_norm_local(x_adv))
    pred_b, conf_b = _pred_and_conf_from_logits(logits_before)
    pred_a, conf_a = _pred_and_conf_from_logits(logits_after)

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
# 메인 실행 예제
# ==============================================================================

if __name__ == "__main__":
    print("JSMA Attack 모듈")
    print("사용법:")
    print("  from attacks.attack_jsma import run_jsma_full_attack, BrainTumorDatasetWithROI")
    print("  ASR, mean_L1, mean_changed_count, mean_time, success_list = run_jsma_full_attack(")
    print("      model, test_set, class_names, mean=[0,0,0], std=[1,1,1], n_samples=100")
    print("  )")

