"""
JSMA (Jacobian-based Saliency Map Attack) 구현
- 마진 기반 공격 (margin = logit[y] - max_{j≠y} logit[j])
- Adam 스타일 모멘텀 적용
- 동적 k 조절 (정체 시 자동 확대)
- ROI(Region of Interest) 마스크 지원
"""

# 전역 변수
attack_utils_module = None

import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
from torchvision import transforms

# 공통 유틸리티 함수는 setup_environment() 후에 임포트하도록 변경


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
# 유틸리티 함수 (attack_utils에서 import)
# ==============================================================================
# to_pixel, to_norm: attack_utils에서 제공


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

def run_jsma_roi_attack(
    model,
    test_set,
    class_names,
    mean=None, std=None,           # jsma_attack 인터페이스 요구 (기본값: attack_utils 사용)
    theta=0.08,
    max_pixels_pct=0.05,
    k_small=2,
    restarts=4,
    topk_pool=5000,
    n_samples=100,
    save_results=True,
    create_visualizations=True,
    out_dir="./jsma_roi_results"
):
    """
    ROI JSMA 공격을 테스트셋 앞 n_samples 장에 대해 실행 (통일된 인터페이스)
    
    Args:
        model: 타겟 모델
        test_set: 테스트 데이터셋 (ROI 마스크 포함)
        class_names: 클래스 이름 리스트
        mean: 정규화 평균 (None이면 attack_utils 기본값 사용)
        std: 정규화 표준편차 (None이면 attack_utils 기본값 사용)
        theta: 스텝 크기
        max_pixels_pct: 최대 변경 픽셀 비율
        k_small: 초기 동시 변경 픽셀 수
        restarts: 재시도 횟수
        topk_pool: 후보 풀 크기
        n_samples: 공격할 샘플 수
        save_results: True면 결과를 파일로 저장
        create_visualizations: True면 시각화 차트 생성
        out_dir: 결과 저장 디렉토리
    
    Returns:
        results: 결과 딕셔너리 리스트
        statistics: 통계 딕셔너리
    """
    global attack_utils_module
    
    device = next(model.parameters()).device
    results = []
    
    # 기본값 설정 (attack_utils의 정규화 사용)
    if mean is None:
        mean = [0.0, 0.0, 0.0]  # 모델이 내부 정규화하는 경우
    if std is None:
        std = [1.0, 1.0, 1.0]
    
    # 랜덤 샘플링으로 편향 방지 (Zoo와 동일한 방식)
    n_total = len(test_set)
    n_samples = min(n_samples, n_total)
    
    # 시드 고정하여 재현 가능한 랜덤 샘플링
    np.random.seed(42)
    sampled_indices = np.random.choice(n_total, size=n_samples, replace=False)
    
    print(f"\n===== JSMA ROI Attack (theta={theta}, max_pixels={max_pixels_pct*100:.1f}%) =====")
    print(f"공격 대상: {n_samples}개 샘플 (ROI 영역만, 랜덤 샘플링)\n")
    
    for idx, i in enumerate(sampled_indices):
        sample = test_set[i]
        
        # dataset 형식 (img, label) 또는 (img, label, mask)
        if len(sample) == 2:
            img, label = sample
            roi_mask = None
        else:
            img, label, roi_mask = sample
        
        x_original = img.unsqueeze(0).to(device)  # (1,C,H,W)
        y = int(label)
        
        # ROI 마스크를 allowed_masks로 변환
        if roi_mask is not None:
            roi_mask_tensor = torch.as_tensor(roi_mask, dtype=torch.bool).to(device)
        else:
            roi_mask_tensor = None
        
        # JSMA ROI 공격 실행
        start_time = time.time()
        x_adv, changed_spatial, l1_total, success_flag = jsma_attack_margin_mom(
            model=model,
            x=x_original,
            y_true=y,
            theta=theta,
            max_pixels_percentage=max_pixels_pct,
            k_small=k_small,
            restarts=restarts,
            topk_pool=topk_pool,
            allowed_masks=roi_mask_tensor  # ROI 마스크 사용
        )
        elapsed_time = time.time() - start_time
        
        # 메트릭 계산 (통일된 함수 사용)
        if attack_utils_module is not None:
            metrics = attack_utils_module.calculate_attack_metrics(x_original, x_adv, label, model, threshold=1.0)
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 기본 메트릭만 계산합니다.")
            # 기본 메트릭 계산
            with torch.no_grad():
                pred_orig = model(x_original).argmax(1).item()
                pred_adv = model(x_adv).argmax(1).item()
            
            success = (pred_orig != pred_adv)
            l2_norm = ((x_adv - x_original) * 255.0).view(1, -1).norm(p=2).item()
            
            metrics = {
                'success': success,
                'pred_original': pred_orig,
                'pred_adv': pred_adv,
                'l2_norm': l2_norm,
                'conf_drop': 0.0  # 기본값
            }
        
        metrics['elapsed_time'] = elapsed_time
        metrics['theta'] = theta
        metrics['max_pixels_pct'] = max_pixels_pct
        metrics['sample_idx'] = i
        
        # 결과 저장
        results.append(metrics)
        
        # 진행 상황 출력 (통일된 형식)
        pred_orig = metrics['pred_original']
        pred_adv = metrics['pred_adv']
        success_mark = '✅' if metrics['success'] else '❌'
        print(f"[{i+1:3d}/{n_samples}] {success_mark} {class_names[pred_orig]} → {class_names[pred_adv]} "
              f"| L2={metrics['l2_norm']:.2f} | Δconf={metrics['conf_drop']:.3f} | {elapsed_time:.3f}s")
        
        # 첫 장 시각화 (통일된 함수 사용)
        if i == 0:
            if attack_utils_module is not None:
                attack_utils_module.visualize_attack_result(
                    model, x_original, x_adv,
                    class_names=class_names,
                    roi_mask=roi_mask,
                    amp_heat=4.0,
                    out_dir=out_dir,
                    filename_prefix="jsma_sample0",
                    display=False,
                    save_file=save_results
                )
            else:
                print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 시각화를 건너뜁니다.")
    
    # 통계 계산 (통일된 함수 사용)
    if attack_utils_module is not None:
        statistics = attack_utils_module.calculate_batch_statistics(results, attack_name="JSMA")
        attack_utils_module.print_statistics(statistics)
    else:
        print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 기본 통계만 출력합니다.")
        ASR = float(np.mean([r['success'] for r in results]) * 100.0)
        mean_l2 = float(np.mean([r['l2_norm'] for r in results]))
        mean_time = float(np.mean([r['elapsed_time'] for r in results]))
        print(f"\n[JSMA Attack Results]")
        print(f"ASR: {ASR:.2f}%")
        print(f"Mean L2: {mean_l2:.2f}")
        print(f"Mean Time: {mean_time:.3f}s")
        statistics = {'ASR': ASR, 'mean_l2': mean_l2, 'mean_time': mean_time}
    
    # 결과 저장
    if save_results:
        os.makedirs(out_dir, exist_ok=True)
        
        # CSV 저장
        if attack_utils_module is not None:
            attack_utils_module.save_results_to_csv(results, os.path.join(out_dir, "jsma_results.csv"))
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. CSV 저장을 건너뜁니다.")
        
        # JSON 저장
        if attack_utils_module is not None:
            attack_utils_module.save_results_to_json(statistics, os.path.join(out_dir, "jsma_statistics.json"))
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. JSON 저장을 건너뜁니다.")
    
    # 시각화 차트 생성
    if create_visualizations:
        if attack_utils_module is not None:
            attack_utils_module.create_result_visualization(results, out_dir=out_dir, attack_name="JSMA")
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 시각화 차트 생성을 건너뜁니다.")
    
    return results, statistics


def run_jsma_full_attack(
    model,
    test_set,
    class_names,
    mean=None, std=None,           # jsma_attack 인터페이스 요구 (기본값: attack_utils 사용)
    theta=0.08,
    max_pixels_pct=0.05,
    k_small=2,
    restarts=4,
    topk_pool=5000,
    n_samples=100,
    save_results=True,
    create_visualizations=True,
    out_dir="./jsma_results"
):
    """
    Full-image JSMA 공격을 테스트셋 앞 n_samples 장에 대해 실행 (통일된 인터페이스)
    
    Args:
        model: 타겟 모델
        test_set: 테스트 데이터셋
        class_names: 클래스 이름 리스트
        mean: 정규화 평균 (None이면 attack_utils 기본값 사용)
        std: 정규화 표준편차 (None이면 attack_utils 기본값 사용)
        theta: 스텝 크기
        max_pixels_pct: 최대 변경 픽셀 비율
        k_small: 초기 동시 변경 픽셀 수
        restarts: 재시도 횟수
        topk_pool: 후보 풀 크기
        n_samples: 공격할 샘플 수
        save_results: True면 결과를 파일로 저장
        create_visualizations: True면 시각화 차트 생성
        out_dir: 결과 저장 디렉토리
    
    Returns:
        results: 결과 딕셔너리 리스트
        statistics: 통계 딕셔너리
    """
    global attack_utils_module
    
    device = next(model.parameters()).device
    results = []
    
    # 기본값 설정 (attack_utils의 정규화 사용)
    if mean is None:
        mean = [0.0, 0.0, 0.0]  # 모델이 내부 정규화하는 경우
    if std is None:
        std = [1.0, 1.0, 1.0]
    
    # 랜덤 샘플링으로 편향 방지 (Zoo와 동일한 방식)
    n_total = len(test_set)
    n_samples = min(n_samples, n_total)
    
    # 시드 고정하여 재현 가능한 랜덤 샘플링
    np.random.seed(42)
    sampled_indices = np.random.choice(n_total, size=n_samples, replace=False)
    
    print(f"\n===== JSMA Full Image Attack (theta={theta}, max_pixels={max_pixels_pct*100:.1f}%) =====")
    print(f"공격 대상: {n_samples}개 샘플 (전체 이미지, 랜덤 샘플링)\n")
    
    for idx, i in enumerate(sampled_indices):
        sample = test_set[i]
        
        # dataset 형식 (img, label) 또는 (img, label, mask)
        if len(sample) == 2:
            img, label = sample
            roi_mask = None
        else:
            img, label, roi_mask = sample
        
        x_original = img.unsqueeze(0).to(device)  # (1,C,H,W)
        y = int(label)
        
        # JSMA 전체 이미지 공격 실행 (ROI 무시)
        start_time = time.time()
        x_adv, changed_spatial, l1_total, success_flag = jsma_attack_margin_mom(
            model=model,
            x=x_original,
            y_true=y,
            theta=theta,
            max_pixels_percentage=max_pixels_pct,
            k_small=k_small,
            restarts=restarts,
            topk_pool=topk_pool,
            allowed_masks=None  # ROI 무시, 전체 이미지 공격
        )
        elapsed_time = time.time() - start_time
        
        # 메트릭 계산 (통일된 함수 사용)
        if attack_utils_module is not None:
            metrics = attack_utils_module.calculate_attack_metrics(x_original, x_adv, label, model, threshold=1.0)
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 기본 메트릭만 계산합니다.")
            # 기본 메트릭 계산
            with torch.no_grad():
                pred_orig = model(x_original).argmax(1).item()
                pred_adv = model(x_adv).argmax(1).item()
            
            success = (pred_orig != pred_adv)
            l2_norm = ((x_adv - x_original) * 255.0).view(1, -1).norm(p=2).item()
            
            metrics = {
                'success': success,
                'pred_original': pred_orig,
                'pred_adv': pred_adv,
                'l2_norm': l2_norm,
                'conf_drop': 0.0  # 기본값
            }
        
        metrics['elapsed_time'] = elapsed_time
        metrics['theta'] = theta
        metrics['max_pixels_pct'] = max_pixels_pct
        metrics['sample_idx'] = idx
        metrics['use_roi'] = False  # 전체 이미지 공격
        
        results.append(metrics)
        
        # 진행 상황 출력 (통일된 형식)
        pred_orig = metrics['pred_original']
        pred_adv = metrics['pred_adv']
        success_mark = '✅' if metrics['success'] else '❌'
        print(f"[{idx+1:3d}/{n_samples}] {success_mark} {class_names[pred_orig]} → {class_names[pred_adv]} "
              f"| L2={metrics['l2_norm']:.2f} | Δconf={metrics['conf_drop']:.3f} | {elapsed_time:.3f}s")
        
        # 첫 장 시각화 (통일된 함수 사용)
        if idx == 0:
            if attack_utils_module is not None:
                attack_utils_module.visualize_attack_result(
                    model, x_original, x_adv,
                    class_names=class_names,
                    roi_mask=roi_mask,
                    amp_heat=4.0,
                    out_dir=out_dir,
                    filename_prefix="jsma_full_sample0",
                    display=False,
                    save_file=save_results
                )
            else:
                print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 시각화를 건너뜁니다.")
    
    # 통계 계산 (통일된 함수 사용)
    if attack_utils_module is not None:
        statistics = attack_utils_module.calculate_batch_statistics(results, attack_name="JSMA_Full")
        attack_utils_module.print_statistics(statistics)
    else:
        print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 기본 통계만 출력합니다.")
        ASR = float(np.mean([r['success'] for r in results]) * 100.0)
        mean_l2 = float(np.mean([r['l2_norm'] for r in results]))
        mean_time = float(np.mean([r['elapsed_time'] for r in results]))
        print(f"\n[JSMA Full Attack Results]")
        print(f"ASR: {ASR:.2f}%")
        print(f"Mean L2: {mean_l2:.2f}")
        print(f"Mean Time: {mean_time:.3f}s")
        statistics = {'ASR': ASR, 'mean_l2': mean_l2, 'mean_time': mean_time}
    
    # 결과 저장
    if save_results:
        import os
        os.makedirs(out_dir, exist_ok=True)
        
        # CSV 저장
        if attack_utils_module is not None:
            attack_utils_module.save_results_to_csv(results, os.path.join(out_dir, "jsma_full_results.csv"))
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. CSV 저장을 건너뜁니다.")
        
        # JSON 저장
        if attack_utils_module is not None:
            attack_utils_module.save_results_to_json(statistics, os.path.join(out_dir, "jsma_full_statistics.json"))
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. JSON 저장을 건너뜁니다.")
    
    # 시각화 차트 생성
    if create_visualizations:
        if attack_utils_module is not None:
            attack_utils_module.create_result_visualization(results, out_dir=out_dir, attack_name="JSMA_Full")
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 시각화 차트 생성을 건너뜁니다.")
    
    return results, statistics


# ==============================================================================
# 시각화 함수
# ==============================================================================

def _tensor_img_to_numpy(x):
    """(1,3,H,W) -> (H,W,3)"""
    if x.dim() == 4:
        x = x.squeeze(0)
    return x.detach().clamp(0,1).permute(1,2,0).cpu().numpy()


def _pred_and_conf_from_logits(logits):
    """로짓으로부터 예측과 신뢰도 계산"""
    probs = torch.softmax(logits, dim=1)
    pred  = probs.argmax(1).item()
    conf  = probs[0, pred].item()
    return pred, conf


# ==============================================================================
# 환경 감지 및 설정
# ==============================================================================

def setup_environment():
    """Colab 및 로컬 환경 자동 감지 및 경로 설정"""
    import sys
    import os
    from pathlib import Path
    
    # Colab 환경 감지
    try:
        import google.colab
        IN_COLAB = True
        print("[INFO] Google Colab 환경 감지")
        
        # Colab 환경에서만 Google Drive를 마운트합니다.
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Colab 환경에서 작업 디렉토리 이동
        work_dir = '/content/drive/MyDrive/Adversarial_medical_AI'
        os.chdir(work_dir)
        print(f"[INFO] 작업 디렉토리 변경: {os.getcwd()}")
        
        # resnet50_model.py 및 attack_utils.py 임포트를 위해 sys.path에 작업 디렉토리 추가
        if work_dir not in sys.path:
            sys.path.insert(0, work_dir)
            print(f"[INFO] {work_dir}를 sys.path에 추가.")
        
        # Colab에서 모듈 임포트 확인
        try:
            import attack_utils
            print("[INFO] attack_utils 모듈 임포트 성공")
        except ImportError as e:
            print(f"[WARNING] attack_utils 모듈 임포트 실패: {e}")
            print(f"[INFO] 현재 작업 디렉토리: {os.getcwd()}")
            print(f"[INFO] attack_utils.py 파일 존재 여부: {os.path.exists(os.path.join(work_dir, 'attack_utils.py'))}")
            # 강제로 현재 디렉토리를 sys.path에 추가
            current_dir = os.getcwd()
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
                print(f"[INFO] 현재 디렉토리 {current_dir}를 sys.path에 추가.")
        
        base_path = Path(work_dir)
        
    except ImportError:
        IN_COLAB = False
        print("[INFO] 로컬 환경에서 실행")
        
        # 로컬 환경: 현재 스크립트 위치 기준
        base_path = Path(__file__).parent.resolve()
        print(f"[INFO] 로컬 작업 경로: {base_path}")
        
        # sys.path에 추가 (resnet50_model.py 및 attack_utils.py 임포트용)
        if str(base_path) not in sys.path:
            sys.path.insert(0, str(base_path))
            print(f"[INFO] {base_path}를 sys.path에 추가했습니다.")
    
    return base_path, IN_COLAB


def seed_everything(seed: int = 42):
    """재현성을 위한 시드 고정"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] 시드 고정 완료: {seed}")


def setup_device(device_config='auto'):
    """
    디바이스 설정 (preprocess2_ResNet50.py와 동일한 로직)
    
    Args:
        device_config: 'auto', 'cuda', 'cpu' 중 선택 가능
    
    Returns:
        device: torch.device 객체
        use_amp: AMP 사용 여부
    """
    import torch
    
    # 디바이스 선택 (우선순위: config -> GPU 가용성)
    if device_config == 'cpu':
        device = torch.device("cpu")
        print("[INFO] 강제로 CPU 사용")
    elif device_config == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("[INFO] 강제로 CUDA 사용")
        else:
            device = torch.device("cpu")
            print("[WARNING] CUDA 사용 불가, CPU로 대체")
    else:  # 'auto' 또는 기본값 : GPU 우선
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] 자동 디바이스 선택: {device}")
    
    # AMP 사용여부 (GPU일 때만 True)
    use_amp = (device.type == "cuda")
    print(f"[INFO] AMP 사용: {use_amp}")
    
    return device, use_amp


def ensure_attack_utils_import():
    """Colab 환경에서 attack_utils 모듈 임포트를 보장하는 함수"""
    global attack_utils_module
    
    try:
        import attack_utils
        attack_utils_module = attack_utils
        print("[INFO] attack_utils 모듈 임포트 성공!")
        
        # 주요 함수들이 실제로 사용 가능한지 확인
        required_functions = ['to_norm', 'to_pixel', 'calculate_attack_metrics', 'visualize_attack_result']
        missing_functions = []
        
        for func_name in required_functions:
            if not hasattr(attack_utils, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"[WARNING] 다음 함수들이 attack_utils 모듈에서 찾을 수 없습니다: {missing_functions}")
            return False
        else:
            print("[INFO] 모든 필수 함수들이 사용 가능합니다.")
            return True
            
    except ImportError:
        print("[INFO] attack_utils 모듈을 다시 시도합니다...")
        
        # 현재 디렉토리 확인
        current_dir = os.getcwd()
        print(f"[INFO] 현재 작업 디렉토리: {current_dir}")
        
        # attack_utils.py 파일 확인
        attack_utils_path = os.path.join(current_dir, 'attack_utils.py')
        if os.path.exists(attack_utils_path):
            print(f"[INFO] attack_utils.py 파일 발견: {attack_utils_path}")
            # 현재 디렉토리를 sys.path에 추가
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
                print(f"[INFO] {current_dir}를 sys.path에 추가했습니다.")
            
            # 다시 임포트 시도
            try:
                import attack_utils
                attack_utils_module = attack_utils
                print("[INFO] attack_utils 모듈 임포트 성공!")
                
                # 주요 함수들이 실제로 사용 가능한지 확인
                required_functions = ['to_norm', 'to_pixel', 'calculate_attack_metrics', 'visualize_attack_result']
                missing_functions = []
                
                for func_name in required_functions:
                    if not hasattr(attack_utils, func_name):
                        missing_functions.append(func_name)
                
                if missing_functions:
                    print(f"[WARNING] 다음 함수들이 attack_utils 모듈에서 찾을 수 없습니다: {missing_functions}")
                    return False
                else:
                    print("[INFO] 모든 필수 함수들이 사용 가능합니다.")
                    return True
                    
            except ImportError as e:
                print(f"[ERROR] 여전히 임포트 실패: {e}")
                return False
        else:
            print(f"[ERROR] attack_utils.py 파일을 찾을 수 없습니다: {attack_utils_path}")
            return False


def setup_logging(log_dir="results"):
    """
    로깅 설정: 콘솔과 파일에 동시 출력
    로그 파일: {log_dir}/attack.log
    """
    import logging
    from pathlib import Path
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "attack.log"
    
    # 기존 로거 제거 (중복 방지)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 로깅 포맷 설정
    log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 로거 설정
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),  # 파일에 저장
            logging.StreamHandler()  # 콘솔에 출력
        ]
    )
    
    logging.info(f"로그 파일 생성: {log_file}")
    return log_file


# ==============================================================================
# 사용 예시 (필요 시 주석 해제)
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("JSMA Attack 실행")
    print("="*70)
    
    # 1. 환경 설정
    base_path, is_colab = setup_environment()
    
    # 2. 시드 고정
    seed_everything(42)
    
    # 3. 로깅 설정
    log_file = setup_logging("jsma_results")
    
    # 4. attack_utils 모듈 임포트 확인 (Colab용)
    if not ensure_attack_utils_import():
        print("[WARNING] attack_utils 모듈을 사용할 수 없습니다. 일부 기능이 제한될 수 있습니다.")
    
    # 5. 디바이스 설정
    device, use_amp = setup_device('auto')  # 'auto', 'cuda', 'cpu' 선택 가능
    
    # 6. 모델 로드
    try:
        from resnet50_model import load_trained_model
        
        print(f"[INFO] 사용 디바이스: {device}")
        
        # 경로 설정 (통일된 경로 사용)
        checkpoint_path = base_path / "models" / "resnet50_binary_best.pth"
        data_path = base_path / "processed_data_np224" / "Testing"
        
        print(f"[INFO] 모델 경로: {checkpoint_path}")
        print(f"[INFO] 데이터 경로: {data_path}")
        
        # 파일 존재 확인
        if not checkpoint_path.exists():
            print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {checkpoint_path}")
            print("[INFO] 모델을 먼저 학습시켜주세요.")
            import sys
            sys.exit(1)
        
        if not data_path.exists():
            print(f"[ERROR] 데이터 경로를 찾을 수 없습니다: {data_path}")
            print("[INFO] 전처리된 데이터를 먼저 준비해주세요.")
            import sys
            sys.exit(1)
        
        # 모델 로드
        model = load_trained_model(str(checkpoint_path), device=device)
        print("[INFO] 모델 로드 완료")
        
        # 5. 테스트셋 로드
        test_set = BrainTumorDatasetWithROI(
            str(data_path),
            transform=transforms.ToTensor()
        )
        class_names = test_set.class_names
        print(f"[INFO] 테스트 데이터 로드 완료: {len(test_set)}개 샘플")
        
        # 6. 공격 파라미터 설정
        theta = 0.08
        max_pixels_pct = 0.05
        n_samples = 100
        
        print("\n" + "="*70)
        print("JSMA Attack 실행")
        print("="*70)
        
        # 7. JSMA 전체 이미지 공격 실행
        results_full, statistics_full = run_jsma_full_attack(
            model=model,
            test_set=test_set,
            class_names=class_names,
            theta=theta,
            max_pixels_pct=max_pixels_pct,
            n_samples=n_samples,
            save_results=True,
            create_visualizations=True,
            out_dir="./jsma_results"
        )
        
        # 통계에서 필요한 값들 추출
        ASR_full = statistics_full.get('ASR', 0.0)
        mean_L2_full = statistics_full.get('mean_l2', 0.0)
        mean_changed_pct_full = statistics_full.get('mean_changed_ratio', 0.0)
        mean_time_full = statistics_full.get('mean_time', 0.0)
        success_list_full = [r['success'] for r in results_full]
        
        # 8. JSMA ROI 공격 실행
        results_roi, statistics_roi = run_jsma_roi_attack(
            model=model,
            test_set=test_set,
            class_names=class_names,
            theta=theta,
            max_pixels_pct=max_pixels_pct,
            n_samples=n_samples,
            save_results=True,
            create_visualizations=True,
            out_dir="./jsma_roi_results"
        )
        
        # 통계에서 필요한 값들 추출
        ASR_roi = statistics_roi.get('ASR', 0.0)
        mean_L2_roi = statistics_roi.get('mean_l2', 0.0)
        mean_changed_pct_roi = statistics_roi.get('mean_changed_ratio', 0.0)
        mean_time_roi = statistics_roi.get('mean_time', 0.0)
        success_list_roi = [r['success'] for r in results_roi]
        
        # 9. 최종 결과 요약
        print("\n" + "="*70)
        print("JSMA Attack 최종 결과 요약")
        print("="*70)
        print(f"전체 이미지 공격:")
        print(f"  ASR: {ASR_full:.2f}% ({sum(success_list_full)}/{len(success_list_full)})")
        print(f"  평균 L2 거리: {mean_L2_full:.2f}")
        print(f"  평균 변경 픽셀 비율: {mean_changed_pct_full:.2f}%")
        print(f"  평균 소요 시간: {mean_time_full:.3f}s")
        print()
        print(f"ROI 공격:")
        print(f"  ASR: {ASR_roi:.2f}% ({sum(success_list_roi)}/{len(success_list_roi)})")
        print(f"  평균 L2 거리: {mean_L2_roi:.2f}")
        print(f"  평균 변경 픽셀 비율: {mean_changed_pct_roi:.2f}%")
        print(f"  평균 소요 시간: {mean_time_roi:.3f}s")
        print("="*70)
        
    except ImportError as e:
        print(f"[ERROR] 모듈 임포트 실패: {e}")
        print("[INFO] resnet50_model.py 파일이 같은 디렉토리에 있는지 확인하세요.")
    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()



