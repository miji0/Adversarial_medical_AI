"""
공통 적대적 공격 유틸리티 함수 (통합 버전)
- 정규화/역정규화
- 메트릭 계산
- 시각화
- 결과 저장 및 통계
- 데이터 수집 및 체크포인트 관리
- 공격별 특화 함수 (JSMA, ZOO 등)
"""

import os
import random
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ==============================================================================
# 상수 정의
# ==============================================================================

# ImageNet 정규화 상수
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ==============================================================================
# 정규화 함수
# ==============================================================================

def to_pixel(x_norm, mean=None, std=None):
    """
    정규화된 이미지를 픽셀 공간 [0,1]로 변환
    
    Args:
        x_norm: 정규화된 텐서 (B,C,H,W)
        mean: 평균값 (list/tuple/tensor), None이면 IMAGENET_MEAN 사용
        std: 표준편차 (list/tuple/tensor), None이면 IMAGENET_STD 사용
    """
    if mean is None:
        mean = IMAGENET_MEAN.to(x_norm.device)
    elif isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean, device=x_norm.device).view(1, -1, 1, 1)
    
    if std is None:
        std = IMAGENET_STD.to(x_norm.device)
    elif isinstance(std, (list, tuple)):
        std = torch.tensor(std, device=x_norm.device).view(1, -1, 1, 1)
    
    return torch.clamp(x_norm * std + mean, 0.0, 1.0)


def to_norm(x_pix, mean=None, std=None):
    """
    픽셀 공간 [0,1] 이미지를 정규화
    
    Args:
        x_pix: 픽셀 공간 텐서 (B,C,H,W)
        mean: 평균값 (list/tuple/tensor), None이면 IMAGENET_MEAN 사용
        std: 표준편차 (list/tuple/tensor), None이면 IMAGENET_STD 사용
    """
    if mean is None:
        mean = IMAGENET_MEAN.to(x_pix.device)
    elif isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean, device=x_pix.device).view(1, -1, 1, 1)
    
    if std is None:
        std = IMAGENET_STD.to(x_pix.device)
    elif isinstance(std, (list, tuple)):
        std = torch.tensor(std, device=x_pix.device).view(1, -1, 1, 1)
    
    return (torch.clamp(x_pix, 0.0, 1.0) - mean) / std


# ==============================================================================
# 메트릭 계산 함수
# ==============================================================================

@torch.no_grad()
def calculate_attack_metrics(x_original, x_adv, y_true, model, threshold=1.0):
    """
    적대적 공격 결과 메트릭 계산 (통일된 인터페이스)
    
    Args:
        x_original: 원본 이미지 (1,C,H,W) [0,1]
        x_adv: 공격된 이미지 (1,C,H,W) [0,1]
        y_true: 실제 레이블 (int or tensor)
        model: 타겟 모델
        threshold: 변경 픽셀 판정 임계값 (0-255 스케일)
    
    Returns:
        metrics: 메트릭 딕셔너리
    """
    device = x_original.device
    
    # 섭동 계산
    delta = (x_adv - x_original).abs()
    
    # L1, L2, Linf 노름
    l1_norm = delta.sum().item()
    l2_norm = delta.pow(2).sum().sqrt().item()
    linf_norm = delta.max().item()
    
    # 0-255 스케일 L2
    l2_255 = (delta * 255.0).view(1, -1).norm(p=2).item()
    
    # 변경된 픽셀 수
    delta_255 = delta * 255.0
    changed_pixels = (delta_255 > threshold).sum().item()
    total_pixels = delta.numel()
    changed_ratio = changed_pixels / total_pixels * 100.0
    
    # 예측 결과
    logits_original = model(x_original)
    logits_adv = model(x_adv)
    
    pred_original = logits_original.argmax(1).item()
    pred_adv = logits_adv.argmax(1).item()
    
    # 레이블 처리
    if torch.is_tensor(y_true):
        if y_true.dim() == 0:
            y_idx = int(y_true.item())
        else:
            y_idx = int(y_true[0].item())
    else:
        y_idx = int(y_true)
    
    # 신뢰도 변화
    prob_original = torch.softmax(logits_original, dim=1)
    prob_adv = torch.softmax(logits_adv, dim=1)
    
    conf_original = prob_original[0, y_idx].item()
    conf_adv = prob_adv[0, y_idx].item()
    conf_drop = conf_original - conf_adv
    
    # 마진 변화 (margin = correct_logit - max_other_logit)
    correct_logit_original = logits_original[0, y_idx].item()
    mask = torch.ones(logits_original.size(1), dtype=torch.bool, device=device)
    mask[y_idx] = False
    max_other_original = logits_original[0, mask].max().item()
    margin_original = correct_logit_original - max_other_original
    
    correct_logit_adv = logits_adv[0, y_idx].item()
    max_other_adv = logits_adv[0, mask].max().item()
    margin_adv = correct_logit_adv - max_other_adv
    margin_drop = margin_original - margin_adv
    
    # 성공 여부
    success = (pred_adv != y_idx)
    
    return {
        'l1_norm': l1_norm,
        'l2_norm': l2_norm,
        'l2_255': l2_255,
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
# 시각화 함수
# ==============================================================================

def _tensor_to_numpy(x):
    """텐서를 numpy 이미지로 변환 (H,W,3)"""
    if x.dim() == 4:
        x = x.squeeze(0)
    return x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def visualize_attack_result(
    model, 
    x_original, 
    x_adv, 
    class_names=None, 
    roi_mask=None,
    amp_heat=3.0, 
    out_dir="./results", 
    filename_prefix="attack", 
    display=False,
    save_file=True
):
    """
    공격 결과 시각화 (통일된 인터페이스)
    
    Args:
        model: 타겟 모델
        x_original: (1,3,H,W) 원본 이미지 [0,1]
        x_adv: (1,3,H,W) 공격된 이미지 [0,1]
        class_names: 클래스 이름 리스트
        roi_mask: (H,W) ROI 마스크 (옵션)
        amp_heat: 섭동 히트맵 증폭 계수
        out_dir: 저장 디렉토리
        filename_prefix: 파일명 접두사
        display: True면 화면에 표시
        save_file: True면 파일로 저장
    
    Returns:
        저장된 파일 경로 (save_file=True인 경우)
    """
    if save_file:
        os.makedirs(out_dir, exist_ok=True)
    
    # 예측 및 신뢰도
    with torch.no_grad():
        logits_original = model(x_original)
        logits_adv = model(x_adv)
        
        probs_original = torch.softmax(logits_original, dim=1)
        probs_adv = torch.softmax(logits_adv, dim=1)
        
        pred_original = probs_original.argmax(1).item()
        pred_adv = probs_adv.argmax(1).item()
        
        conf_original = probs_original[0, pred_original].item()
        conf_adv = probs_adv[0, pred_adv].item()
    
    # 이미지 변환
    img_original = _tensor_to_numpy(x_original)
    img_adv = _tensor_to_numpy(x_adv)
    
    # 섭동 맵 (L2 노름)
    delta = (x_adv - x_original).squeeze(0).norm(p=2, dim=0).cpu().numpy()
    pert_map = delta * amp_heat
    
    # 시각화
    fig = plt.figure(figsize=(12, 4))
    
    # 원본 이미지
    ax = plt.subplot(1, 3, 1)
    ax.imshow(img_original)
    ax.set_title("Original", fontsize=12)
    ax.axis('off')
    
    # ROI 마스크 표시
    if roi_mask is not None:
        mask_np = (roi_mask.detach().cpu().numpy() if torch.is_tensor(roi_mask) else roi_mask).astype(float)
        ax.contour(mask_np, levels=[0.5], linewidths=1.5, colors='cyan')
    
    # 공격된 이미지
    ax = plt.subplot(1, 3, 2)
    ax.imshow(img_adv)
    ax.set_title("Adversarial", fontsize=12)
    ax.axis('off')
    
    if roi_mask is not None:
        ax.contour(mask_np, levels=[0.5], linewidths=1.5, colors='cyan')
    
    # 섭동 맵
    ax = plt.subplot(1, 3, 3)
    im = ax.imshow(pert_map, cmap='magma')
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Perturbation (×{amp_heat:g})", fontsize=12)
    ax.axis('off')
    
    # 타이틀
    def get_class_name(idx):
        if class_names and 0 <= idx < len(class_names):
            return class_names[idx]
        return str(idx)
    
    title = f"Before: {get_class_name(pred_original)} ({conf_original:.3f}) → After: {get_class_name(pred_adv)} ({conf_adv:.3f})"
    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    
    # 저장 또는 표시
    out_path = None
    if save_file:
        timestamp = int(time.time())
        out_path = os.path.join(out_dir, f"{filename_prefix}_{timestamp}.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"[VIS] 저장 완료: {out_path}")
    
    if display:
        plt.show()
    else:
        plt.close(fig)
    
    return out_path


# ==============================================================================
# 결과 저장 함수
# ==============================================================================

def save_results_to_csv(results, out_path):
    """
    공격 결과를 CSV로 저장
    
    Args:
        results: 결과 딕셔너리 리스트
        out_path: 저장 경로
    """
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[SAVE] CSV 저장 완료: {out_path}")


def save_results_to_json(results, out_path):
    """
    공격 결과를 JSON으로 저장
    
    Args:
        results: 결과 딕셔너리 또는 리스트
        out_path: 저장 경로
    """
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] JSON 저장 완료: {out_path}")


# ==============================================================================
# 통계 계산 함수
# ==============================================================================

def calculate_batch_statistics(results, attack_name="Attack"):
    """
    배치 공격 결과의 통계 계산 (통일된 인터페이스)
    
    Args:
        results: 결과 딕셔너리 리스트
        attack_name: 공격 이름
    
    Returns:
        statistics: 통계 딕셔너리
    """
    if not results:
        return {}
    
    df = pd.DataFrame(results)
    
    # 기본 통계
    total_samples = len(results)
    success_count = df['success'].sum() if 'success' in df.columns else 0
    success_rate = success_count / total_samples * 100 if total_samples > 0 else 0
    
    # 성공한 샘플만 필터링
    success_df = df[df['success'] == True] if 'success' in df.columns else df
    
    # 평균/표준편차 계산 헬퍼
    def safe_stats(column):
        if column in df.columns:
            return {
                'mean': float(df[column].mean()),
                'std': float(df[column].std()),
                'min': float(df[column].min()),
                'max': float(df[column].max())
            }
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    
    def safe_mean(column, dataframe):
        if column in dataframe.columns and len(dataframe) > 0:
            return float(dataframe[column].mean())
        return 0.0
    
    statistics = {
        'attack_name': attack_name,
        'total_samples': total_samples,
        'success_count': int(success_count),
        'success_rate': success_rate,
        
        # 전체 통계
        'l2_255': safe_stats('l2_255'),
        'l2_norm': safe_stats('l2_norm'),
        'changed_ratio': safe_stats('changed_ratio'),
        'margin_drop': safe_stats('margin_drop'),
        'conf_drop': safe_stats('conf_drop'),
        'elapsed_time': safe_stats('elapsed_time'),
        
        # 성공 케이스 평균
        'success_l2_255_mean': safe_mean('l2_255', success_df),
        'success_changed_ratio_mean': safe_mean('changed_ratio', success_df),
        'success_margin_drop_mean': safe_mean('margin_drop', success_df),
        'success_conf_drop_mean': safe_mean('conf_drop', success_df),
        'success_elapsed_time_mean': safe_mean('elapsed_time', success_df),
    }
    
    return statistics


def print_statistics(statistics):
    """
    통계 결과를 포맷팅하여 출력
    
    Args:
        statistics: calculate_batch_statistics의 반환값
    """
    if not statistics:
        print("[WARN] 통계 데이터가 없습니다.")
        return
    
    print(f"\n===== {statistics['attack_name']} 통계 =====")
    print(f"총 샘플 수                    : {statistics['total_samples']}")
    print(f"공격 성공 수                  : {statistics['success_count']}")
    print(f"공격 성공률 (ASR)             : {statistics['success_rate']:.2f}%")
    
    print("\n--- 전체 평균 ---")
    print(f"L2 (0~255 스케일)             : {statistics['l2_255']['mean']:.3f} ± {statistics['l2_255']['std']:.3f}")
    print(f"변경 픽셀 비율                : {statistics['changed_ratio']['mean']:.2f}% ± {statistics['changed_ratio']['std']:.2f}%")
    print(f"Margin 감소                   : {statistics['margin_drop']['mean']:.3f} ± {statistics['margin_drop']['std']:.3f}")
    print(f"신뢰도 감소                   : {statistics['conf_drop']['mean']:.3f} ± {statistics['conf_drop']['std']:.3f}")
    print(f"평균 소요 시간                : {statistics['elapsed_time']['mean']:.2f}s ± {statistics['elapsed_time']['std']:.2f}s")
    
    print("\n--- 성공 케이스 평균 ---")
    print(f"L2 (0~255 스케일)             : {statistics['success_l2_255_mean']:.3f}")
    print(f"변경 픽셀 비율                : {statistics['success_changed_ratio_mean']:.2f}%")
    print(f"Margin 감소                   : {statistics['success_margin_drop_mean']:.3f}")
    print(f"신뢰도 감소                   : {statistics['success_conf_drop_mean']:.3f}")
    print(f"평균 소요 시간                : {statistics['success_elapsed_time_mean']:.2f}s")


# ==============================================================================
# 시각화 차트 생성
# ==============================================================================

def create_result_visualization(results, out_dir="./results", attack_name="Attack"):
    """
    공격 결과의 시각화 차트 생성
    
    Args:
        results: 결과 딕셔너리 리스트
        out_dir: 저장 디렉토리
        attack_name: 공격 이름
    """
    if not results:
        print("[WARN] 시각화할 데이터가 없습니다.")
        return
    
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    
    # 그림 생성
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 성공률 파이 차트
    if 'success' in df.columns:
        plt.subplot(3, 3, 1)
        success_count = df['success'].sum()
        fail_count = len(df) - success_count
        plt.pie([success_count, fail_count], 
                labels=['성공', '실패'], 
                autopct='%1.1f%%',
                colors=['lightgreen', 'lightcoral'])
        plt.title(f'{attack_name} 성공률 (n={len(df)})')
    
    # 2. L2 노름 분포
    if 'l2_255' in df.columns:
        plt.subplot(3, 3, 2)
        plt.hist(df['l2_255'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('L2 섭동량 (0-255)')
        plt.ylabel('빈도')
        plt.title('L2 섭동량 분포')
        plt.axvline(df['l2_255'].mean(), color='red', linestyle='--', 
                   label=f'평균: {df["l2_255"].mean():.2f}')
        plt.legend()
    
    # 3. 변경 픽셀 비율 분포
    if 'changed_ratio' in df.columns:
        plt.subplot(3, 3, 3)
        plt.hist(df['changed_ratio'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('변경 픽셀 비율 (%)')
        plt.ylabel('빈도')
        plt.title('변경 픽셀 비율 분포')
        plt.axvline(df['changed_ratio'].mean(), color='red', linestyle='--',
                   label=f'평균: {df["changed_ratio"].mean():.2f}%')
        plt.legend()
    
    # 4. Margin 감소 분포
    if 'margin_drop' in df.columns:
        plt.subplot(3, 3, 4)
        plt.hist(df['margin_drop'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Margin 감소')
        plt.ylabel('빈도')
        plt.title('Margin 감소 분포')
        plt.axvline(df['margin_drop'].mean(), color='red', linestyle='--',
                   label=f'평균: {df["margin_drop"].mean():.3f}')
        plt.legend()
    
    # 5. 신뢰도 감소 분포
    if 'conf_drop' in df.columns:
        plt.subplot(3, 3, 5)
        plt.hist(df['conf_drop'], bins=20, alpha=0.7, color='pink', edgecolor='black')
        plt.xlabel('신뢰도 감소')
        plt.ylabel('빈도')
        plt.title('신뢰도 감소 분포')
        plt.axvline(df['conf_drop'].mean(), color='red', linestyle='--',
                   label=f'평균: {df["conf_drop"].mean():.3f}')
        plt.legend()
    
    # 6. 소요 시간 분포
    if 'elapsed_time' in df.columns:
        plt.subplot(3, 3, 6)
        plt.hist(df['elapsed_time'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        plt.xlabel('소요 시간 (초)')
        plt.ylabel('빈도')
        plt.title('소요 시간 분포')
        plt.axvline(df['elapsed_time'].mean(), color='red', linestyle='--',
                   label=f'평균: {df["elapsed_time"].mean():.2f}s')
        plt.legend()
    
    # 7. 성공 vs 실패 L2 비교 (박스플롯)
    if 'success' in df.columns and 'l2_255' in df.columns:
        plt.subplot(3, 3, 7)
        success_l2 = df[df['success'] == True]['l2_255']
        fail_l2 = df[df['success'] == False]['l2_255']
        plt.boxplot([success_l2, fail_l2], labels=['성공', '실패'])
        plt.ylabel('L2 섭동량 (0-255)')
        plt.title('성공 vs 실패 L2 비교')
    
    # 8. L2 vs 변경 픽셀 비율 산점도
    if 'l2_255' in df.columns and 'changed_ratio' in df.columns:
        plt.subplot(3, 3, 8)
        if 'success' in df.columns:
            success_mask = df['success'] == True
            plt.scatter(df[success_mask]['l2_255'], df[success_mask]['changed_ratio'], 
                       c='green', alpha=0.6, label='성공', s=30)
            plt.scatter(df[~success_mask]['l2_255'], df[~success_mask]['changed_ratio'], 
                       c='red', alpha=0.6, label='실패', s=30)
            plt.legend()
        else:
            plt.scatter(df['l2_255'], df['changed_ratio'], alpha=0.6, s=30)
        plt.xlabel('L2 섭동량 (0-255)')
        plt.ylabel('변경 픽셀 비율 (%)')
        plt.title('L2 vs 변경 픽셀 비율')
    
    # 9. 샘플별 성공 여부
    if 'success' in df.columns:
        plt.subplot(3, 3, 9)
        success_values = df['success'].astype(int).values
        plt.plot(success_values, 'o-', markersize=3, linewidth=0.5)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('샘플 번호')
        plt.ylabel('성공 (1) / 실패 (0)')
        plt.title('샘플별 공격 성공 여부')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    timestamp = int(time.time())
    out_path = os.path.join(out_dir, f"{attack_name.lower()}_visualization_{timestamp}.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[VIS] 차트 저장 완료: {out_path}")


# ==============================================================================
# 안전 평균 함수
# ==============================================================================

def safe_mean(arr):
    """빈 배열을 안전하게 처리하는 평균 계산"""
    if isinstance(arr, (list, tuple)):
        arr = np.array(arr)
    return float(np.mean(arr)) if len(arr) > 0 else 0.0


# ==============================================================================
# 체크포인트 및 데이터 수집 (adv_utils.py에서 통합)
# ==============================================================================

def find_ckpt(out_dir: str | Path) -> Path:
    """가장 흔한 위치에서 체크포인트 탐색"""
    out_dir = Path(out_dir)
    candidates = [
        out_dir / "resnet50_binary_best.pth",
        out_dir / "resnet50_binary_final.pth",
        Path("./resnet50_binary_best.pth"),
        Path("./resnet50_binary_final.pth"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Checkpoint not found. Tried:\n" + "\n".join(map(str, candidates)))


@torch.no_grad()
def collect_first_n_correct(
    model: torch.nn.Module,
    loader,
    n: int,
    device: torch.device,
    random_pick: bool = True,
    seed: Optional[int] = None,
):
    """
    테스트 로더에서 '정분류된' 샘플만 최대 n개 수집
    
    Args:
        model: 모델
        loader: 데이터로더
        n: 수집할 샘플 수
        device: 디바이스
        random_pick: 무작위 선택 여부
        seed: 랜덤 시드
    
    Returns:
        xN: (N,C,H,W) 이미지
        yN: (N,) 레이블
        mN: None 또는 (N,H,W) bool 마스크
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    xs, ys, ms, total = [], [], [], 0

    for batch in loader:
        if len(batch) == 2:
            x, y = batch
            masks = None
        else:
            x, y, masks = batch

        x = x.to(device)
        y = y.to(device)

        pred = model(x).argmax(1)
        correct_idx = torch.nonzero(pred == y, as_tuple=False).squeeze(1)

        if correct_idx.numel() == 0:
            continue

        # 무작위 순서로 선택
        if random_pick:
            perm = torch.randperm(correct_idx.numel(), device=correct_idx.device)
            correct_idx = correct_idx[perm]

        need = n - total
        take = min(need, correct_idx.numel())
        sel = correct_idx[:take]

        xs.append(x[sel].detach())
        ys.append(y[sel].detach())
        if masks is not None:
            ms.append(torch.as_tensor(masks, device=device)[sel].bool())

        total += take
        if total >= n:
            break

    if total == 0:
        raise RuntimeError("No correctly classified samples in loader.")
    if total < n:
        raise RuntimeError(f"Only found {total} correctly classified samples (< {n}).")

    xN = torch.cat(xs, dim=0)
    yN = torch.cat(ys, dim=0)
    mN = None if len(ms) == 0 else torch.cat(ms, dim=0)
    return xN, yN, mN


# ==============================================================================
# 공격별 특화 메트릭 함수 (adv_utils.py에서 통합)
# ==============================================================================

def summarize_jsma_metrics(
    success_arr: Sequence[bool] | torch.Tensor,
    changed_arr: Sequence[int] | torch.Tensor,
    l1_arr: Sequence[float] | torch.Tensor,
    image_hw: Tuple[int, int],
    times_sec: Optional[Sequence[float]] = None,
    print_summary: bool = False,
) -> Dict[str, float]:
    """
    JSMA 공격 결과 통계 요약
    
    Args:
        success_arr: 성공 여부 배열
        changed_arr: 변경된 픽셀 수 배열
        l1_arr: L1 노름 배열
        image_hw: (H, W) 이미지 크기
        times_sec: 소요 시간 배열
        print_summary: 출력 여부
    
    Returns:
        메트릭 딕셔너리
    """
    H, W = image_hw
    Npix = float(H * W)

    # to numpy
    succ = torch.as_tensor(success_arr, dtype=torch.bool).cpu().numpy()
    ch   = torch.as_tensor(changed_arr, dtype=torch.float32).cpu().numpy()
    l1   = torch.as_tensor(l1_arr, dtype=torch.float32).cpu().numpy()

    asr = float(np.mean(succ))
    mean_changed = float(np.mean(ch))
    median_changed = float(np.median(ch))
    mean_changed_pct = 100.0 * mean_changed / Npix
    median_changed_pct = 100.0 * median_changed / Npix

    mean_l1 = float(np.mean(l1))
    median_l1 = float(np.median(l1))

    # 평균 Δ/px (수정된 픽셀에 한정)
    eps = 1e-12
    mean_delta_per_px = float(np.mean(l1 / (ch + eps)))

    out = {
        "ASR": asr,
        "Mean_Changed": mean_changed,
        "Median_Changed": median_changed,
        "Mean_Changed_Pct": mean_changed_pct,
        "Median_Changed_Pct": median_changed_pct,
        "Mean_L1": mean_l1,
        "Median_L1": median_l1,
        "Mean_Delta_per_ModifiedPixel": mean_delta_per_px,
    }

    if times_sec is not None and len(times_sec) > 0:
        times = np.asarray(times_sec, dtype=np.float64)
        out["Mean_Time_sec"] = float(times.mean())
        out["Median_Time_sec"] = float(np.median(times))

    if print_summary:
        print("[JSMA] Summary")
        for k, v in out.items():
            if isinstance(v, float):
                print(f"  - {k}: {v:.6f}")
            else:
                print(f"  - {k}: {v}")

    return out


def summarize_zoo_metrics(
    success_arr, 
    changed_arr, 
    l1_arr, 
    queries_arr, 
    times_sec=None, 
    print_summary=False
):
    """
    ZOO 공격 결과 통계 요약
    
    Args:
        success_arr: 성공 여부 배열
        changed_arr: 변경된 픽셀 수 배열
        l1_arr: L1 노름 배열
        queries_arr: 쿼리 수 배열
        times_sec: 소요 시간 배열
        print_summary: 출력 여부
    
    Returns:
        메트릭 딕셔너리
    """
    # to numpy
    succ = torch.as_tensor(success_arr, dtype=torch.bool).cpu().numpy()
    ch   = torch.as_tensor(changed_arr, dtype=torch.float32).cpu().numpy()
    l1   = torch.as_tensor(l1_arr, dtype=torch.float32).cpu().numpy()
    qry  = torch.as_tensor(queries_arr, dtype=torch.float32).cpu().numpy()

    asr = float(np.mean(succ))
    mean_changed = float(np.mean(ch))
    median_changed = float(np.median(ch))
    mean_l1 = float(np.mean(l1))
    median_l1 = float(np.median(l1))
    mean_queries = float(np.mean(qry))
    median_queries = float(np.median(qry))

    out = {
        "ASR": asr,
        "Mean_Changed": mean_changed,
        "Median_Changed": median_changed,
        "Mean_L1": mean_l1,
        "Median_L1": median_l1,
        "Mean_Queries": mean_queries,
        "Median_Queries": median_queries,
    }

    if times_sec is not None and len(times_sec) > 0:
        times = np.asarray(times_sec, dtype=np.float64)
        out["Mean_Time_sec"] = float(times.mean())
        out["Median_Time_sec"] = float(np.median(times))

    if print_summary:
        print("[ZOO] Summary")
        for k, v in out.items():
            if isinstance(v, float):
                print(f"  - {k}: {v:.6f}")
            else:
                print(f"  - {k}: {v}")

    return out


def stats_table_like(x_pix_clean, x_adv_pix, elapsed_sec, count_mode='element', tau_255=1.0):
    """
    공격 결과 통계 계산 (ZOO용)
    
    Args:
        x_pix_clean: 원본 이미지 [0,1]
        x_adv_pix: 공격된 이미지 [0,1]
        elapsed_sec: 소요 시간
        count_mode: 'element' 또는 'spatial'
        tau_255: 변경 판정 임계값 (0-255 스케일)
    
    Returns:
        통계 딕셔너리
    """
    with torch.no_grad():
        delta = (x_adv_pix - x_pix_clean)
        l2_255 = (delta * 255.0).view(1, -1).norm(p=2).item()
        
        if count_mode == 'element':
            # 픽셀 단위로 변경량 계산
            changed = ((delta.abs() * 255.0) > tau_255).sum().item()
            total   = int(delta.numel())
        else:
            # 공간 단위로 변경량 계산
            per_spatial = delta.abs().max(dim=1)[0].squeeze(0)
            changed = ((per_spatial * 255.0) > tau_255).sum().item()
            total   = int(per_spatial.numel())
    
    return {
        "L2_255": l2_255, 
        "changed": int(changed), 
        "total": total, 
        "time": float(elapsed_sec)
    }


# ==============================================================================
# 추가 시각화 함수 (adv_utils.py에서 통합)
# ==============================================================================

def save_side_by_side(x_pix_clean, x_adv_pix, path_png, left_title="Original", right_title="Adversarial"):
    """
    원본과 공격 이미지를 나란히 저장
    
    Args:
        x_pix_clean: 원본 이미지 (1,C,H,W) [0,1]
        x_adv_pix: 공격 이미지 (1,C,H,W) [0,1]
        path_png: 저장 경로
        left_title: 왼쪽 타이틀
        right_title: 오른쪽 타이틀
    """
    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()
    orig_img = to_pil(x_pix_clean.squeeze().cpu())
    adv_img  = to_pil(x_adv_pix.squeeze().cpu())
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img)
    plt.title(left_title)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(adv_img)
    plt.title(right_title)
    plt.axis("off")
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    plt.savefig(path_png, dpi=150)
    plt.close()
    print(f"[VIS] 저장 완료: {path_png}")


def visualize_zoo_triplet(
    x, 
    x_adv, 
    save_path, 
    title="ZOO Adversarial Example", 
    show=False, 
    dpi=220
):
    """
    ZOO 공격 결과 트리플릿 시각화
    
    Args:
        x: 원본 이미지 (1,C,H,W) [0,1]
        x_adv: 공격 이미지 (1,C,H,W) [0,1]
        save_path: 저장 경로
        title: 타이틀
        show: 화면 표시 여부
        dpi: 해상도
    """
    def _to_img01(t):
        if t.ndim == 4:
            t = t[0]
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        arr = t.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        return np.clip(arr, 0.0, 1.0)
    
    x0 = _to_img01(x)
    xa = _to_img01(x_adv)

    fig = plt.figure(figsize=(12, 4))

    ax = plt.subplot(1, 3, 1)
    ax.imshow(x0)
    ax.axis("off")
    ax.set_title("Original")

    ax = plt.subplot(1, 3, 2)
    ax.imshow(xa)
    ax.axis("off")
    ax.set_title("Adversarial")

    ax = plt.subplot(1, 3, 3)
    diff = xa - x0
    mag = np.sqrt(np.sum(diff * diff, axis=2))
    vmax = np.percentile(mag.ravel(), 99.5) if np.any(mag > 0) else 1.0
    im = ax.imshow(np.clip(mag / (vmax + 1e-12), 0.0, 1.0), cmap="magma", vmin=0, vmax=1)
    ax.set_title("Perturbation ‖Δ‖")
    ax.axis("off")

    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    plt.suptitle(title)
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[VIS] 저장 완료: {save_path}")