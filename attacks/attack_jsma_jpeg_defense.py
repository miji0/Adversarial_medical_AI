"""
JSMA + JPEG Defense 평가
- JSMA 공격 후 JPEG 압축 방어의 효과 측정
- 방어 전후 ASR, 마진, 신뢰도 변화 비교
"""

import io
import time
import os
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset
import json

# JSMA 공격 모듈 import
try:
    from attack_jsma import jsma_attack, BrainTumorDatasetWithROI
except ImportError:
    # 상대 경로로 시도
    from .attack_jsma import jsma_attack, BrainTumorDatasetWithROI


# ==============================================================================
# JPEG 압축 유틸
# ==============================================================================

def jpeg_compress_tensor_pil(x_tensor, quality=75):
    """
    JPEG 압축 (PIL 사용)
    
    Args:
        x_tensor: torch.Tensor (1,C,H,W) or (C,H,W) in [0,1] (float)
        quality: JPEG 품질 (1-100)
        
    Returns:
        압축된 텐서 (same shape as input)
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


# ==============================================================================
# 마진 계산 유틸
# ==============================================================================

def margin_from_logits(logits, y_idx):
    """
    마진 계산: logit[y] - max_{j!=y} logit[j]
    
    Args:
        logits: tensor shape (1, num_classes)
        y_idx: int
        
    Returns:
        margin: float
    """
    with torch.no_grad():
        logits0 = logits[0]
        tmp = logits0.clone()
        tmp[y_idx] = -1e9
        max_other = tmp.max().item()
        margin = float((logits0[y_idx].item() - max_other))
    return margin


# ==============================================================================
# 샘플별 평가 함수
# ==============================================================================

@torch.no_grad()
def eval_example_with_jpeg(model, x_orig_pix, x_adv_pix, mean, std, quality=75, y_true=None):
    """
    JPEG 방어 전후 평가
    
    Args:
        model: 타겟 모델
        x_orig_pix: 원본 이미지 (1,C,H,W) [0,1]
        x_adv_pix: 공격된 이미지 (1,C,H,W) [0,1]
        mean: 정규화 평균
        std: 정규화 표준편차
        quality: JPEG 품질
        y_true: 실제 레이블
        
    Returns:
        metrics: 메트릭 딕셔너리
    """
    device = next(model.parameters()).device

    # Normalize helper
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

    # 원본 및 공격 이미지 평가
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

    # 실제 레이블 기준 메트릭
    if y_true is None:
        y_idx = pred_clean
    else:
        y_idx = int(y_true)

    margin_clean = margin_from_logits(logits_clean, y_idx)
    margin_adv   = margin_from_logits(logits_adv, y_idx)
    margin_drop = margin_clean - margin_adv
    conf_drop = float((torch.softmax(logits_clean, dim=1)[0, y_idx] - 
                      torch.softmax(logits_adv, dim=1)[0, y_idx]).item())

    # JPEG 압축 적용
    x_adv_jpeg_pix = jpeg_compress_tensor_pil(x_adv_pix, quality=quality).to(device)
    xadv_jpeg_norm = _to_norm_local(x_adv_jpeg_pix)

    logits_adv_jpeg = model(xadv_jpeg_norm)
    probs_adv_jpeg = torch.softmax(logits_adv_jpeg, dim=1)[0]
    pred_adv_jpeg = int(probs_adv_jpeg.argmax().item())
    conf_adv_jpeg = float(probs_adv_jpeg[pred_adv_jpeg].item())

    margin_adv_jpeg = margin_from_logits(logits_adv_jpeg, y_idx)
    margin_drop_after = margin_clean - margin_adv_jpeg
    conf_drop_after = float((torch.softmax(logits_clean, dim=1)[0, y_idx] - 
                            probs_adv_jpeg[y_idx]).item())

    # 섭동 노름 (픽셀 공간)
    delta_adv = (x_adv_pix - x_orig_pix).detach().cpu()
    l2_255 = (delta_adv.view(1, -1) * 255.0).norm(p=2).item()
    l1 = delta_adv.abs().sum().item()
    linf = delta_adv.abs().max().item()

    # 변경 픽셀 수
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
        'x_adv_jpeg_pix': x_adv_jpeg_pix
    }


# ==============================================================================
# 배치 실험 함수
# ==============================================================================

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
    JSMA 공격 후 JPEG 방어 효과 평가
    
    Args:
        model: 타겟 모델
        test_set: 테스트 데이터셋
        class_names: 클래스 이름 리스트
        mean: 정규화 평균
        std: 정규화 표준편차
        n_samples: 샘플 수
        jsma_params: JSMA 파라미터 딕셔너리
        jpeg_quality: JPEG 품질
        visualize_first: 첫 샘플 시각화 여부
        viz_func: 시각화 함수
        
    Returns:
        results: 결과 딕셔너리
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

    # JSMA 호출 wrapper
    def _call_jsma(x, y):
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

        x_adv = x_adv.to(device)
        times.append(elapsed)

        # JPEG 방어 전후 평가
        metrics = eval_example_with_jpeg(model, x, x_adv, mean, std, quality=jpeg_quality, y_true=y)

        # 결과 수집
        success_before_list.append(int(metrics['success_before']))
        success_after_list.append(int(metrics['success_after_jpeg']))
        l2_list.append(metrics['l2_255'])
        changed_list.append(metrics['changed_pixels'])
        total_list.append(metrics['total_pixels'])
        margin_drop_before_list.append(metrics['margin_drop_before'])
        margin_drop_after_list.append(metrics['margin_drop_after'])
        conf_drop_before_list.append(metrics['conf_drop_before'])
        conf_drop_after_list.append(metrics['conf_drop_after'])

        # 샘플별 출력
        print(f"[{i+1}/{n_run}] JSMA success_before={metrics['success_before']} "
              f"-> after JPEG success={metrics['success_after_jpeg']} | "
              f"pred: {class_names[metrics['pred_clean']]} -> adv:{class_names[metrics['pred_adv']]} -> "
              f"adv+jpeg:{class_names[metrics['pred_adv_jpeg']]} "
              f"| L2_255={metrics['l2_255']:.3f} changed={metrics['changed_pixels']}/{metrics['total_pixels']} "
              f"time={elapsed:.2f}s")

        # 첫 샘플 시각화
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

    # 결과 집계
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


# ==============================================================================
# 시각화 함수
# ==============================================================================

def _to_norm_local(x_pix, mean, std):
    """픽셀 공간을 정규화"""
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
    """예측과 신뢰도 반환"""
    with torch.no_grad():
        logits = model(_to_norm_local(x_pix, mean, std))
        probs = torch.softmax(logits, dim=1)
        pred  = int(probs.argmax(1).item())
        conf  = float(probs[0, pred].item())
    return pred, conf


def _to_numpy_img(x):
    """텐서를 numpy 이미지로 변환"""
    if x.dim()==4: x = x.squeeze(0)
    return x.detach().clamp(0,1).permute(1,2,0).cpu().numpy()


def visualize_attack_with_jpeg(
    model,
    x_orig_pix,
    x_adv_pix,
    x_adv_jpeg_pix,
    class_names=None,
    mean=None, std=None,
    amp_delta=3.0,
    out_dir=".",
    filename_prefix="attack_vis_jpeg",
    display=True
):
    """
    JPEG 방어 전후 시각화
    
    Args:
        model: 타겟 모델
        x_orig_pix: 원본 (1,C,H,W)
        x_adv_pix: 공격 (1,C,H,W)
        x_adv_jpeg_pix: JPEG 적용 (1,C,H,W)
        class_names: 클래스 이름 리스트
        mean/std: 정규화 파라미터
        amp_delta: 섭동 증폭 계수
        out_dir: 저장 디렉토리
        filename_prefix: 파일명 접두사
        display: 화면 표시 여부
    """
    os.makedirs(out_dir, exist_ok=True)

    # 예측/신뢰도
    p0,c0 = _pred_and_conf(model, x_orig_pix,      mean, std)
    p1,c1 = _pred_and_conf(model, x_adv_pix,       mean, std)
    p2,c2 = _pred_and_conf(model, x_adv_jpeg_pix,  mean, std)

    # 이미지/맵 준비
    img0 = _to_numpy_img(x_orig_pix)
    img1 = _to_numpy_img(x_adv_pix)
    img2 = _to_numpy_img(x_adv_jpeg_pix)

    # 제거된 섭동 (JPEG가 없앤 성분)
    delta_removed = (x_adv_pix - x_adv_jpeg_pix).squeeze(0).norm(p=2, dim=0).cpu().numpy()
    delta_removed_vis = delta_removed * amp_delta

    # 그리기
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,4,1)
    ax.imshow(img0)
    ax.set_title(f"Original\n{class_names[p0]}({c0:.3f})")
    ax.axis('off')
    
    ax = plt.subplot(1,4,2)
    ax.imshow(img1)
    ax.set_title(f"Adversarial\n{class_names[p1]}({c1:.3f})")
    ax.axis('off')
    
    ax = plt.subplot(1,4,3)
    ax.imshow(img2)
    ax.set_title(f"Adv + JPEG\n{class_names[p2]}({c2:.3f})")
    ax.axis('off')
    
    ax = plt.subplot(1,4,4)
    im = ax.imshow(delta_removed_vis, cmap='magma')
    plt.colorbar(im)
    ax.set_title(f"|δ_removed| (x{amp_delta:g})")
    ax.axis('off')

    title = f"Before→After: {class_names[p0]}({c0:.3f}) → {class_names[p1]}({c1:.3f}) → {class_names[p2]}({c2:.3f})"
    plt.suptitle(title, y=1.03)
    plt.tight_layout()

    ts = int(time.time())
    save_path = os.path.join(out_dir, f"{filename_prefix}_{ts}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if display:
        plt.show()
    else:
        plt.close(fig)
    print(f"[VIS] Saved to: {save_path}")
    return save_path


# ==============================================================================
# 메인 실행 예제
# ==============================================================================

if __name__ == "__main__":
    print("JSMA + JPEG Defense 평가 모듈")
    print("사용법:")
    print("  from attacks.attack_jsma_jpeg_defense import run_jsma_and_eval_jpeg_defense")
    print("  results = run_jsma_and_eval_jpeg_defense(")
    print("      model, test_set, class_names, mean=[0,0,0], std=[1,1,1],")
    print("      n_samples=50, jsma_params={'theta': 0.12}, jpeg_quality=60")
    print("  )")

