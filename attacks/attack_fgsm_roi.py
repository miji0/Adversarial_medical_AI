"""
FGSM ROI (Fast Gradient Sign Method - Region of Interest)
- ROI 영역 내에서만 섭동 적용
- 픽셀 공간 기준 공격
"""

import time
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
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
        if self.meta is not None:
            print(f"[INFO] Meta: {self.meta}")

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

def to_norm(x):
    """모델이 내부에서 정규화하는 경우 사용"""
    return x


# ==============================================================================
# FGSM ROI 공격
# ==============================================================================

def fgsm_attack_roi_pixelspace(model, image, label, roi_mask, eps=8/255.0):
    """
    FGSM ROI 공격 (ROI 영역에만 섭동 추가, 픽셀 공간 기준)
    
    Args:
        model: torch.nn.Module
        image: (1,C,H,W) tensor, [0,1] 범위
        label: (1,) tensor
        roi_mask: (H,W) tensor {0,1}
        eps: 0~1 scale (픽셀 단위 섭동 크기)
        
    Returns:
        adv_image: 공격된 이미지
        perturb: 섭동
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


# ==============================================================================
# 시각화
# ==============================================================================

def visualize_fgsm_roi(model, x, x_adv, pert, roi_mask, class_names):
    """
    FGSM ROI 공격 결과 시각화
    
    Args:
        model: 타겟 모델
        x: 원본 이미지
        x_adv: 공격된 이미지
        pert: 섭동
        roi_mask: ROI 마스크
        class_names: 클래스 이름 리스트
    """
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


# ==============================================================================
# 배치 실행 함수
# ==============================================================================

def run_fgsm_roi_attack(model, test_set, class_names, eps=8/255.0, n_samples=100):
    """
    테스트셋 앞 n_samples 장에 대해 FGSM ROI 공격 실행
    
    Args:
        model: 타겟 모델
        test_set: 테스트 데이터셋 (ROI 포함)
        class_names: 클래스 이름 리스트
        eps: 섭동 크기 (0~1 스케일)
        n_samples: 샘플 수
        
    Returns:
        ASR: 공격 성공률
        mean_L2_255: 평균 L2 노름
        mean_changed_pct: 평균 변경 픽셀 비율
        mean_time: 평균 생성 시간
        success_list: 성공 여부 리스트
    """
    device = next(model.parameters()).device
    success_list, l2_list, changed_list, total_list, elapsed_list = [], [], [], [], []
    dmargin_list, conf_drop_list = [], []

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

        # L2 & 변경 픽셀 계산
        delta = (x_adv - x).cpu()
        l2_255 = (delta.view(1,-1)*255).norm(p=2).item()
        changed = ((delta.abs()*255) > 1.0).sum().item()
        total = int(roi.numel())
        l2_list.append(l2_255)
        changed_list.append(changed)
        total_list.append(total)

        # ΔMargin / ΔConfidence
        margin_clean = _margin_from_logits(logits_clean, y_idx)
        margin_adv   = _margin_from_logits(logits_adv,   y_idx)
        dmargin_list.append(float(margin_clean - margin_adv))

        probs_clean = torch.softmax(logits_clean, dim=1)[0]
        probs_adv   = torch.softmax(logits_adv,   dim=1)[0]
        conf_drop_list.append(float(probs_clean[y_idx] - probs_adv[y_idx]))

        # 한 장마다 결과 출력
        print(f"[{i+1}/{n_samples}] 공격 {'성공 ✅' if success else '실패 ❌'} "
              f"(원래: {class_names[pred_orig]} → 적대: {class_names[pred_adv]})")

        # 첫 장만 시각화
        if i == 0:
            visualize_fgsm_roi(model, x, x_adv, pert, roi, class_names)

    # 안전 평균 함수
    def _safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else float('nan')

    # 전체 평균 통계
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

    print("\n===== FGSM ROI Attack (Batch) 평균 통계 =====")
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


# ==============================================================================
# 메인 실행 예제
# ==============================================================================

if __name__ == "__main__":
    print("FGSM ROI Attack 모듈")
    print("사용법:")
    print("  from attacks.attack_fgsm_roi import run_fgsm_roi_attack, BrainTumorDatasetWithROI")
    print("  from torchvision import transforms")
    print("  ")
    print("  test_set = BrainTumorDatasetWithROI(")
    print("      'path/to/Testing',")
    print("      transform=transforms.ToTensor()")
    print("  )")
    print("  ")
    print("  ASR, mean_L2, mean_changed_pct, mean_time, success_list = run_fgsm_roi_attack(")
    print("      model, test_set, class_names, eps=8/255.0, n_samples=100")
    print("  )")

