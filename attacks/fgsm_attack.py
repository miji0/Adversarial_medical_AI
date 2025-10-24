"""
FGSM (Fast Gradient Sign Method) Attack Implementation
뇌종양 의료 이미지 분류 모델에 대한 FGSM 공격 구현
"""

# 전역 변수
attack_utils_module = None

import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import json
import matplotlib.pyplot as plt

# attack_utils는 setup_environment() 후에 임포트하도록 변경


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
# to_norm, to_pixel: attack_utils에서 제공


# ==============================================================================
# FGSM 공격 - 전체 이미지
# ==============================================================================

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

    # 모델 입력은 정규화 버전 (integrated.py와 동일하게 정규화 없음)
    output = model(image)
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
    """FGSM 전체 이미지 공격 시각화"""
    import matplotlib.pyplot as plt
    
    global attack_utils_module
    
    if attack_utils_module is None:
        print("[ERROR] attack_utils 모듈이 로드되지 않았습니다.")
        print("[INFO] ensure_attack_utils_import() 함수가 먼저 호출되었는지 확인하세요.")
        return

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
        pred_orig = model(attack_utils_module.to_norm(x)).argmax(1).item()
        pred_adv  = model(attack_utils_module.to_norm(x_adv)).argmax(1).item()

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


def run_fgsm_full_attack(model, test_set, class_names, eps=8/255.0, n_samples=100,
                         save_results=True, create_visualizations=True, out_dir="./fgsm_results"):
    """
    테스트셋 앞 n_samples 장에 대해 FGSM 전체 이미지 공격 실행 (통일된 인터페이스)
    
    Args:
        model: 타겟 모델
        test_set: 테스트 데이터셋
        class_names: 클래스 이름 리스트
        eps: epsilon 값 (0~1 스케일)
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
    
    # 랜덤 샘플링으로 편향 방지 (Zoo와 동일한 방식)
    n_total = len(test_set)
    n_samples = min(n_samples, n_total)
    
    # 시드 고정하여 재현 가능한 랜덤 샘플링
    np.random.seed(42)
    sampled_indices = np.random.choice(n_total, size=n_samples, replace=False)
    
    print(f"\n===== FGSM Full Image Attack (eps={eps:.4f}) =====")
    print(f"공격 대상: {n_samples}개 샘플 (랜덤 샘플링)\n")
    
    for idx, i in enumerate(sampled_indices):
        sample = test_set[i]
        
        # Dataset 클래스에 따라 (img, label) 또는 (img, label, mask)
        if len(sample) == 2:
            img, label = sample
            roi_mask = None
        else:
            img, label, roi_mask = sample
        
        x_original = img.unsqueeze(0).to(device)  # (1,3,H,W)
        y = torch.tensor([label], device=device)
        
        # FGSM 공격 실행
        start_time = time.time()
        x_adv, pert = fgsm_attack_full(model, x_original, y, eps=eps)
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
        metrics['epsilon'] = eps
        metrics['sample_idx'] = i
        
        # 결과 저장
        results.append(metrics)
        
        # 진행 상황 출력 (통일된 형식)
        pred_orig = metrics['pred_original']
        pred_adv = metrics['pred_adv']
        success_mark = '✅' if metrics['success'] else '❌'
        print(f"[{i+1:3d}/{n_samples}] {success_mark} {class_names[pred_orig]} → {class_names[pred_adv]} "
              f"| L2={metrics['l2_255']:.2f} | Δconf={metrics['conf_drop']:.3f} | {elapsed_time:.3f}s")
        
        # 첫 장 시각화 (통일된 함수 사용)
        if i == 0:
            if attack_utils_module is not None:
                attack_utils_module.visualize_attack_result(
                    model, x_original, x_adv,
                    class_names=class_names,
                    roi_mask=roi_mask,
                    amp_heat=4.0,
                    out_dir=out_dir,
                    filename_prefix="fgsm_sample0",
                    display=False,
                    save_file=save_results
                )
            else:
                print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 시각화를 건너뜁니다.")
    
    # 통계 계산 (통일된 함수 사용)
    if attack_utils_module is not None:
        statistics = attack_utils_module.calculate_batch_statistics(results, attack_name="FGSM")
        attack_utils_module.print_statistics(statistics)
    else:
        print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 기본 통계만 출력합니다.")
        ASR = float(np.mean([r['success'] for r in results]) * 100.0)
        mean_l2 = float(np.mean([r['l2_norm'] for r in results]))
        mean_time = float(np.mean([r['elapsed_time'] for r in results]))
        print(f"\n[FGSM Attack Results]")
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
            attack_utils_module.save_results_to_csv(results, os.path.join(out_dir, "fgsm_results.csv"))
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. CSV 저장을 건너뜁니다.")
        
        # JSON 저장
        if attack_utils_module is not None:
            attack_utils_module.save_results_to_json(statistics, os.path.join(out_dir, "fgsm_statistics.json"))
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. JSON 저장을 건너뜁니다.")
    
    # 시각화 차트 생성
    if create_visualizations:
        if attack_utils_module is not None:
            attack_utils_module.create_result_visualization(results, out_dir=out_dir, attack_name="FGSM")
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 시각화 차트 생성을 건너뜁니다.")
    
    return results, statistics


# ==============================================================================
# FGSM 공격 - ROI (Region of Interest)
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
    """
    device = next(model.parameters()).device
    image = image.clone().detach().to(device).requires_grad_(True)
    label = label.to(device)

    # 모델 입력은 정규화된 이미지 (integrated.py와 동일하게 정규화 없음)
    output = model(image)
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
    """FGSM ROI 공격 시각화"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    global attack_utils_module
    
    if attack_utils_module is None:
        print("[ERROR] attack_utils 모듈이 로드되지 않았습니다.")
        print("[INFO] ensure_attack_utils_import() 함수가 먼저 호출되었는지 확인하세요.")
        return

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
        pred_orig = model(attack_utils_module.to_norm(x)).argmax(1).item()
        pred_adv  = model(attack_utils_module.to_norm(x_adv)).argmax(1).item()

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


def run_fgsm_roi_attack(model, test_set, class_names, eps=8/255.0, n_samples=100,
                        save_results=True, create_visualizations=True, out_dir="./fgsm_roi_results"):
    """
    테스트셋 앞 n_samples 장에 대해 FGSM ROI 공격 실행 (통일된 인터페이스)
    
    Args:
        model: 타겟 모델
        test_set: 테스트 데이터셋 (ROI 마스크 포함)
        class_names: 클래스 이름 리스트
        eps: epsilon 값 (0~1 스케일)
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
    
    # 랜덤 샘플링으로 편향 방지 (Zoo와 동일한 방식)
    n_total = len(test_set)
    n_samples = min(n_samples, n_total)
    
    # 시드 고정하여 재현 가능한 랜덤 샘플링
    np.random.seed(42)
    sampled_indices = np.random.choice(n_total, size=n_samples, replace=False)
    
    print(f"\n===== FGSM ROI Attack (eps={eps:.4f}) =====")
    print(f"공격 대상: {n_samples}개 샘플 (ROI 영역만, 랜덤 샘플링)\n")
    
    for idx, i in enumerate(sampled_indices):
        img, label, roi_mask = test_set[i]
        
        x_original = img.unsqueeze(0).to(device)  # (1,3,H,W)
        y = torch.tensor([label], device=device)
        roi = torch.as_tensor(roi_mask, dtype=torch.float32)  # (H,W)
        
        # FGSM ROI 공격 실행
        start_time = time.time()
        x_adv, pert = fgsm_attack_roi_pixelspace(model, x_original, y, roi, eps=eps)
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
        metrics['epsilon'] = eps
        metrics['sample_idx'] = i
        metrics['use_roi'] = True
        
        # 결과 저장
        results.append(metrics)
        
        # 진행 상황 출력 (통일된 형식)
        pred_orig = metrics['pred_original']
        pred_adv = metrics['pred_adv']
        success_mark = '✅' if metrics['success'] else '❌'
        print(f"[{i+1:3d}/{n_samples}] {success_mark} {class_names[pred_orig]} → {class_names[pred_adv]} "
              f"| L2={metrics['l2_255']:.2f} | Δconf={metrics['conf_drop']:.3f} | {elapsed_time:.3f}s")
        
        # 첫 장 시각화 (통일된 함수 사용)
        if i == 0:
            if attack_utils_module is not None:
                attack_utils_module.visualize_attack_result(
                    model, x_original, x_adv,
                    class_names=class_names,
                    roi_mask=roi,
                    amp_heat=4.0,
                    out_dir=out_dir,
                    filename_prefix="fgsm_roi_sample0",
                    display=False,
                    save_file=save_results
                )
            else:
                print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 시각화를 건너뜁니다.")
    
    # 통계 계산 (통일된 함수 사용)
    if attack_utils_module is not None:
        statistics = attack_utils_module.calculate_batch_statistics(results, attack_name="FGSM_ROI")
        attack_utils_module.print_statistics(statistics)
    else:
        print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 기본 통계만 출력합니다.")
        ASR = float(np.mean([r['success'] for r in results]) * 100.0)
        mean_l2 = float(np.mean([r['l2_norm'] for r in results]))
        mean_time = float(np.mean([r['elapsed_time'] for r in results]))
        print(f"\n[FGSM ROI Attack Results]")
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
            attack_utils_module.save_results_to_csv(results, os.path.join(out_dir, "fgsm_roi_results.csv"))
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. CSV 저장을 건너뜁니다.")
        
        # JSON 저장
        if attack_utils_module is not None:
            attack_utils_module.save_results_to_json(statistics, os.path.join(out_dir, "fgsm_roi_statistics.json"))
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. JSON 저장을 건너뜁니다.")
    
    # 시각화 차트 생성
    if create_visualizations:
        if attack_utils_module is not None:
            attack_utils_module.create_result_visualization(results, out_dir=out_dir, attack_name="FGSM_ROI")
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 시각화 차트 생성을 건너뜁니다.")
    
    return results, statistics


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
# 사용 예제
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FGSM Attack 실행")
    print("="*70)
    
    # 1. 환경 설정
    base_path, is_colab = setup_environment()
    
    # 2. 시드 고정
    seed_everything(42)
    
    # 3. 로깅 설정
    log_file = setup_logging("fgsm_results")
    
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
            sys.exit(1)
        
        if not data_path.exists():
            print(f"[ERROR] 데이터 경로를 찾을 수 없습니다: {data_path}")
            print("[INFO] 전처리된 데이터를 먼저 준비해주세요.")
            sys.exit(1)
        
        # 모델 로드
        model = load_trained_model(str(checkpoint_path), device=device)
        print("[INFO] 모델 로드 완료")
        
        # 3. 테스트셋 로드
        test_set = BrainTumorDatasetWithROI(
            str(data_path),
            transform=transforms.ToTensor()
        )
        class_names = test_set.class_names
        print(f"[INFO] 테스트 데이터 로드 완료: {len(test_set)}개 샘플")
        
        # 4. 공격 파라미터 설정
        eps = 8/255.0  # epsilon 값
        n_samples = 100  # 테스트할 샘플 수 (필요시 조정)
        
        print("\n" + "="*70)
        print("FGSM 전체 이미지 공격 (Full Image Attack)")
        print("="*70)
        
        # 5. FGSM 전체 이미지 공격 실행
        results_full, statistics_full = run_fgsm_full_attack(
            model, test_set, class_names, eps=eps, n_samples=n_samples
        )
        
        # 통계에서 필요한 값들 추출
        ASR_full = statistics_full.get('ASR', 0.0)
        mean_L2_full = statistics_full.get('mean_l2', 0.0)
        mean_changed_pct_full = statistics_full.get('mean_changed_ratio', 0.0)
        mean_time_full = statistics_full.get('mean_time', 0.0)
        success_list_full = [r['success'] for r in results_full]
        
        print("\n" + "="*70)
        print("FGSM ROI 공격 (Region of Interest Attack)")
        print("="*70)
        
        # 6. FGSM ROI 공격 실행
        results_roi, statistics_roi = run_fgsm_roi_attack(
            model, test_set, class_names, eps=eps, n_samples=n_samples
        )
        
        # 통계에서 필요한 값들 추출
        ASR_roi = statistics_roi.get('ASR', 0.0)
        mean_L2_roi = statistics_roi.get('mean_l2', 0.0)
        mean_changed_pct_roi = statistics_roi.get('mean_changed_ratio', 0.0)
        mean_time_roi = statistics_roi.get('mean_time', 0.0)
        success_list_roi = [r['success'] for r in results_roi]
        
        print("\n" + "="*70)
        print("공격 비교 요약")
        print("="*70)
        print(f"{'지표':<30} {'Full Image':<20} {'ROI':<20}")
        print("-"*70)
        print(f"{'Attack Success Rate (ASR)':<30} {ASR_full:>18.2f}% {ASR_roi:>18.2f}%")
        print(f"{'Mean L2 Distance':<30} {mean_L2_full:>18.3f} {mean_L2_roi:>18.3f}")
        print(f"{'Mean Changed Pixels':<30} {mean_changed_pct_full:>17.2f}% {mean_changed_pct_roi:>17.2f}%")
        print(f"{'Mean Generation Time (sec)':<30} {mean_time_full:>18.2f} {mean_time_roi:>18.2f}")
        print("="*70)
        
    except ImportError as e:
        print(f"[ERROR] 모듈 임포트 실패: {e}")
        print("[INFO] resnet50_model.py 파일이 같은 디렉토리에 있는지 확인하세요.")
    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

