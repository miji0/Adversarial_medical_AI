"""
ZOO (Zeroth Order Optimization) Attack Implementation - 간소화
뇌종양 의료 이미지 분류 모델에 대한 ZOO 공격 구현
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple, List, Union
import warnings

warnings.filterwarnings('ignore')

# 전역 변수
attack_utils_module = None

# ==============================================================================
# 환경 설정 (통합)
# ==============================================================================

def setup_zoo_environment(seed=42, device_config='auto', log_dir="zoo_results"):
    """ZOO 공격을 위한 통합 환경 설정"""
    import random
    import logging
    from datetime import datetime
    
    # 1. 환경 감지 및 경로 설정
    try:
        import google.colab
        is_colab = True
        print("[INFO] Google Colab 환경 감지")
        from google.colab import drive
        drive.mount('/content/drive')
        base_path = Path("/content/drive/MyDrive/Adversarial_medical_AI")
        os.chdir(base_path)
        print(f"[INFO] 작업 디렉토리 변경: {base_path}")
        if str(base_path) not in sys.path:
            sys.path.append(str(base_path))
    except ImportError:
        is_colab = False
        print("[INFO] 로컬 환경 감지")
        base_path = Path.cwd()
    
    # 2. 시드 고정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] 시드 고정 완료: {seed}")
    
    # 3. 디바이스 설정
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_amp = torch.cuda.is_available()
    elif device_config == 'cuda':
        device = torch.device('cuda')
        use_amp = True
    else:
        device = torch.device('cpu')
        use_amp = False
    
    print(f"[INFO] 사용 디바이스: {device}")
    print(f"[INFO] AMP 사용: {use_amp}")
    
    # 4. 로깅 설정
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "attack.log")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"로그 파일 생성: {log_file}")
    
    # 5. attack_utils 모듈 임포트
    global attack_utils_module
    try:
        import attack_utils
        attack_utils_module = attack_utils
        print("[INFO] attack_utils 모듈 임포트 성공!")
    except ImportError as e:
        print(f"[ERROR] attack_utils 모듈 임포트 실패: {e}")
        attack_utils_module = None
    
    return base_path, device, use_amp, logger

# ==============================================================================
# 모델 및 데이터 로딩
# ==============================================================================

def load_model(weights_path: Path, num_classes: int = 2, device=None):
    """모델 로드"""
    try:
        from resnet50_model import load_trained_model
        model = load_trained_model(weights_path, device)
        print(f"[INFO] 모델 로드 완료: {weights_path}")
        return model
    except ImportError:
        print("[ERROR] resnet50_model.py를 찾을 수 없습니다.")
        return None

class BrainTumorDataset:
    """뇌종양 데이터셋 클래스 (간소화)"""
    def __init__(self, data_dir, transform=None, use_roi=False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_roi = use_roi
        
        # 데이터 로드
        images_path = self.data_dir / "images.npy"
        labels_path = self.data_dir / "labels.npy"
        
        if images_path.exists() and labels_path.exists():
            self.images = np.load(images_path)
            self.labels = np.load(labels_path)
            print(f"[INFO] Loaded {len(self.images)} images from {data_dir}")
        else:
            raise FileNotFoundError(f"Data files not found in {data_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # 텐서 형태 변환: (H, W, C) -> (C, H, W)
        if len(image.shape) == 3 and image.shape[-1] == 1:
            # (H, W, 1) -> (1, H, W)
            image = image.permute(2, 0, 1)
        elif len(image.shape) == 2:
            # (H, W) -> (1, H, W)
            image = image.unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def build_test_dataset(test_dir: str, use_roi: bool = False):
    """테스트 데이터셋 구성"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    dataset = BrainTumorDataset(test_dir, transform=transform, use_roi=use_roi)
    return dataset

# ==============================================================================
# ZOO 공격 핵심 함수들
# ==============================================================================

def to_norm(x):
    """정규화 함수 (integrated.py와 동일하게 정규화 없음)"""
    return x

def predict_logits(model, x_pix, model_hw=(224,224)):
    """모델 예측"""
    with torch.no_grad():
        return model(x_pix)

def hinge_untargeted_from_logits(logits, src_idx_tensor, kappa: float = 0.0):
    """Untargeted hinge loss"""
    src_logit = logits[0, src_idx_tensor.item()]
    max_other = torch.cat([logits[0, :src_idx_tensor], logits[0, src_idx_tensor+1:]]).max()
    return torch.clamp(max_other - src_logit + kappa, min=0.0)

def zoo_total_loss_untargeted(model, x_pix, x0_pix, src_idx_tensor, c, kappa):
    """ZOO 총 손실 함수"""
    logits = predict_logits(model, x_pix)
    hinge_loss = hinge_untargeted_from_logits(logits, src_idx_tensor, kappa)
    l2_dist = torch.norm((x_pix - x0_pix).view(1, -1), p=2)
    return c * hinge_loss + l2_dist

def adam_update(m, g, mt, vt, idx, lr, beta1, beta2, epoch, up, down, project=True):
    """Adam 옵티마이저 업데이트 (gradient 호환 버전)"""
    # in-place 연산 대신 새로운 텐서 생성
    mt_new = mt.clone()
    vt_new = vt.clone()
    m_new = m.clone()
    
    mt_new[idx] = beta1 * mt[idx] + (1 - beta1) * g[idx]
    vt_new[idx] = beta2 * vt[idx] + (1 - beta2) * (g[idx] ** 2)
    mhat = mt_new[idx] / (1 - beta1 ** epoch)
    vhat = vt_new[idx] / (1 - beta2 ** epoch)
    m_new[idx] = m[idx] - lr * mhat / (torch.sqrt(vhat) + 1e-8)
    
    if project:
        m_new[idx] = torch.clamp(m_new[idx], up[idx], down[idx])
    
    # 원본 텐서 업데이트
    m.data = m_new.data
    mt.data = mt_new.data
    vt.data = vt_new.data

def zoo_attack_l2_adam_untargeted(
    model, x0_pix, src_idx_tensor, roi_mask=None, 
    max_iterations=320, batch_coords=256, step_eps=0.1, 
    lr=2e-3, initial_const_c=6.0, kappa=0.0, 
    beta1=0.9, beta2=0.999, print_every=50
):
    """ZOO L2 Adam Untargeted 공격 (핵심 함수)"""
    device = x0_pix.device
    x0_pix = x0_pix.clone().detach()
    
    # 초기화
    x_adv = x0_pix.clone().detach()
    x_adv.requires_grad_(True)
    
    # Adam 상태
    mt = torch.zeros_like(x_adv)
    vt = torch.zeros_like(x_adv)
    
    # 경계 설정
    up = torch.clamp(x0_pix + step_eps, 0.0, 1.0)
    down = torch.clamp(x0_pix - step_eps, 0.0, 1.0)
    
    # ROI 마스크 적용
    if roi_mask is not None:
        mask = roi_mask.unsqueeze(0).unsqueeze(0)
        up = up * mask + x0_pix * (1 - mask)
        down = down * mask + x0_pix * (1 - mask)
    
    c = initial_const_c
    best_img = x0_pix.clone()
    best_c = c
    n_queries = 0
    
    for iteration in range(max_iterations):
        # 현재 손실 계산
        loss = zoo_total_loss_untargeted(model, x_adv, x0_pix, src_idx_tensor, c, kappa)
        
        # 그래디언트 추정
        grad_est = torch.zeros_like(x_adv)
        for _ in range(batch_coords):
            # 랜덤 방향 생성
            u = torch.randn_like(x_adv)
            u = u / torch.norm(u.view(1, -1), p=2)
            
            # 유한 차분으로 그래디언트 추정
            eps = 1e-3
            loss_plus = zoo_total_loss_untargeted(model, x_adv + eps * u, x0_pix, src_idx_tensor, c, kappa)
            loss_minus = zoo_total_loss_untargeted(model, x_adv - eps * u, x0_pix, src_idx_tensor, c, kappa)
            
            grad_est += (loss_plus - loss_minus) / (2 * eps) * u
            n_queries += 2
        
        grad_est = grad_est / batch_coords
        
        # Adam 업데이트
        adam_update(x_adv, grad_est, mt, vt, slice(None), lr, beta1, beta2, iteration + 1, up, down)
        
        # 성공 확인
        with torch.no_grad():
            pred = predict_logits(model, x_adv).argmax(1)
            if pred.item() != src_idx_tensor.item():
                best_img = x_adv.clone()
                best_c = c
                break
        
        # 출력
        if iteration % print_every == 0:
            print(f"[{iteration:3d}] Loss: {loss.item():.4f}, Pred: {pred.item()}, Target: {src_idx_tensor.item()}")
    
    # 최종 결과
    elapsed = time.time()
    final_margin = hinge_untargeted_from_logits(predict_logits(model, best_img), src_idx_tensor, kappa)
    
    return best_img, n_queries, elapsed, float(final_margin), float(best_c)

# ==============================================================================
# 통합 공격 실행 함수
# ==============================================================================

def run_zoo_attack_unified(
    model, test_loader, device, idx_to_class, 
    max_samples=100, max_iterations=320, batch_coords=256, 
    step_eps=0.1, lr=2e-3, initial_const_c=6.0, 
    print_every=50, save_dir="zoo_outputs", use_roi=False
):
    """통합 ZOO 공격 실행 (ROI + Full 통합)"""
    global attack_utils_module
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 전체 데이터를 리스트로 변환
    all_data = []
    for batch_data in test_loader:
        if len(batch_data) == 3:  # ROI 마스크 포함
            x_norm, y_true, roi_mask = batch_data
            for i in range(x_norm.size(0)):
                all_data.append((x_norm[i:i+1], y_true[i:i+1], roi_mask[i:i+1]))
        else:  # ROI 마스크 없음
            x_norm, y_true = batch_data[:2]
            for i in range(x_norm.size(0)):
                all_data.append((x_norm[i:i+1], y_true[i:i+1], None))
    
    # 랜덤 샘플링
    n_total = len(all_data)
    n_samples = min(max_samples, n_total)
    np.random.seed(42)
    sampled_indices = np.random.choice(n_total, size=n_samples, replace=False)
    sampled_data = [all_data[i] for i in sampled_indices]
    
    attack_type = "ROI" if use_roi else "Full"
    print(f"\n===== ZOO {attack_type} Attack (eps={step_eps}, n_iters={max_iterations}) =====")
    print(f"공격 대상: {n_samples}개 샘플 (랜덤 샘플링)\n")
    
    results = []
    success_count = 0
    total_queries = 0
    total_time = 0
    
    for idx, (x_norm, y_true, roi_mask) in enumerate(tqdm(sampled_data, desc="공격 진행")):
        # 데이터 준비
        x_pix_clean = attack_utils_module.to_pixel(x_norm.to(device))
        roi_mask_device = roi_mask.to(device) if (use_roi and roi_mask is not None) else None
        
        # 원본 예측
        with torch.no_grad():
            pred_orig = predict_logits(model, x_pix_clean).argmax(1).item()
        
        # 공격 실행
        start_time = time.time()
        x_adv, n_queries, elapsed, final_margin, best_c = zoo_attack_l2_adam_untargeted(
            model=model,
            x0_pix=x_pix_clean,
            src_idx_tensor=y_true.to(device),
            roi_mask=roi_mask_device,
            max_iterations=max_iterations,
            batch_coords=batch_coords,
            step_eps=step_eps,
            lr=lr,
            initial_const_c=initial_const_c,
            print_every=print_every
        )
        
        # 결과 확인
        with torch.no_grad():
            pred_adv = predict_logits(model, x_adv).argmax(1).item()
            success = (pred_adv != pred_orig)
        
        if success:
            success_count += 1
        
        # 메트릭 계산
        if attack_utils_module is not None:
            metrics = attack_utils_module.calculate_attack_metrics(
                x_pix_clean, x_adv, y_true.item(), model, threshold=1.0
            )
        else:
            metrics = {'l2_255': 0, 'changed_ratio': 0, 'margin_drop': 0, 'conf_drop': 0}
        
        # 결과 저장
        result = {
            'sample_idx': idx,
            'original_pred': pred_orig,
            'adversarial_pred': pred_adv,
            'success': success,
            'queries': n_queries,
            'elapsed_time': elapsed,
            'l2_255': metrics['l2_255'],
            'changed_ratio': metrics['changed_ratio'],
            'margin_drop': metrics['margin_drop'],
            'conf_drop': metrics['conf_drop'],
            'epsilon': step_eps,
            'use_roi': use_roi
        }
        results.append(result)
        
        total_queries += n_queries
        total_time += elapsed
        
        # 진행 상황 출력
        status = "✅" if success else "❌"
        print(f"[{idx+1:3d}/{n_samples}] {status} {idx_to_class[pred_orig]} → {idx_to_class[pred_adv]} | "
              f"L2={metrics['l2_255']:.2f} | Δconf={metrics['conf_drop']:.3f} | {elapsed:.2f}s")
    
    # 통계 계산
    statistics = {
        'ASR': float(success_count / n_samples * 100.0),
        'success_count': success_count,
        'total_samples': n_samples,
        'total_queries': total_queries,
        'total_time': total_time,
        'mean_queries': float(total_queries / n_samples),
        'mean_time': float(total_time / n_samples),
        'epsilon': step_eps,
        'use_roi': use_roi
    }
    
    return results, statistics

# ==============================================================================
# 메인 실행 함수
# ==============================================================================

def main():
    """메인 실행 함수"""
    try:
        print("\n" + "="*70)
        print("ZOO Attack 실행 (간소화)")
        print("="*70)
        
        # 환경 설정
        base_path, device, use_amp, logger = setup_zoo_environment()
        
        # 모델 로드
        model_path = base_path / "models" / "resnet50_binary_best.pth"
        model = load_model(model_path, num_classes=2, device=device)
        if model is None:
            print("[ERROR] 모델 로드 실패")
            return
        
        # 데이터셋 로드
        test_dir = base_path / "processed_data_np224" / "Testing"
        test_dataset = build_test_dataset(str(test_dir), use_roi=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # 클래스 매핑
        idx_to_class = {0: 'notumor', 1: 'tumor'}
        
        # 전체 이미지 공격 실행
        print(f"\n[INFO] 전체 이미지 공격 실행")
        results_full, statistics_full = run_zoo_attack_unified(
            model, test_loader, device, idx_to_class,
            max_samples=100,
            max_iterations=320,
            batch_coords=256,
            step_eps=0.1,
            lr=2e-3,
            initial_const_c=6.0,
            print_every=50,
            save_dir="zoo_results",
            use_roi=False
        )
        
        # ROI 공격 실행
        print(f"\n[INFO] ROI 공격 실행")
        results_roi, statistics_roi = run_zoo_attack_unified(
            model, test_loader, device, idx_to_class,
            max_samples=100,
            max_iterations=320,
            batch_coords=256,
            step_eps=0.1,
            lr=2e-3,
            initial_const_c=6.0,
            print_every=50,
            save_dir="zoo_roi_results",
            use_roi=True
        )
        
        # 결과 출력
        print(f"\n===== ZOO 공격 결과 요약 =====")
        print(f"전체 이미지 공격 ASR: {statistics_full['ASR']:.2f}%")
        print(f"ROI 공격 ASR: {statistics_roi['ASR']:.2f}%")
        
        # 결과 저장
        if attack_utils_module is not None:
            # CSV 저장
            attack_utils_module.save_results_to_csv(results_full, "zoo_results/zoo_full_results.csv")
            attack_utils_module.save_results_to_csv(results_roi, "zoo_roi_results/zoo_roi_results.csv")
            
            # JSON 저장
            attack_utils_module.save_results_to_json(statistics_full, "zoo_results/zoo_full_statistics.json")
            attack_utils_module.save_results_to_json(statistics_roi, "zoo_roi_results/zoo_roi_statistics.json")
            
            # 시각화
            attack_utils_module.create_result_visualization(results_full, out_dir="zoo_results", attack_name="ZOO Full")
            attack_utils_module.create_result_visualization(results_roi, out_dir="zoo_roi_results", attack_name="ZOO ROI")
        
        print("[INFO] ZOO 공격 완료!")
        
    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
