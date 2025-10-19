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

import os, time, json
from typing import Optional, Tuple, List, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# resnet50_model.py에서 ResNet50 모델 로드
from resnet50_model import load_trained_model, get_model_info

# 공통 유틸리티 import
from adv_utils import (
    collect_first_n_correct, 
    summarize_zoo_metrics, 
    visualize_zoo_triplet, 
    stats_table_like, 
    save_side_by_side
)

import warnings
warnings.filterwarnings('ignore')

# 결과/모델 폴더 안전하게 생성
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------------------
# Utils & Normalization
# ---------------------------
def seed_everything(seed: int = 42):
    """모든 랜덤 시드를 고정하는 함수"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ImageNet 정규화 상수들
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def to_pixel(x_norm):
    """정규화된 텐서를 픽셀 값 [0,1]로 변환"""
    return torch.clamp(x_norm * IMAGENET_STD.to(x_norm.device) + IMAGENET_MEAN.to(x_norm.device), 0.0, 1.0)

def to_norm(x_pix):
    """픽셀 값 [0,1]을 정규화된 텐서로 변환"""
    return (torch.clamp(x_pix, 0.0, 1.0) - IMAGENET_MEAN.to(x_pix.device)) / IMAGENET_STD.to(x_pix.device)

# ---------------------------
# Model / Data
# ---------------------------
def load_model(weights_path: Path, num_classes: int = 2, device=None):
    """사전 훈련된 모델을 로드하는 함수 - resnet50_model.py와 통일"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # resnet50_model.py의 load_trained_model 함수 사용
    model = load_trained_model(str(weights_path), device=device)
    
    # 모델 정보 가져오기
    model_info = get_model_info(str(weights_path))
    class_names = model_info.get('class_names', ["notumor", "tumor"])
    
    # class_names를 class_to_idx 형태로 변환
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    return model, class_to_idx, idx_to_class

class BrainTumorDataset(torch.utils.data.Dataset):
    """
    뇌종양 이미지 데이터셋 (ROI 기능 포함)
    - images.npy : (N,224,224,1) [0,1]
    - labels.npy : (N,) int64
    - class_names.txt : 클래스명
    - roi_masks.npy : (N,224,224,1) {0,1} (옵션, ROI 마스크)
    - meta.json  : (옵션) 메타데이터
    """
    def __init__(self, data_dir, transform=None, use_roi=False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_roi = use_roi

        # 이미지, 라벨 로드
        self.images = np.load(self.data_dir / "images.npy")   # (N,224,224,1)
        self.labels = np.load(self.data_dir / "labels.npy")   # (N,)

        # ROI 마스크 로드 (있으면)
        roi_path = self.data_dir / "roi_masks.npy"
        self.roi_masks = None
        if roi_path.exists():
            self.roi_masks = np.load(roi_path)   # (N,224,224,1)
            print(f"[INFO] ROI 마스크 로드: {roi_path}")
        elif use_roi:
            print(f"[WARN] ROI 마스크 파일이 없습니다: {roi_path}")

        # 클래스명
        with open(self.data_dir / "class_names.txt", "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f]

        # 메타데이터 (있으면)
        meta_path = self.data_dir / "meta.json"
        self.meta = None
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

        print(f"[INFO] Loaded {len(self.images)} images from {self.data_dir}")
        print(f"[INFO] Classes: {self.class_names}")
        print(f"[INFO] ROI 사용: {use_roi and self.roi_masks is not None}")
        if self.meta is not None:
            print(f"[INFO] Meta: {self.meta}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1채널 → 3채널 복제
        img = self.images[idx]              # (224,224,1)
        img = np.repeat(img, 3, axis=2)    # (224,224,3)

        label = int(self.labels[idx])

        # ROI 마스크 처리
        roi_mask = None
        if self.use_roi and self.roi_masks is not None:
            roi_mask = self.roi_masks[idx]  # (224,224,1)
            roi_mask = np.repeat(roi_mask, 3, axis=2)  # (224,224,3)

        if self.transform:
            img = self.transform(img)      # (3,224,224)
            if roi_mask is not None:
                roi_mask = self.transform(roi_mask)  # (3,224,224)

        if roi_mask is not None:
            return img, label, roi_mask
        else:
            return img, label

def load_brain_tumor_data(train_dir="processed_data_np224/Training", 
                         test_dir="processed_data_np224/Testing",
                         batch_size=32, use_roi=False):
    """
    뇌종양 데이터셋 로드 (ROI 기능 포함)
    
    Args:
        train_dir: 훈련 데이터 경로
        test_dir: 테스트 데이터 경로
        batch_size: 배치 크기
        use_roi: ROI 마스크 사용 여부
    
    Returns:
        train_loader, test_loader, class_names
    """
    print("[INFO] Loading brain tumor dataset...")
    
    # 데이터 증강 (훈련용)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # 정규화는 모델 내부에 포함됨
    ])
    
    # 테스트용 (증강 없음)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # 정규화는 모델 내부에 포함됨
    ])
    
    # BrainTumorDataset 사용
    train_dataset = BrainTumorDataset(train_dir, transform=train_transform, use_roi=use_roi)
    test_dataset = BrainTumorDataset(test_dir, transform=test_transform, use_roi=use_roi)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"[INFO] Train images: {len(train_dataset)}")
    print(f"[INFO] Test images: {len(test_dataset)}")
    print(f"[INFO] Classes: {train_dataset.class_names}")
    
    return train_loader, test_loader, train_dataset.class_names

def build_test_dataset(test_dir: str, use_roi: bool = False):
    """테스트 데이터셋을 구성하는 함수 (ROI 기능 포함)"""
    # 테스트용 변환 (증강 없음)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # 정규화는 모델 내부에 포함됨
    ])
    
    return BrainTumorDataset(test_dir, transform=test_transform, use_roi=use_roi)

def fetch_sample(ds, index: int, device):
    """데이터셋에서 특정 인덱스의 샘플을 가져오는 함수 (ROI 기능 포함)"""
    data = ds[index]
    if len(data) == 3:  # ROI 마스크 포함
        x_norm, y, roi_mask = data
        roi_mask = roi_mask.unsqueeze(0).to(device) if roi_mask is not None else None
    else:  # ROI 마스크 없음
        x_norm, y = data
        roi_mask = None
    
    path = f"sample_{index}"
    return x_norm.unsqueeze(0).to(device), int(y), path, roi_mask

@torch.no_grad()
def predict_logits(model, x_pix, model_hw=(224,224)):
    """모델의 로짓을 예측하는 함수"""
    # 필요시 크기 조정
    xin = x_pix if x_pix.shape[-2:] == model_hw else F.interpolate(x_pix, size=model_hw, mode="bilinear", align_corners=False)
    return model(to_norm(xin))

# ---------------------------
# Losses (hinge + L2), Untargeted
# ---------------------------
@torch.no_grad()
def hinge_untargeted_from_logits(logits, src_idx_tensor, kappa: float = 0.0):
    """
    Untargeted 공격을 위한 hinge loss 계산
    margin = z_y - max_{j≠y} z_j
    목표: margin <= -kappa (즉 원래 클래스가 지도록)
    """
    B, C = logits.shape
    ar = torch.arange(B, device=logits.device)
    sel = logits[ar, src_idx_tensor]  # 현재 클래스의 로짓
    
    # 현재 클래스를 제외한 최대 로짓 찾기
    tmp = logits.clone()
    tmp[ar, src_idx_tensor] = -1e9
    max_other = tmp.max(dim=1).values
    
    margin = sel - max_other
    hinge = torch.clamp(margin + kappa, min=0.0)
    return hinge, margin

@torch.no_grad()
def zoo_total_loss_untargeted(model, x_pix, x0_pix, src_idx_tensor, c, kappa):
    """ZOO 공격의 총 손실함수 계산 (untargeted)"""
    logits = predict_logits(model, x_pix)
    hinge, margin = hinge_untargeted_from_logits(logits, src_idx_tensor, kappa)
    l2 = ((x_pix - x0_pix)**2).flatten(1).sum(1)
    total = l2 + c * hinge
    return total, hinge, l2, logits, margin

# ---------------------------
# Adam coordinate update
# ---------------------------
def adam_update(m, g, mt, vt, idx, lr, beta1, beta2, epoch, up, down, project=True):
    """Adam 옵티마이저를 사용한 좌표 업데이트"""
    # Adam 모멘텀 업데이트
    mt[idx] = beta1 * mt[idx] + (1 - beta1) * g
    vt[idx] = beta2 * vt[idx] + (1 - beta2) * (g * g)
    
    # 바이어스 보정
    corr = np.sqrt(1 - beta2**epoch[idx]) / (1 - beta1**epoch[idx])
    
    # 파라미터 업데이트
    m[idx] -= lr * corr * (mt[idx] / (np.sqrt(vt[idx]) + 1e-8))
    
    # 경계 제약 적용
    if project:
        m[idx] = np.minimum(np.maximum(m[idx], down[idx]), up[idx])
    
    epoch[idx] += 1

# ---------------------------
# ZOO L2 (ROI 기능 포함)
# ---------------------------
def zoo_attack_l2_adam_untargeted(
    model,
    x0_pix,                 # (1,3,H,W) in [0,1] - 원본 이미지
    src_idx_tensor,         # (1,) 현재 예측(원본의 source class)
    roi_mask=None,          # (1,3,H,W) ROI 마스크 (0 또는 1)
    max_iterations=320,     # 최대 반복 횟수
    batch_coords=256,       # 배치당 좌표 수
    step_eps=0.1,           # 유한 차분을 위한 스텝 크기 (입실론 통일)
    lr=2e-3,               # Adam 학습률
    beta1=0.9, beta2=0.999, # Adam 모멘텀 파라미터
    kappa=0.0,             # 마진 파라미터
    initial_const_c=6.0,    # 초기 c 값
    binary_search_steps=3,  # 이진 탐색 단계
    print_every=100,        # 로그 출력 주기
    early_stop_window=200,  # 조기 종료 윈도우
    abort_early=True,       # 조기 종료 여부
    early_success=True      # 초기 오분류 시 무교란 성공 처리
):
    """ZOO L2 공격 (Adam 옵티마이저, Untargeted, ROI 기능 포함)"""
    device = x0_pix.device
    _, C, H, W = x0_pix.shape
    N = C * H * W  # 총 픽셀 수
    h = step_eps   # 유한 차분 스텝 (입실론 0.1로 통일)

    # 초기 오분류 체크 (무교란 성공 처리)
    if early_success:
        with torch.no_grad():
            pred_orig = predict_logits(model, x0_pix).argmax(1).item()
            if pred_orig != int(src_idx_tensor.item()):
                # 이미 오분류된 경우 무교란 성공으로 처리
                return x0_pix.clone(), 0, 0.0, True, 0.0, 0.0

    # 경계 조건: x0 + m ∈ [0,1]
    x0_np = x0_pix.detach().cpu().numpy().reshape(-1)
    up   = (1.0 - x0_np).astype(np.float32)  # 상한
    down = (-x0_np).astype(np.float32)       # 하한

    # ROI 마스크 처리
    roi_indices = None
    if roi_mask is not None:
        # ROI 영역 내의 픽셀 인덱스만 선택
        roi_flat = roi_mask.detach().cpu().numpy().reshape(-1)
        roi_indices = np.where(roi_flat > 0.5)[0]  # ROI 영역 (값이 0.5보다 큰 픽셀)
        print(f"[INFO] ROI 제약: {len(roi_indices)}/{N} 픽셀 ({len(roi_indices)/N*100:.1f}%)")
        
        if len(roi_indices) == 0:
            print("[WARN] ROI 영역이 비어있습니다. 전체 이미지를 사용합니다.")
            roi_indices = None

    # Adam 옵티마이저 버퍼 초기화
    mt = np.zeros(N, dtype=np.float32)    # 1차 모멘텀
    vt = np.zeros(N, dtype=np.float32)    # 2차 모멘텀
    epoch = np.ones(N, dtype=np.int32)    # 에포크 카운터

    n_queries = 0  # 쿼리 카운터
    t0 = time.time()

    # 최적 결과 저장용 변수들
    best_img   = x0_pix.clone()
    best_l2    = float("inf")
    best_c     = float(initial_const_c)

    # 이진 탐색을 위한 경계값 (더 넓은 범위)
    lower_c, upper_c = 0.001, 1e6  # 더 넓은 범위
    CONST = float(initial_const_c)

    @torch.no_grad()
    def eval_batch(X_list):
        """배치 평가 함수"""
        nonlocal n_queries
        X = torch.cat(X_list, dim=0).to(device)
        tot, hin, l2, logits, margin = zoo_total_loss_untargeted(
            model, X, x0_pix, src_idx_tensor.expand(X.size(0)), CONST, kappa
        )
        n_queries += X.size(0)
        return tot.cpu(), hin.cpu(), l2.cpu(), logits.cpu(), margin.cpu()

    # 이진 탐색 루프
    for bs in range(int(binary_search_steps)):
        # 변수 초기화 (작은 랜덤 노이즈로 시작)
        m = np.random.normal(0, 0.01, N).astype(np.float32)  # 작은 랜덤 초기화
        mt.fill(0.0)
        vt.fill(0.0)
        epoch.fill(1)
        
        prev_total = 1e9
        no_improve = 0
        success_this_c = False
        best_l2_this_c = float("inf")

        # 반복 최적화 루프
        for it in range(int(max_iterations)):
            # 현재 이미지 생성
            m_tensor = torch.from_numpy(m.reshape(1,C,H,W)).to(device)
            x_base = torch.clamp(x0_pix + m_tensor, 0.0, 1.0)

            # 좌표 선택 (ROI 제약 적용)
            if roi_indices is not None:
                # ROI 영역 내에서만 선택
                available_coords = min(batch_coords, len(roi_indices))
                idx = np.random.choice(roi_indices, size=available_coords, replace=False)
            else:
                # 전체 영역에서 선택
                idx = np.random.choice(N, size=min(batch_coords, N), replace=False)

            # 유한 차분을 위한 배치 구성
            X_list = [x_base]  # 현재 이미지
            
            for j in idx:
                # 양의 방향 섭동
                m_p = m.copy()
                m_p[j] = np.minimum(m_p[j] + h, up[j])
                m_p_tensor = torch.from_numpy(m_p.reshape(1,C,H,W)).to(device)
                X_list.append(torch.clamp(x0_pix + m_p_tensor, 0.0, 1.0))
                
                # 음의 방향 섭동
                m_m = m.copy()
                m_m[j] = np.maximum(m_m[j] - h, down[j])
                m_m_tensor = torch.from_numpy(m_m.reshape(1,C,H,W)).to(device)
                X_list.append(torch.clamp(x0_pix + m_m_tensor, 0.0, 1.0))

            # 배치 평가
            total, hinge, l2, logits, margin = eval_batch(X_list)
            L0 = total[0].item()

            # 그래디언트 계산 (유한 차분)
            plus  = total[1::2].numpy()  # 양의 방향 손실
            minus = total[2::2].numpy()  # 음의 방향 손실
            grad  = (plus - minus) / (2.0 * h)

            # Adam 업데이트
            adam_update(m, grad, mt, vt, idx, lr, beta1, beta2, epoch, up, down, project=True)

            # 진행 상황 로깅 (상세 정보 포함)
            if (it % max(1, print_every)) == 0:
                roi_info = f"ROI:{len(roi_indices)}" if roi_indices is not None else "Full"
                print(f"[ZOO-Adam-Untgt][BS{bs}][{it:4d}/{max_iterations}] c={CONST:.3g}  total={L0:.5g}  hinge={hinge[0].item():.5g}  l2={l2[0].item():.5g}  margin={margin[0].item():.5g}  K={len(idx)} {roi_info}")

            # 조기 종료 체크 (완화된 조건)
            if L0 > prev_total * 0.999:
                no_improve += 1
                if abort_early and (no_improve >= max(1, int(early_stop_window))):
                    print("  Early stopping (no improvement).")
                    break
            else:
                no_improve = 0
                prev_total = L0

            # 성공 판정: margin <= 0.1 (완화된 성공 기준)
            if margin[0].item() <= 0.1:
                success_this_c = True
                l2_now = l2[0].item()
                
                if l2_now < best_l2_this_c:
                    best_l2_this_c = l2_now
                
                if l2_now < best_l2:
                    best_l2 = l2_now
                    m_final = torch.from_numpy(m.reshape(1,C,H,W)).to(device)
                    best_img = torch.clamp(x0_pix + m_final, 0.0, 1.0).detach().clone()
                    best_c = CONST

        # c 값 이진 탐색 업데이트 (더 공격적인 전략)
        if success_this_c:
            upper_c = min(upper_c, CONST)
            CONST = (lower_c + upper_c)/2 if upper_c < 1e6 else CONST/2
            print(f"  [BS{bs}] 성공! c={CONST:.3g} (범위: {lower_c:.3g}~{upper_c:.3g})")
        else:
            lower_c = max(lower_c, CONST)
            CONST = (lower_c + upper_c)/2 if upper_c < 1e6 else CONST*5  # 더 작은 배수
            print(f"  [BS{bs}] 실패. c={CONST:.3g} (범위: {lower_c:.3g}~{upper_c:.3g})")

    elapsed = time.time() - t0
    
    # 최종 마진 계산
    with torch.no_grad():
        lg = predict_logits(model, best_img)[0]
        s_idx = int(src_idx_tensor.item())
        slog = lg[s_idx].item()
        tmp = lg.clone()
        tmp[s_idx] = -1e9
        final_margin = slog - tmp.max().item()
    
    return best_img, n_queries, elapsed, float(final_margin), float(best_c)

# ---------------------------
# Batch Attack Functions (랜덤 100장 샘플링, ROI 기능 포함)
# ---------------------------
def batch_zoo_attack(model, test_loader, device, idx_to_class, 
                    max_samples=100, max_iterations=320, batch_coords=256, 
                    step_eps=0.1, lr=2e-3, initial_const_c=6.0, 
                    print_every=50, save_dir="zoo_outputs_untargeted", use_roi=False):
    """100장 이미지에 대한 배치 ZOO 공격 (랜덤 샘플링, ROI 기능 포함)"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 전체 데이터를 리스트로 변환
    all_data = []
    for batch_data in test_loader:
        if len(batch_data) == 3 and use_roi:  # ROI 마스크 포함
            x_norm, y_true, roi_mask = batch_data
            # 배치에서 개별 샘플 추출
            for i in range(x_norm.size(0)):
                all_data.append((x_norm[i:i+1], y_true[i:i+1], roi_mask[i:i+1]))
        else:  # ROI 마스크 없음
            x_norm, y_true = batch_data[:2]  # ROI가 있어도 처음 2개만 사용
            # 배치에서 개별 샘플 추출
            for i in range(x_norm.size(0)):
                all_data.append((x_norm[i:i+1], y_true[i:i+1], None))
    
    # 랜덤 샘플링 (100장 또는 전체 데이터 수 중 작은 값)
    n_total = len(all_data)
    n_samples = min(max_samples, n_total)
    
    # 시드 고정하여 재현 가능한 랜덤 샘플링
    np.random.seed(42)
    sampled_indices = np.random.choice(n_total, size=n_samples, replace=False)
    sampled_data = [all_data[i] for i in sampled_indices]
    
    print(f"[INFO] 전체 {n_total}장 중 랜덤하게 {n_samples}장 샘플링")
    print(f"[INFO] ROI 기능 사용: {use_roi}")
    
    results = []
    success_count = 0
    total_queries = 0
    total_time = 0.0
    
    print(f"[INFO] 배치 공격 시작: 랜덤 {n_samples}장 이미지")
    
    for idx, (x_norm, y_true, roi_mask) in enumerate(tqdm(sampled_data, desc="공격 진행")):
        # 데이터 준비
        x_pix_clean = to_pixel(x_norm.to(device))
        roi_mask_device = roi_mask.to(device) if roi_mask is not None else None
        
        # 원본 예측
        with torch.no_grad():
            pred_orig = predict_logits(model, x_pix_clean).argmax(1).item()
        
        # untargeted 공격: source 클래스를 현재 예측으로 설정
        src_idx_tensor = torch.tensor([int(pred_orig)], device=device)
        
        # ZOO 공격 실행 (입실론 0.1로 통일, ROI 마스크 전달)
        start_time = time.time()
        x_adv_pix, n_queries, elapsed, final_margin, best_c = zoo_attack_l2_adam_untargeted(
            model, x_pix_clean, src_idx_tensor,
            roi_mask=roi_mask_device,  # ROI 마스크 전달
            max_iterations=max_iterations, 
            batch_coords=batch_coords,
            step_eps=step_eps,  # 입실론 0.1로 통일
            lr=5e-3,  # 학습률 증가
            beta1=0.9, 
            beta2=0.999,
            kappa=0.0, 
            initial_const_c=initial_const_c,
            binary_search_steps=5,  # 이진 탐색 단계 증가
            print_every=print_every,
            early_stop_window=max(100, max_iterations//4),  # 조기 종료 윈도우 증가
            abort_early=True,
            early_success=True
        )
        
        # 공격 후 예측
        with torch.no_grad():
            after = predict_logits(model, x_adv_pix).argmax(1).item()
        
        success = (after != pred_orig)
        if success:
            success_count += 1
        
        # 통계 계산 (ΔMargin 포함)
        S = stats_table_like_enhanced(x_pix_clean, x_adv_pix, elapsed, pred_orig, model, count_mode='element', tau_255=1.0)
        
        # 결과 저장 (실제 데이터셋 인덱스 기록)
        original_idx = sampled_indices[idx]
        result = {
            "index": int(original_idx),  # 원본 데이터셋에서의 인덱스
            "sampled_index": int(idx),   # 샘플링된 순서
            "pred_before": int(pred_orig),
            "pred_after": int(after),
            "success": bool(success),
            "queries": int(n_queries),
            "elapsed_sec": float(elapsed),
            "final_margin": float(final_margin),
            "best_c": float(best_c),
            "L2_255": float(S["L2_255"]),
            "changed_pixels": int(S["changed"]),
            "total_pixels": int(S["total"]),
            "changed_ratio": float(S["changed"]) / float(S["total"]) * 100.0,
            "delta_margin": float(S["dmargin"]),
            "conf_drop": float(S["conf_drop"]),
            "epsilon": float(step_eps),  # 입실론 값 기록
            "use_roi": bool(use_roi and roi_mask is not None)  # ROI 사용 여부 기록
        }
        results.append(result)
        
        total_queries += n_queries
        total_time += elapsed
        
        # 개별 이미지 저장 (성공한 경우만)
        if success:
            roi_suffix = "_roi" if use_roi and roi_mask is not None else ""
            stem = f"idx{original_idx:05d}_untargeted_{pred_orig}to{after}{roi_suffix}"
            out_png = Path(save_dir) / f"{stem}.png"
            left_t = f"Original: {idx_to_class.get(pred_orig, str(pred_orig))}"
            right_t = f"ZOO-Adam-Untgt{roi_suffix} (q={n_queries})\nPred: {idx_to_class.get(after, str(after))}\nΔMargin: {S['dmargin']:.3f}\nε: {step_eps}"
            save_side_by_side_with_roi(x_pix_clean, x_adv_pix, roi_mask_device, out_png, left_t, right_t, use_roi=use_roi)
        
        # 진행 상황 출력
        if (idx + 1) % 10 == 0:
            current_success_rate = success_count / (idx + 1) * 100
            avg_queries = total_queries / (idx + 1)
            avg_time = total_time / (idx + 1)
            print(f"[진행상황] {idx+1}/{n_samples} | 성공률: {current_success_rate:.1f}% | 평균 쿼리: {avg_queries:.0f} | 평균 시간: {avg_time:.2f}초")
    
    return results, success_count, total_queries, total_time

# ---------------------------
# Enhanced Metrics & Viz (ROI 기능 포함)
# ---------------------------
@torch.no_grad()
def stats_table_like_enhanced(x_pix_clean, x_adv_pix, elapsed_sec, y_true, model, count_mode='element', tau_255=1.0):
    """향상된 통계 계산 (ΔMargin, ΔConfidence 포함)"""
    delta = (x_adv_pix - x_pix_clean)
    l2_255 = (delta * 255.0).view(1, -1).norm(p=2).item()

    if count_mode == 'element':   # H*W*C 기준
        changed = ((delta.abs() * 255.0) > tau_255).sum().item()
        total   = int(delta.numel())
    else:                         # H*W 기준
        per_spatial = delta.abs().max(dim=1)[0].squeeze(0)
        changed = ((per_spatial * 255.0) > tau_255).sum().item()
        total   = int(per_spatial.numel())

    # ΔMargin 계산
    logits_clean = predict_logits(model, x_pix_clean)
    logits_adv   = predict_logits(model, x_adv_pix)

    y = int(y_true)
    correct_logit_clean = logits_clean[0, y].item()
    max_other_clean     = logits_clean[0, torch.arange(logits_clean.size(1)) != y].max().item()
    margin_clean = correct_logit_clean - max_other_clean

    correct_logit_adv = logits_adv[0, y].item()
    max_other_adv     = logits_adv[0, torch.arange(logits_adv.size(1)) != y].max().item()
    margin_adv = correct_logit_adv - max_other_adv

    delta_margin = margin_clean - margin_adv  # 양수면 공격이 margin을 줄였다는 뜻

    # ΔConfidence 계산
    probs_clean = torch.softmax(logits_clean, dim=1)[0]
    probs_adv   = torch.softmax(logits_adv, dim=1)[0]
    conf_drop = (probs_clean[y] - probs_adv[y]).item()

    return {
        "L2_255": float(l2_255),
        "changed": int(changed),
        "total": int(total),
        "time": float(elapsed_sec),
        "dmargin": float(delta_margin),
        "conf_drop": float(conf_drop)
    }

def save_side_by_side_with_roi(x_clean, x_adv, roi_mask, out_path, left_title, right_title, use_roi=False):
    """ROI 기능을 포함한 side-by-side 저장"""
    import matplotlib.pyplot as plt
    
    # 텐서를 numpy로 변환
    clean_np = x_clean.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    adv_np = x_adv.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    
    # 픽셀 값 클램핑
    clean_np = np.clip(clean_np, 0, 1)
    adv_np = np.clip(adv_np, 0, 1)
    
    # ROI 마스크 처리
    if use_roi and roi_mask is not None:
        roi_np = roi_mask.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        roi_np = np.clip(roi_np, 0, 1)
        
        # 3개 서브플롯
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 원본 이미지
        axes[0].imshow(clean_np)
        axes[0].set_title(left_title, fontsize=10)
        axes[0].axis('off')
        
        # ROI 마스크
        axes[1].imshow(roi_np, cmap='gray')
        axes[1].set_title('ROI Mask', fontsize=10)
        axes[1].axis('off')
        
        # 적대 이미지
        axes[2].imshow(adv_np)
        axes[2].set_title(right_title, fontsize=10)
        axes[2].axis('off')
    else:
        # 2개 서브플롯 (기본)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # 원본 이미지
        axes[0].imshow(clean_np)
        axes[0].set_title(left_title, fontsize=10)
        axes[0].axis('off')
        
        # 적대 이미지
        axes[1].imshow(adv_np)
        axes[1].set_title(right_title, fontsize=10)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

def save_side_by_side(x_clean, x_adv, out_path, left_title, right_title):
    """기본 side-by-side 저장 (하위 호환성)"""
    save_side_by_side_with_roi(x_clean, x_adv, None, out_path, left_title, right_title, use_roi=False)

def calculate_batch_statistics_enhanced(results):
    """향상된 배치 공격 결과 통계 계산 (ΔMargin, ΔConfidence, ROI 기능 포함)"""
    if not results:
        return {}
    
    df = pd.DataFrame(results)
    
    # 기본 통계
    total_samples = len(results)
    success_count = df['success'].sum()
    success_rate = success_count / total_samples * 100
    
    # ROI 사용 여부 통계
    roi_used = df['use_roi'].sum() if 'use_roi' in df.columns else 0
    roi_usage_rate = roi_used / total_samples * 100 if total_samples > 0 else 0
    
    # 쿼리 통계
    avg_queries = df['queries'].mean()
    std_queries = df['queries'].std()
    min_queries = df['queries'].min()
    max_queries = df['queries'].max()
    
    # 시간 통계
    avg_time = df['elapsed_sec'].mean()
    std_time = df['elapsed_sec'].std()
    total_time = df['elapsed_sec'].sum()
    
    # 섭동량 통계 (L2 norm)
    avg_l2 = df['L2_255'].mean()
    std_l2 = df['L2_255'].std()
    min_l2 = df['L2_255'].min()
    max_l2 = df['L2_255'].max()
    
    # 조작된 픽셀 수 통계
    avg_changed_pixels = df['changed_pixels'].mean()
    std_changed_pixels = df['changed_pixels'].std()
    avg_changed_ratio = df['changed_ratio'].mean()
    std_changed_ratio = df['changed_ratio'].std()
    
    # ΔMargin 통계
    avg_delta_margin = df['delta_margin'].mean()
    std_delta_margin = df['delta_margin'].std()
    
    # ΔConfidence 통계
    avg_conf_drop = df['conf_drop'].mean()
    std_conf_drop = df['conf_drop'].std()
    
    # 성공한 샘플만의 통계
    success_df = df[df['success'] == True]
    if len(success_df) > 0:
        success_avg_queries = success_df['queries'].mean()
        success_avg_time = success_df['elapsed_sec'].mean()
        success_avg_l2 = success_df['L2_255'].mean()
        success_avg_changed_ratio = success_df['changed_ratio'].mean()
        success_avg_delta_margin = success_df['delta_margin'].mean()
        success_avg_conf_drop = success_df['conf_drop'].mean()
    else:
        success_avg_queries = 0
        success_avg_time = 0
        success_avg_l2 = 0
        success_avg_changed_ratio = 0
        success_avg_delta_margin = 0
        success_avg_conf_drop = 0
    
    return {
        'total_samples': total_samples,
        'success_count': success_count,
        'success_rate': success_rate,
        'roi_used': roi_used,
        'roi_usage_rate': roi_usage_rate,
        'avg_queries': avg_queries,
        'std_queries': std_queries,
        'min_queries': min_queries,
        'max_queries': max_queries,
        'avg_time': avg_time,
        'std_time': std_time,
        'total_time': total_time,
        'avg_l2': avg_l2,
        'std_l2': std_l2,
        'min_l2': min_l2,
        'max_l2': max_l2,
        'avg_changed_pixels': avg_changed_pixels,
        'std_changed_pixels': std_changed_pixels,
        'avg_changed_ratio': avg_changed_ratio,
        'std_changed_ratio': std_changed_ratio,
        'avg_delta_margin': avg_delta_margin,
        'std_delta_margin': std_delta_margin,
        'avg_conf_drop': avg_conf_drop,
        'std_conf_drop': std_conf_drop,
        'success_avg_queries': success_avg_queries,
        'success_avg_time': success_avg_time,
        'success_avg_l2': success_avg_l2,
        'success_avg_changed_ratio': success_avg_changed_ratio,
        'success_avg_delta_margin': success_avg_delta_margin,
        'success_avg_conf_drop': success_avg_conf_drop,
        'epsilon': 0.1  # 입실론 값 추가
    }

def create_visualization_enhanced(results, save_dir="zoo_outputs_untargeted"):
    """향상된 결과 시각화 생성 (ΔMargin, ΔConfidence, ROI 기능 포함)"""
    if not results:
        print("[WARNING] 시각화할 결과가 없습니다.")
        return
    
    df = pd.DataFrame(results)
    os.makedirs(save_dir, exist_ok=True)
    
    # ROI 사용 여부 확인
    use_roi_stats = 'use_roi' in df.columns and df['use_roi'].any()
    
    # 시각화 생성
    fig_height = 15 if use_roi_stats else 12
    plt.figure(figsize=(18, fig_height))
    
    subplot_rows = 4 if use_roi_stats else 3
    
    # 1. 성공률 파이 차트
    plt.subplot(subplot_rows, 3, 1)
    success_count = df['success'].sum()
    fail_count = len(df) - success_count
    plt.pie([success_count, fail_count], labels=['성공', '실패'], autopct='%1.1f%%', 
            colors=['lightgreen', 'lightcoral'])
    title_suffix = " (ROI 포함)" if use_roi_stats else ""
    plt.title(f'공격 성공률 (ε=0.1, n={len(df)}){title_suffix}')
    
    # 2. 쿼리 수 분포
    plt.subplot(subplot_rows, 3, 2)
    plt.hist(df['queries'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('쿼리 수')
    plt.ylabel('빈도')
    plt.title('쿼리 수 분포')
    plt.axvline(df['queries'].mean(), color='red', linestyle='--', label=f'평균: {df["queries"].mean():.0f}')
    plt.legend()
    
    # 3. 섭동량(L2) 분포
    plt.subplot(subplot_rows, 3, 3)
    plt.hist(df['L2_255'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('L2 섭동량 (0-255)')
    plt.ylabel('빈도')
    plt.title('섭동량 분포')
    plt.axvline(df['L2_255'].mean(), color='red', linestyle='--', label=f'평균: {df["L2_255"].mean():.2f}')
    plt.legend()
    
    # 4. ΔMargin 분포
    plt.subplot(subplot_rows, 3, 4)
    plt.hist(df['delta_margin'], bins=20, alpha=0.7, color='gold', edgecolor='black')
    plt.xlabel('ΔMargin (Confidence Gap Drop)')
    plt.ylabel('빈도')
    plt.title('ΔMargin 분포')
    plt.axvline(df['delta_margin'].mean(), color='red', linestyle='--', label=f'평균: {df["delta_margin"].mean():.3f}')
    plt.legend()
    
    # 5. ΔConfidence 분포
    plt.subplot(subplot_rows, 3, 5)
    plt.hist(df['conf_drop'], bins=20, alpha=0.7, color='salmon', edgecolor='black')
    plt.xlabel('ΔConfidence (Softmax Drop)')
    plt.ylabel('빈도')
    plt.title('ΔConfidence 분포')
    plt.axvline(df['conf_drop'].mean(), color='red', linestyle='--', label=f'평균: {df["conf_drop"].mean():.3f}')
    plt.legend()
    
    # 6. 조작된 픽셀 비율 분포
    plt.subplot(subplot_rows, 3, 6)
    plt.hist(df['changed_ratio'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('조작된 픽셀 비율 (%)')
    plt.ylabel('빈도')
    plt.title('조작된 픽셀 비율 분포')
    plt.axvline(df['changed_ratio'].mean(), color='red', linestyle='--', label=f'평균: {df["changed_ratio"].mean():.2f}%')
    plt.legend()
    
    # 7. 공격 시간 분포
    plt.subplot(subplot_rows, 3, 7)
    plt.hist(df['elapsed_sec'], bins=20, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('공격 시간 (초)')
    plt.ylabel('빈도')
    plt.title('공격 시간 분포')
    plt.axvline(df['elapsed_sec'].mean(), color='red', linestyle='--', label=f'평균: {df["elapsed_sec"].mean():.2f}초')
    plt.legend()
    
    # 8. L2 vs ΔMargin 상관관계
    plt.subplot(subplot_rows, 3, 8)
    success_df = df[df['success'] == True]
    if len(success_df) > 0:
        plt.scatter(success_df['L2_255'], success_df['delta_margin'], alpha=0.6, color='green')
        plt.xlabel('L2 섭동량')
        plt.ylabel('ΔMargin')
        plt.title('L2 vs ΔMargin 상관관계 (성공 케이스)')
        
        # 상관계수 계산
        corr = success_df['L2_255'].corr(success_df['delta_margin'])
        plt.text(0.05, 0.95, f'상관계수: {corr:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        plt.text(0.5, 0.5, '성공 케이스 없음', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('L2 vs ΔMargin 상관관계')
    
    # 9. 성공/실패별 ΔMargin 비교
    plt.subplot(subplot_rows, 3, 9)
    success_margins = df[df['success'] == True]['delta_margin']
    fail_margins = df[df['success'] == False]['delta_margin']
    
    if len(success_margins) > 0 and len(fail_margins) > 0:
        plt.boxplot([success_margins, fail_margins], labels=['성공', '실패'])
        plt.ylabel('ΔMargin')
        plt.title('성공/실패별 ΔMargin 비교')
    else:
        plt.text(0.5, 0.5, '데이터 부족', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('성공/실패별 ΔMargin 비교')
    
    # ROI 관련 시각화 (ROI 사용 시에만)
    if use_roi_stats:
        # 10. ROI 사용률
        plt.subplot(subplot_rows, 3, 10)
        roi_used = df['use_roi'].sum()
        roi_not_used = len(df) - roi_used
        plt.pie([roi_used, roi_not_used], labels=['ROI 사용', 'ROI 미사용'], autopct='%1.1f%%',
                colors=['lightblue', 'lightgray'])
        plt.title('ROI 사용률')
        
        # 11. ROI 사용 vs 성공률
        plt.subplot(subplot_rows, 3, 11)
        roi_success = df[df['use_roi'] == True]['success'].mean() * 100
        no_roi_success = df[df['use_roi'] == False]['success'].mean() * 100
        plt.bar(['ROI 사용', 'ROI 미사용'], [roi_success, no_roi_success], 
                color=['lightblue', 'lightgray'], alpha=0.7)
        plt.ylabel('성공률 (%)')
        plt.title('ROI 사용 여부별 성공률')
        
        # 12. ROI 사용 vs L2 섭동량
        plt.subplot(subplot_rows, 3, 12)
        roi_l2 = df[df['use_roi'] == True]['L2_255']
        no_roi_l2 = df[df['use_roi'] == False]['L2_255']
        if len(roi_l2) > 0 and len(no_roi_l2) > 0:
            plt.boxplot([roi_l2, no_roi_l2], labels=['ROI 사용', 'ROI 미사용'])
            plt.ylabel('L2 섭동량')
            plt.title('ROI 사용 여부별 섭동량 비교')
        else:
            plt.text(0.5, 0.5, 'ROI 데이터 부족', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('ROI 사용 여부별 섭동량 비교')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'batch_attack_visualization_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 향상된 통계 요약 텍스트 파일 저장
    stats = calculate_batch_statistics_enhanced(results)
    
    with open(Path(save_dir) / 'batch_statistics_summary_enhanced.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ZOO 배치 공격 결과 요약 (향상된 지표 + ROI 기능 포함, ε=0.1)\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"총 샘플 수: {stats['total_samples']} (랜덤 샘플링)\n")
        f.write(f"입실론 값 (step_eps): {stats['epsilon']}\n")
        f.write(f"성공한 공격: {stats['success_count']}\n")
        f.write(f"성공률: {stats['success_rate']:.2f}%\n")
        f.write(f"ROI 사용 샘플: {stats['roi_used']}\n")
        f.write(f"ROI 사용률: {stats['roi_usage_rate']:.2f}%\n\n")
        
        f.write("쿼리 통계:\n")
        f.write(f"  평균: {stats['avg_queries']:.2f} ± {stats['std_queries']:.2f}\n")
        f.write(f"  범위: {stats['min_queries']} ~ {stats['max_queries']}\n")
        f.write(f"  성공한 샘플 평균: {stats['success_avg_queries']:.2f}\n\n")
        
        f.write("시간 통계:\n")
        f.write(f"  평균: {stats['avg_time']:.2f} ± {stats['std_time']:.2f}초\n")
        f.write(f"  총 시간: {stats['total_time']:.2f}초\n")
        f.write(f"  성공한 샘플 평균: {stats['success_avg_time']:.2f}초\n\n")
        
        f.write("섭동량 통계 (L2 norm):\n")
        f.write(f"  평균: {stats['avg_l2']:.2f} ± {stats['std_l2']:.2f}\n")
        f.write(f"  범위: {stats['min_l2']:.2f} ~ {stats['max_l2']:.2f}\n")
        f.write(f"  성공한 샘플 평균: {stats['success_avg_l2']:.2f}\n\n")
        
        f.write("조작된 픽셀 통계:\n")
        f.write(f"  평균 픽셀 수: {stats['avg_changed_pixels']:.0f} ± {stats['std_changed_pixels']:.0f}\n")
        f.write(f"  평균 비율: {stats['avg_changed_ratio']:.2f}% ± {stats['std_changed_ratio']:.2f}%\n")
        f.write(f"  성공한 샘플 평균 비율: {stats['success_avg_changed_ratio']:.2f}%\n\n")
        
        f.write("ΔMargin (Confidence Gap Drop) 통계:\n")
        f.write(f"  평균: {stats['avg_delta_margin']:.3f} ± {stats['std_delta_margin']:.3f}\n")
        f.write(f"  성공한 샘플 평균: {stats['success_avg_delta_margin']:.3f}\n\n")
        
        f.write("ΔConfidence (Softmax Drop) 통계:\n")
        f.write(f"  평균: {stats['avg_conf_drop']:.3f} ± {stats['std_conf_drop']:.3f}\n")
        f.write(f"  성공한 샘플 평균: {stats['success_avg_conf_drop']:.3f}\n")
    
    print(f"[INFO] 향상된 시각화 결과 저장: {Path(save_dir) / 'batch_attack_visualization_enhanced.png'}")
    print(f"[INFO] 향상된 통계 요약 저장: {Path(save_dir) / 'batch_statistics_summary_enhanced.txt'}")

# ---------------------------
# Colab-compatible Main Function (ROI 기능 포함)
# ---------------------------
def run_zoo_attack(
    test_dir="processed_data_np224/Testing",
    weights="models/resnet50_binary_final.pth",
    index=0,
    seed=42,
    save_dir="zoo_outputs_untargeted",
    mode="batch",
    max_samples=100,  # 기본값을 100으로 설정
    iters=500,        # 반복 횟수 증가
    batch_coords=128, # 배치 크기 감소 (더 자주 업데이트)
    c0=1.0,           # 초기 c 값 감소 (더 공격적)
    step_eps=0.1,     # 입실론 0.1로 통일
    use_roi=False     # ROI 기능 사용 여부
):
    """Colab용 ZOO 공격 실행 함수 (입실론 0.1, 랜덤 100장 샘플링, ROI 기능 포함)"""
    # 시드 설정
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] 입실론(step_eps): {step_eps}")
    print(f"[INFO] ROI 기능 사용: {use_roi}")

    # 모델 가중치 경로 확인
    weights_path = Path(weights)
    if not weights_path.exists():
        alt = Path(str(weights_path).replace("final", "best"))
        if alt.exists():
            print(f"[WARN] {weights_path} 없음. 대체로 {alt} 사용.")
            weights_path = alt
        else:
            raise FileNotFoundError(f"No weights found: {weights} or {alt}")

    # 모델 로드
    model, class_to_idx, idx_to_class = load_model(weights_path, num_classes=2, device=device)
    print(f"[INFO] Classes: {idx_to_class}")

    # 데이터셋 구성 (ROI 기능 포함)
    ds = build_test_dataset(test_dir, use_roi=use_roi)
    os.makedirs(save_dir, exist_ok=True)

    if mode == "single":
        # 단일 이미지 공격
        print(f"[INFO] 단일 이미지 공격 모드: 인덱스 {index}")
        
        # 샘플 선택
        x_norm, y_true, path, roi_mask = fetch_sample(ds, index, device)
        x_pix_clean = to_pixel(x_norm)

        # 원본 예측
        with torch.no_grad():
            pred_orig = predict_logits(model, x_pix_clean).argmax(1).item()

        # untargeted 공격: source 클래스를 현재 예측으로 설정
        src_idx_tensor = torch.tensor([int(pred_orig)], device=device)

        # ZOO 공격 실행 (입실론 0.1로 통일, ROI 마스크 전달)
        x_adv_pix, n_queries, elapsed, final_margin, best_c = zoo_attack_l2_adam_untargeted(
            model, x_pix_clean, src_idx_tensor,
            roi_mask=roi_mask,  # ROI 마스크 전달
            max_iterations=iters, 
            batch_coords=batch_coords,
            step_eps=step_eps,  # 입실론 0.1로 통일
            lr=2e-3, 
            beta1=0.9, 
            beta2=0.999,
            kappa=0.0, 
            initial_const_c=c0,
            binary_search_steps=3, 
            print_every=max(50, iters//4),
            early_stop_window=max(80, iters//3), 
            abort_early=True,
            early_success=True
        )

        # 공격 후 예측
        with torch.no_grad():
            before = pred_orig
            after  = predict_logits(model, x_adv_pix).argmax(1).item()
        
        success = (after != before)

        # 결과 통계 계산 및 출력 (향상된 버전)
        S = stats_table_like_enhanced(x_pix_clean, x_adv_pix, elapsed, before, model, count_mode='element', tau_255=1.0)
        
        print(f"[공격 {'성공' if success else '실패'}] Before={idx_to_class.get(before, str(before))} → After={idx_to_class.get(after, str(after))}")
        print(f"Queries={n_queries}, FinalMargin={final_margin:.5f}, Best c={best_c:.3g}")
        print(f"투입 섭동량(L2, 0~255): {S['L2_255']:.4f}")
        print(f"변경 픽셀수(element): {S['changed']}/{S['total']} ({S['changed']/S['total']*100:.2f}%)")
        print(f"ΔMargin (Confidence Gap Drop): {S['dmargin']:.3f}")
        print(f"ΔConfidence (Softmax Drop): {S['conf_drop']:.3f}")
        print(f"생성시간(초): {S['time']:.2f}")
        print(f"입실론 값: {step_eps}")
        print(f"ROI 사용: {use_roi and roi_mask is not None}")

        # 시각화 저장
        roi_suffix = "_roi" if use_roi and roi_mask is not None else ""
        stem = f"idx{index:05d}_untargeted_{before}to{after}{roi_suffix}"
        out_png = Path(save_dir) / f"{stem}.png"
        left_t  = f"Original: {idx_to_class.get(before, str(before))}"
        right_t = f"ZOO-Adam-Untgt{roi_suffix} (q={n_queries})\nPred: {idx_to_class.get(after, str(after))}\nΔMargin: {S['dmargin']:.3f}\nε: {step_eps}"
        save_side_by_side_with_roi(x_pix_clean, x_adv_pix, roi_mask, out_png, left_t, right_t, use_roi=use_roi)

        # 결과 로그 저장
        out_json = Path(save_dir) / "zoo_untargeted_result_one_enhanced.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "index": int(index),
                "path": path,
                "pred_before": int(before),
                "pred_after": int(after),
                "success": bool(success),
                "queries": int(n_queries),
                "elapsed_sec": float(elapsed),
                "final_margin": float(final_margin),
                "best_c": float(best_c),
                "L2_255": float(S["L2_255"]),
                "changed": int(S["changed"]),
                "total": int(S["total"]),
                "delta_margin": float(S["dmargin"]),
                "conf_drop": float(S["conf_drop"]),
                "epsilon": float(step_eps),
                "use_roi": bool(use_roi and roi_mask is not None),
                "out_png": str(out_png),
            }, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Saved image → {out_png}")
        print(f"[INFO] Saved log   → {out_json}")
        
    else:
        # 배치 공격 모드 (랜덤 100장 샘플링, ROI 기능 포함)
        print(f"[INFO] 배치 공격 모드: 랜덤 {max_samples}장 이미지 (ε={step_eps}, ROI={use_roi})")
        
        # 데이터 로더 생성 (배치 공격용, ROI 기능 포함)
        test_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        
        # 배치 공격 실행 (랜덤 샘플링, ROI 기능 포함)
        results, success_count, total_queries, total_time = batch_zoo_attack(
            model, test_loader, device, idx_to_class,
            max_samples=max_samples,
            max_iterations=iters,
            batch_coords=batch_coords,
            step_eps=step_eps,  # 입실론 0.1로 통일
            lr=2e-3,
            initial_const_c=c0,
            print_every=max(50, iters//4),
            save_dir=save_dir,
            use_roi=use_roi  # ROI 기능 사용 여부 전달
        )
        
        # 배치 결과 통계 계산
        stats = calculate_batch_statistics_enhanced(results)
        
        # 결과 출력
        print(f"\n{'='*70}")
        print(f"배치 공격 결과 요약 (ε={step_eps}, 랜덤 {max_samples}장, ROI={use_roi})")
        print(f"{'='*70}")
        print(f"총 샘플 수: {stats['total_samples']} (랜덤 샘플링)")
        print(f"입실론 값: {stats['epsilon']}")
        print(f"ROI 사용 샘플: {stats['roi_used']} ({stats['roi_usage_rate']:.1f}%)")
        print(f"성공한 공격: {stats['success_count']}")
        print(f"성공률: {stats['success_rate']:.2f}%")
        print(f"평균 쿼리 수: {stats['avg_queries']:.2f} ± {stats['std_queries']:.2f}")
        print(f"평균 공격 시간: {stats['avg_time']:.2f} ± {stats['std_time']:.2f}초")
        print(f"평균 섭동량 (L2): {stats['avg_l2']:.2f} ± {stats['std_l2']:.2f}")
        print(f"평균 조작된 픽셀 비율: {stats['avg_changed_ratio']:.2f}% ± {stats['std_changed_ratio']:.2f}%")
        print(f"평균 ΔMargin: {stats['avg_delta_margin']:.3f} ± {stats['std_delta_margin']:.3f}")
        print(f"평균 ΔConfidence: {stats['avg_conf_drop']:.3f} ± {stats['std_conf_drop']:.3f}")
        
        # 향상된 시각화 생성
        create_visualization_enhanced(results, save_dir)
        
        # 상세 결과 JSON 저장
        out_json = Path(save_dir) / f"zoo_batch_results_enhanced.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "summary": stats,
                "detailed_results": results,
                "attack_params": {
                    "max_samples": max_samples,
                    "max_iterations": iters,
                    "batch_coords": batch_coords,
                    "step_eps": step_eps,
                    "initial_const_c": c0,
                    "lr": 2e-3,
                    "use_roi": use_roi,
                    "random_sampling": True
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] 배치 결과 저장: {out_json}")
        print(f"[INFO] 향상된 시각화 저장: {Path(save_dir) / 'batch_attack_visualization_enhanced.png'}")
        print(f"[INFO] 향상된 통계 요약 저장: {Path(save_dir) / 'batch_statistics_summary_enhanced.txt'}")

# ---------------------------
# Main (Colab 호환 버전, ROI 기능 포함)
# ---------------------------
def main():
    """메인 실행 함수 (Colab 호환, ROI 기능 포함)"""
    # Jupyter/Colab 환경에서는 argparse 없이 직접 실행
    if 'ipykernel' in sys.modules or 'google.colab' in sys.modules:
        print("[INFO] Colab/Jupyter 환경에서 실행 중...")
        print("[INFO] 기본 매개변수로 배치 공격을 실행합니다.")
        print("[INFO] 사용법: run_zoo_attack(mode='single', index=0, use_roi=True) 등으로 호출하세요.")
        
        # 기본값으로 배치 공격 실행 (입실론 0.1, 랜덤 100장, ROI 기능 비활성화)
        run_zoo_attack(
            test_dir="processed_data_np224/Testing",
            weights="models/resnet50_binary_final.pth",
            mode="batch",
            max_samples=100,  # 랜덤 100장
            iters=100,        # 빠른 테스트를 위해 축소
            batch_coords=128,
            step_eps=0.1,     # 입실론 0.1로 통일
            save_dir="zoo_outputs_untargeted",
            use_roi=False     # ROI 기능 기본값 비활성화
        )
    else:
        # 일반 Python 환경에서는 argparse 사용
        import argparse
        
        ap = argparse.ArgumentParser(description="ZOO L2 (single-scale, Adam only, Untargeted, ε=0.1, ROI 기능 포함)")
        ap.add_argument("--test_dir", type=str, default="processed_data_np224/Testing", 
                        help="테스트 데이터 디렉토리")
        ap.add_argument("--weights", type=str, default="models/resnet50_binary_final.pth", 
                        help="모델 가중치 파일 (final 없으면 best 자동 대체)")
        ap.add_argument("--index", type=int, default=0, 
                        help="테스트셋에서 공격할 샘플 인덱스 (단일 공격용)")
        ap.add_argument("--seed", type=int, default=42, 
                        help="랜덤 시드")
        ap.add_argument("--save_dir", type=str, default="zoo_outputs_untargeted",
                        help="결과 저장 디렉토리")
        
        # 공격 모드 선택
        ap.add_argument("--mode", type=str, default="batch", choices=["single", "batch"],
                        help="공격 모드: single(단일 이미지) 또는 batch(랜덤 100장)")
        ap.add_argument("--max_samples", type=int, default=100,
                        help="배치 공격 시 랜덤 샘플 수")
        
        # 공격 하이퍼파라미터
        ap.add_argument("--iters", type=int, default=320,
                        help="최대 반복 횟수")
        ap.add_argument("--batch_coords", type=int, default=256,
                        help="배치당 좌표 수")
        ap.add_argument("--c0", type=float, default=6.0,
                        help="초기 c 값")
        ap.add_argument("--step_eps", type=float, default=0.1,
                        help="유한 차분 스텝 크기 (입실론, 기본값 0.1)")
        
        # ROI 기능
        ap.add_argument("--use_roi", action="store_true",
                        help="ROI 마스크 기능 사용")
        
        args = ap.parse_args()
        
        run_zoo_attack(
            test_dir=args.test_dir,
            weights=args.weights,
            index=args.index,
            seed=args.seed,
            save_dir=args.save_dir,
            mode=args.mode,
            max_samples=args.max_samples,
            iters=args.iters,
            batch_coords=args.batch_coords,
            c0=args.c0,
            step_eps=args.step_eps,
            use_roi=args.use_roi
        )

if __name__ == "__main__":
    main()