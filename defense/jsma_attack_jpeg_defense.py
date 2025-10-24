"""
JSMA 공격 + JPEG 방어 평가
- JSMA (Jacobian-based Saliency Map Attack) 구현
- JPEG 압축을 통한 방어 효과 평가
- 마진 기반 공격 (margin = logit[y] - max_{j≠y} logit[j])
- Adam 스타일 모멘텀 적용
- 동적 k 조절 (정체 시 자동 확대)
"""

import os
import io
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 프로젝트 모듈은 setup_environment() 후에 임포트하도록 변경


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
        
        # sys.path에 현재 디렉토리 추가
        if work_dir not in sys.path:
            sys.path.insert(0, work_dir)
            print(f"[INFO] sys.path에 추가: {work_dir}")
        
        base_path = Path(work_dir)
        
    except ImportError:
        IN_COLAB = False
        print("[INFO] 로컬 환경 감지")
        base_path = Path(os.getcwd())
    
    return base_path, IN_COLAB


def seed_everything(seed=42):
    """재현 가능한 결과를 위한 시드 고정"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] 시드 고정 완료: {seed}")


def setup_logging(out_dir):
    """로깅 설정"""
    import logging
    import os
    from datetime import datetime
    
    os.makedirs(out_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(out_dir, "attack.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    print(f"[INFO] 로그 파일 생성: {log_file}")
    return logging.getLogger(__name__)


def setup_device(mode='auto'):
    """디바이스 설정"""
    import torch
    
    if mode == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] 강제 CUDA 사용: {device}")
    elif mode == 'cpu':
        device = torch.device("cpu")
        print(f"[INFO] CPU 사용: {device}")
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
        
        # 필요한 함수들이 있는지 확인
        required_functions = ['to_norm', 'to_pixel', 'calculate_attack_metrics', 'visualize_attack_result']
        missing_functions = []
        
        for func_name in required_functions:
            if not hasattr(attack_utils_module, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"[WARNING] attack_utils에서 누락된 함수들: {missing_functions}")
        else:
            print("[INFO] attack_utils의 모든 필수 함수 확인 완료")
            
    except ImportError as e:
        print(f"[WARNING] attack_utils 모듈 임포트 실패: {e}")
        print("[INFO] 현재 디렉토리와 파일 존재 여부 확인 중...")
        
        import os
        current_dir = os.getcwd()
        attack_utils_path = os.path.join(current_dir, "attack_utils.py")
        
        print(f"[INFO] 현재 디렉토리: {current_dir}")
        print(f"[INFO] attack_utils.py 존재 여부: {os.path.exists(attack_utils_path)}")
        
        if os.path.exists(attack_utils_path):
            print("[INFO] attack_utils.py 파일이 존재합니다. sys.path에 현재 디렉토리 추가 시도...")
            import sys
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
                try:
                    import attack_utils
                    attack_utils_module = attack_utils
                    print("[INFO] sys.path 수정 후 attack_utils 임포트 성공!")
                except ImportError as e2:
                    print(f"[ERROR] sys.path 수정 후에도 임포트 실패: {e2}")
                    attack_utils_module = None
        else:
            print("[ERROR] attack_utils.py 파일이 현재 디렉토리에 없습니다.")
            attack_utils_module = None


# 전역 변수
attack_utils_module = None


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
# JSMA 공격 함수들
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
    JSMA 공격 - 애플리케이션용 인터페이스
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
# JPEG 방어 관련 함수들
# ==============================================================================

def jpeg_compress_tensor_pil(x_tensor, quality=75):
    """
    JPEG 압축 적용
    
    Args:
        x_tensor: (1,C,H,W) or (C,H,W) in [0,1]
        quality: JPEG 품질 (1-100)
    
    Returns:
        압축된 텐서 (동일한 shape)
    """
    single = False
    if x_tensor.dim() == 3:
        x_tensor = x_tensor.unsqueeze(0)
        single = True

    device = x_tensor.device
    x = x_tensor.detach().cpu().clamp(0, 1)
    bs, c, h, w = x.shape
    out = torch.empty_like(x)

    for i in range(bs):
        arr = (x[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr)
        bio = io.BytesIO()
        pil.save(bio, format='JPEG', quality=int(quality))
        bio.seek(0)
        pil2 = Image.open(bio).convert('RGB')
        arr2 = np.asarray(pil2).astype(np.float32) / 255.0
        out[i] = torch.from_numpy(arr2).permute(2, 0, 1)

    out = out.to(device)
    return out.squeeze(0) if single else out


def margin_from_logits(logits, y_idx):
    """
    Margin 계산: logit[y] - max_{j!=y} logit[j]
    """
    with torch.no_grad():
        logits0 = logits[0]
        tmp = logits0.clone()
        tmp[y_idx] = -1e9
        max_other = tmp.max().item()
        margin = float((logits0[y_idx].item() - max_other))
    return margin


@torch.no_grad()
def eval_example_with_jpeg(model, x_orig_pix, x_adv_pix, mean, std, quality=75, y_true=None):
    """
    JPEG 적용 전후 평가
    
    Args:
        model: 타겟 모델
        x_orig_pix: 원본 이미지 (1,C,H,W) [0,1]
        x_adv_pix: 공격 이미지 (1,C,H,W) [0,1]
        mean, std: 정규화 파라미터
        quality: JPEG 품질
        y_true: 실제 레이블
    
    Returns:
        메트릭 딕셔너리
    """
    device = next(model.parameters()).device

    # 정규화 함수
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
    logits_adv = model(xadv_norm)

    probs_clean = torch.softmax(logits_clean, dim=1)[0]
    probs_adv = torch.softmax(logits_adv, dim=1)[0]
    pred_clean = int(probs_clean.argmax().item())
    pred_adv = int(probs_adv.argmax().item())
    conf_clean = float(probs_clean[pred_clean].item())
    conf_adv = float(probs_adv[pred_adv].item())

    # 레이블 설정
    if y_true is None:
        y_idx = pred_clean
    else:
        y_idx = int(y_true)

    margin_clean = margin_from_logits(logits_clean, y_idx)
    margin_adv = margin_from_logits(logits_adv, y_idx)
    margin_drop = margin_clean - margin_adv
    conf_drop = float((torch.softmax(logits_clean, dim=1)[0, y_idx] - torch.softmax(logits_adv, dim=1)[0, y_idx]).item())

    # JPEG 압축 적용
    x_adv_jpeg_pix = jpeg_compress_tensor_pil(x_adv_pix, quality=quality).to(device)
    xadv_jpeg_norm = _to_norm_local(x_adv_jpeg_pix)

    logits_adv_jpeg = model(xadv_jpeg_norm)
    probs_adv_jpeg = torch.softmax(logits_adv_jpeg, dim=1)[0]
    pred_adv_jpeg = int(probs_adv_jpeg.argmax().item())
    conf_adv_jpeg = float(probs_adv_jpeg[pred_adv_jpeg].item())

    margin_adv_jpeg = margin_from_logits(logits_adv_jpeg, y_idx)
    margin_drop_after = margin_clean - margin_adv_jpeg
    conf_drop_after = float((torch.softmax(logits_clean, dim=1)[0, y_idx] - probs_adv_jpeg[y_idx]).item())

    # 섭동 계산
    delta_adv = (x_adv_pix - x_orig_pix).detach().cpu()
    l2_255 = (delta_adv.view(1, -1) * 255.0).norm(p=2).item()
    l1 = delta_adv.abs().sum().item()
    linf = delta_adv.abs().max().item()
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
# 배치 실험 실행 함수
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
    JSMA 공격 후 JPEG 방어 평가
    
    1) 각 샘플에 대해 JSMA 공격 실행
    2) JPEG 압축 전후 평가
    3) 통계 집계 및 출력
    """
    device = next(model.parameters()).device

    # 결과 수집 리스트
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

    # JSMA 호출 래퍼
    def _call_jsma(x, y):
        params = jsma_params or {}
        return jsma_attack(model=model, x_pix=x, y_true=y, mean=mean, std=std, **params)

    n_run = min(n_samples, len(test_set))
    
    # 랜덤 샘플링으로 편향 방지 (다른 공격들과 동일한 방식)
    np.random.seed(42)
    sampled_indices = np.random.choice(len(test_set), size=n_run, replace=False)
    
    print(f"[INFO] 전체 {len(test_set)}장 중 랜덤하게 {n_run}장 샘플링")
    
    for i, sample_idx in enumerate(sampled_indices):
        sample = test_set[sample_idx]
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

        # JPEG 전후 평가
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
              f"pred: {class_names[metrics['pred_clean']]} -> adv:{class_names[metrics['pred_adv']]} -> adv+jpeg:{class_names[metrics['pred_adv_jpeg']]} "
              f"| L2_255={metrics['l2_255']:.3f} changed={metrics['changed_pixels']}/{metrics['total_pixels']} time={elapsed:.2f}s")

        # 첫 번째 샘플 시각화
        if i == 0 and visualize_first and viz_func is not None:
            try:
                viz_func(
                    model,
                    x_orig_pix=x,
                    x_adv_pix=x_adv,
                    x_adv_jpeg_pix=metrics['x_adv_jpeg_pix'],
                    class_names=class_names,
                    mean=mean, std=std
                )
            except Exception as e:
                print(f"[VIS] jpeg visualization failed: {e}")

    # 통계 집계
    ASR_before = float(np.mean(success_before_list) * 100.0)
    ASR_after = float(np.mean(success_after_list) * 100.0)
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
    """정규화 헬퍼"""
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
    """예측 및 신뢰도"""
    with torch.no_grad():
        logits = model(_to_norm_local(x_pix, mean, std))
        probs = torch.softmax(logits, dim=1)
        pred = int(probs.argmax(1).item())
        conf = float(probs[0, pred].item())
    return pred, conf


def _to_numpy_img(x):
    """텐서를 numpy 이미지로 변환"""
    if x.dim() == 4:
        x = x.squeeze(0)
    return x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def visualize_attack_with_jpeg(
    model,
    x_orig_pix,
    x_adv_pix,
    x_adv_jpeg_pix,
    class_names=None,
    mean=None, std=None,
    amp_delta=3.0,
    out_dir="./results/jsma_jpeg",
    filename_prefix="attack_vis_jpeg",
    display=False
):
    """
    JPEG 방어 시각화 (1×4 레이아웃)
    
    [Original] [Adversarial] [Adv+JPEG] [Perturbation Removed]
    """
    os.makedirs(out_dir, exist_ok=True)

    # 예측/신뢰도
    p0, c0 = _pred_and_conf(model, x_orig_pix, mean, std)
    p1, c1 = _pred_and_conf(model, x_adv_pix, mean, std)
    p2, c2 = _pred_and_conf(model, x_adv_jpeg_pix, mean, std)

    # 이미지 변환
    img0 = _to_numpy_img(x_orig_pix)
    img1 = _to_numpy_img(x_adv_pix)
    img2 = _to_numpy_img(x_adv_jpeg_pix)

    # JPEG가 제거한 섭동
    delta_removed = (x_adv_pix - x_adv_jpeg_pix).squeeze(0).norm(p=2, dim=0).cpu().numpy()
    delta_removed_vis = delta_removed * amp_delta

    # 시각화
    fig = plt.figure(figsize=(16, 4))
    
    ax = plt.subplot(1, 4, 1)
    ax.imshow(img0)
    ax.set_title(f"Original\n{class_names[p0]}({c0:.3f})")
    ax.axis('off')
    
    ax = plt.subplot(1, 4, 2)
    ax.imshow(img1)
    ax.set_title(f"Adversarial\n{class_names[p1]}({c1:.3f})")
    ax.axis('off')
    
    ax = plt.subplot(1, 4, 3)
    ax.imshow(img2)
    ax.set_title(f"Adv + JPEG\n{class_names[p2]}({c2:.3f})")
    ax.axis('off')
    
    ax = plt.subplot(1, 4, 4)
    im = ax.imshow(delta_removed_vis, cmap='magma')
    plt.colorbar(im)
    ax.set_title(f"|δ_removed| (×{amp_delta:g})")
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
# 메인 실행
# ==============================================================================

def main():
    """메인 실행 함수 (다른 공격 파일들과 일관성 있게 수정)"""
    try:
        print("\n" + "="*70)
        print("JSMA Attack with JPEG Defense 실행")
        print("="*70)
        
        # 1. 환경 설정
        base_path, is_colab = setup_environment()
        
        # 2. 시드 고정
        seed_everything(42)
        
        # 3. 로깅 설정
        logger = setup_logging(str(base_path / "jsma_jpeg_results"))
        
        # 4. 디바이스 설정
        device, use_amp = setup_device('auto')
        
        # 5. attack_utils 모듈 임포트 확인
        ensure_attack_utils_import()
        
        # 6. 모델 로드
        from resnet50_model import load_trained_model
        
        print(f"[INFO] 사용 디바이스: {device}")
        
        # 경로 설정 (통일된 경로 사용)
        checkpoint_path = base_path / "models" / "resnet50_binary_best.pth"
        data_path = base_path / "processed_data_np224" / "Testing"
        
        print(f"[INFO] 모델 경로: {checkpoint_path}")
        print(f"[INFO] 데이터 경로: {data_path}")
        
        model = load_trained_model(str(checkpoint_path), device=device)
        model.eval()
        print("[INFO] 모델 로드 완료")
        
        # 7. 데이터셋 로드
        test_set = BrainTumorDatasetWithROI(
            str(data_path),
            transform=transforms.ToTensor()
        )
        class_names = test_set.class_names
        print(f"[INFO] 테스트 데이터 로드 완료: {len(test_set)}개 샘플")
        
        # 8. 공격 파라미터 설정
        jsma_params = {
            'theta': 0.12,
            'max_pixels_pct': 0.10,
            'k_small': 4,
            'restarts': 4,
            'topk_pool': 5000
        }
        
        # 정규화 파라미터 (모델이 내부에서 정규화하는 경우 [0,0,0], [1,1,1] 사용)
        mean = [0, 0, 0]
        std = [1, 1, 1]
        
        print("\n" + "="*70)
        print("JSMA + JPEG Defense 평가 실행")
        print("="*70)
        
        # 9. 실험 실행
        results = run_jsma_and_eval_jpeg_defense(
            model=model,
            test_set=test_set,
            class_names=class_names,
            mean=mean,
            std=std,
            n_samples=50,
            jsma_params=jsma_params,
            jpeg_quality=60,
            visualize_first=True,
            viz_func=visualize_attack_with_jpeg
        )
        
        # 10. 결과 저장
        results_dir = base_path / "jsma_jpeg_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"jsma_jpeg_results_{int(time.time())}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # per_sample 데이터는 제외하고 저장 (너무 클 수 있음)
            save_dict = {k: v for k, v in results.items() if k != 'per_sample'}
            json.dump(save_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n[INFO] 결과 저장: {results_file}")
        print("[INFO] 실험 완료!")
        
    except ImportError as e:
        print(f"[ERROR] 모듈 임포트 실패: {e}")
        print("[INFO] resnet50_model.py 파일이 같은 디렉토리에 있는지 확인하세요.")
    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

