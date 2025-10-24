"""
Square Attack L2 구현
- ROI(Region of Interest) 기반 공격 지원
- L2 제약 조건 하에서 적대적 예제 생성
- 타겟/비타겟 공격 지원
"""

# 전역 변수
attack_utils_module = None

import os
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

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


# ==============================================================================
# 유틸리티 함수
# ==============================================================================

def to_norm(x):
    """정규화 함수 (integrated.py와 동일하게 정규화 없음)"""
    return x

@torch.no_grad()
def predict_logits(model, x_pix):
    """로짓 예측 함수"""
    model.eval()
    return model(to_norm(x_pix))


# ==============================================================================
# 손실 함수
# ==============================================================================

def margin_loss_untargeted(logits, y):
    """비타겟 공격을 위한 마진 손실"""
    bs = logits.size(0)
    correct_logit = logits[torch.arange(bs, device=logits.device), y]
    tmp = logits.clone()
    tmp[torch.arange(bs, device=logits.device), y] = -1e9
    max_other = tmp.max(dim=1).values
    return correct_logit - max_other   # <=0 success


def margin_loss_targeted(logits, y_tgt):
    """타겟 공격을 위한 마진 손실"""
    bs = logits.size(0)
    target_logit = logits[torch.arange(bs, device=logits.device), y_tgt]
    tmp = logits.clone()
    tmp[torch.arange(bs, device=logits.device), y_tgt] = -1e9
    max_other = tmp.max(dim=1).values
    return max_other - target_logit    # <=0 success


def ce_loss_targeted(logits, y_tgt):
    """타겟 공격을 위한 크로스 엔트로피 손실"""
    return nn.CrossEntropyLoss(reduction='none')(logits, y_tgt)


# ==============================================================================
# Square Attack 보조 함수
# ==============================================================================

def p_selection(p_init, it, n_iters):
    """공격 진행에 따른 패치 크기 조절"""
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
    """가우시안 섭동 생성 (직사각형 영역)"""
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
    """가우시안 섭동 생성 (정사각형)"""
    delta = torch.zeros(s, s, device=device)
    delta[:s//2] = pseudo_gaussian_pert_rectangles(s//2, s, device)
    delta[s//2:] = -pseudo_gaussian_pert_rectangles(s - s//2, s, device)
    delta = delta / (delta.pow(2).sum().sqrt() + 1e-12)
    if torch.rand(1, device=device).item() > 0.5:
        delta = delta.t()
    return delta


# ==============================================================================
# ROI 마스크 관련 함수
# ==============================================================================

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
    """마스크 내부 L2 노름 계산"""
    # t, m: (1,C,H,W). m in {0,1}. 마스크 내부만 계산
    v = (t * m).view(1, -1)
    return v.norm(p=2, dim=1, keepdim=True).view(1,1,1,1).clamp(min=1e-12)


def _l2_norm_over_mask_region(t, m, region_mask):
    """특정 영역과 마스크 교집합의 L2 노름 계산"""
    # region_mask: (1,C,H,W) with {0,1} selecting region; use intersection with m
    mm = torch.maximum((m>0).float(), torch.zeros_like(m))  # ensure 0/1
    r = torch.maximum((region_mask>0).float(), torch.zeros_like(region_mask))
    v = (t * mm * r).view(1, -1)
    return v.norm(p=2, dim=1, keepdim=True).view(1,1,1,1).clamp(min=1e-12)


def _apply_mask(delta, m):
    """마스크 적용 (ROI 밖은 0으로)"""
    delta.mul_(m)
    return delta


# ==============================================================================
# Square Attack L2 (핵심 알고리즘)
# ==============================================================================

def square_attack_l2_single_roi(
    model,
    x_pix, y_true, mask_2d, y_target=None,
    eps=0.5, n_iters=2000, p_init=0.1,
    targeted=False, use_ce_for_targeted=True, seed=0
):
    """
    Square Attack L2 with ROI support
    
    Args:
        model: 타겟 모델
        x_pix: (1,3,H,W), [0,1] 범위의 입력 이미지
        y_true: 실제 레이블 (비타겟 공격용)
        mask_2d: (H,W) torch/numpy, {0,1}/bool → ROI만 공격
        y_target: 타겟 레이블 (타겟 공격용)
        eps: L2 제약 크기 (픽셀 공간 기준)
        n_iters: 최대 반복 횟수
        p_init: 초기 패치 크기 비율
        targeted: True면 타겟 공격, False면 비타겟 공격
        use_ce_for_targeted: 타겟 공격 시 CE 손실 사용 여부
        seed: 랜덤 시드
    
    Returns:
        x_best: 공격된 이미지
        n_queries: 모델 쿼리 수
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
        # scale^2 = max(eps^2 - ||delta||_ROI^2, 0)/C + ||(delta * mask_union)||_ROI^2
        scale = ((torch.clamp(eps**2 - curr_norm_img**2, min=0.0) / c) + norms_windows**2).sqrt()

        # 새 패치 정규화 후 스케일
        nd_norm = _l2_norm_over_mask(new_deltas, torch.ones_like(new_deltas))  # 패치 자체 norm
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
# 메트릭 계산
# ==============================================================================

@torch.no_grad()
def stats_table_like(model, x_pix_clean, x_adv_pix, elapsed_sec, y_true, count_mode='element', tau_255=1.0):
    """
    공격 결과 통계 계산
    
    Args:
        model: 타겟 모델
        x_pix_clean: 원본 이미지
        x_adv_pix: 공격된 이미지
        elapsed_sec: 소요 시간
        y_true: 실제 레이블
        count_mode: 'element' (픽셀별) 또는 'spatial' (공간별)
        tau_255: 변경 픽셀 판정 임계값 (0-255 스케일)
    
    Returns:
        메트릭 딕셔너리
    """
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


# ==============================================================================
# 시각화
# ==============================================================================

def _tensor_img_to_numpy(x):
    """(1,3,H,W) -> (H,W,3) 변환"""
    if x.dim() == 4:
        x = x.squeeze(0)
    return x.detach().clamp(0,1).permute(1,2,0).cpu().numpy()


def _pred_and_conf(model, logits):
    """예측 및 신뢰도 계산"""
    probs = torch.softmax(logits, dim=1)
    pred  = probs.argmax(1).item()
    conf  = probs[0, pred].item()
    return pred, conf


# ==============================================================================
# 배치 실행 함수
# ==============================================================================

def _full_ones_mask_like(x1):
    """전체 이미지를 ROI로 설정하는 마스크 생성"""
    # x1: (1,3,H,W)
    _, _, H, W = x1.shape
    return torch.ones(H, W, dtype=torch.float32, device=x1.device)


def _safe_mean(arr):
    """안전한 평균 계산 (빈 배열 처리)"""
    return float(np.mean(arr)) if len(arr) > 0 else float('nan')


def run_square_roi_attack(
    model,
    test_loader,
    class_names,
    eps=2.0,
    n_iters=10000,  # integrated.py와 동일하게 수정
    p_init=0.1,
    save_results=True,
    create_visualizations=True,
    out_dir="./square_roi_results",
    amp_heat=4.0
):
    """
    ROI Square Attack 실행 (통일된 인터페이스)
    
    Args:
        model: 타겟 모델
        test_loader: 데이터 로더
        class_names: 클래스 이름 리스트
        eps: L2 제약 크기
        n_iters: 최대 반복 횟수
        p_init: 초기 패치 크기 비율
        save_results: True면 결과를 파일로 저장
        create_visualizations: True면 시각화 차트 생성
        out_dir: 결과 저장 디렉토리
        amp_heat: 섭동 히트맵 증폭 계수
    
    Returns:
        results: 결과 딕셔너리 리스트
        statistics: 통계 딕셔너리
    """
    global attack_utils_module
    
    device = next(model.parameters()).device
    
    # 전체 데이터를 수집하여 랜덤 샘플링 적용
    all_data = []
    for batch_data in test_loader:
        if len(batch_data) == 3:  # ROI 마스크 포함
            x_pix, y_true, masks_2d = batch_data
            for i in range(x_pix.size(0)):
                all_data.append((x_pix[i:i+1], y_true[i:i+1], masks_2d[i]))
        else:  # ROI 마스크 없음
            x_pix, y_true = batch_data[:2]
            for i in range(x_pix.size(0)):
                all_data.append((x_pix[i:i+1], y_true[i:i+1], None))
    
    # 랜덤 샘플링으로 편향 방지 (다른 공격들과 동일한 방식)
    n_total = len(all_data)
    n_samples = min(100, n_total)  # 기본 100개 샘플
    
    # 시드 고정하여 재현 가능한 랜덤 샘플링
    np.random.seed(42)
    sampled_indices = np.random.choice(n_total, size=n_samples, replace=False)
    sampled_data = [all_data[i] for i in sampled_indices]
    
    x_adv_list = []
    elapsed_list = []
    success_list = []
    l2_list = []
    changed_list = []
    total_list = []
    dmargin_list = []
    conf_drop_list = []

    print(f"\n===== Square ROI Attack (eps={eps}, n_iters={n_iters}) =====")
    print(f"공격 대상: {n_samples}개 샘플 (ROI 영역만, 랜덤 샘플링)\n")

    for idx, (xi, yi, mi) in enumerate(sampled_data):
        xi = xi.to(device)
        yi = yi.to(device)
        if mi is not None:
            mi = mi.to(device)
            mi_eff = mi          # ROI 공격 (데이터셋 마스크 사용)
            mask_for_vis = mi
        else:
            mi_eff = _full_ones_mask_like(xi)  # ROI 마스크가 없으면 전체 이미지
            mask_for_vis = None

        with torch.no_grad():
            src_idx = predict_logits(model, xi).argmax(1)  # untargeted: 현재 예측을 소스 라벨로

        # Square Attack 실행
        x_adv_pix, n_queries, elapsed, final_margin = square_attack_l2_single_roi(
            model, xi, y_true=src_idx, mask_2d=mi_eff,
            eps=eps, n_iters=n_iters, p_init=p_init,
            targeted=False, use_ce_for_targeted=True, seed=0
        )

        # 단일 샘플 통계
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

        # 샘플별 성공 여부 (통일된 형식)
        pred_orig = predict_logits(model, xi).argmax(1).item()
        pred_adv  = predict_logits(model, x_adv_pix).argmax(1).item()
        success   = (pred_orig != pred_adv)
        success_mark = '✅' if success else '❌'
        print(f"[{idx+1:3d}/{n_samples}] {success_mark} {class_names[pred_orig]} → {class_names[pred_adv]} "
              f"| L2={S['L2_255']:.2f} | Δconf={conf_drop:.3f} | {S['time']:.3f}s")

        # 첫 장 시각화 (통일된 함수 사용)
        if idx == 0 and create_visualizations:
            # attack_utils 전역 변수 사용
            if attack_utils_module is not None:
                attack_utils_module.visualize_attack_result(
                    model, xi, x_adv_pix,
                    class_names=class_names,
                    roi_mask=mask_for_vis,
                    amp_heat=amp_heat,
                    out_dir=out_dir,
                    filename_prefix="square_sample0",
                    display=False,
                    save_file=save_results
                )
            else:
                print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 시각화를 건너뜁니다.")

        # 누적
        x_adv_list.append(x_adv_pix)
        elapsed_list.append(S['time'])
        success_list.append(success)
        l2_list.append(S['L2_255'])
        changed_list.append(S['changed'])
        total_list.append(S['total'])
        dmargin_list.append(S['dmargin'])

    # 통계 계산
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

    # 결과를 통일된 형식으로 변환
    unified_results = []
    for i, (success, l2, changed_ratio, elapsed, dmargin, conf_drop) in enumerate(zip(
        success_list, l2_list, 
        [c/t*100 for c,t in zip(changed_list, total_list)], 
        elapsed_list, dmargin_list, conf_drop_list
    )):
        unified_results.append({
            'success': success,
            'l2_255': l2,
            'changed_ratio': changed_ratio,
            'elapsed_time': elapsed,
            'conf_drop': conf_drop,
            'sample_idx': i
        })
    
    # 통계 계산 (통일된 함수 사용)
    if attack_utils_module is not None:
        statistics = attack_utils_module.calculate_batch_statistics(unified_results, attack_name="Square")
        attack_utils_module.print_statistics(statistics)
    else:
        print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 기본 통계만 출력합니다.")
        ASR = float(np.mean(success_list) * 100.0)
        mean_l2 = float(np.mean(l2_list))
        mean_time = float(np.mean(elapsed_list))
        print(f"\n[Square Attack Results]")
        print(f"ASR: {ASR:.2f}%")
        print(f"Mean L2: {mean_l2:.2f}")
        print(f"Mean Time: {mean_time:.3f}s")
        statistics = {'ASR': ASR, 'mean_l2': mean_l2, 'mean_time': mean_time}
    
    # 결과 저장 (통일된 형식)
    if save_results:
        import os
        os.makedirs(out_dir, exist_ok=True)
        
        # CSV 저장
        if attack_utils_module is not None:
            attack_utils_module.save_results_to_csv(unified_results, os.path.join(out_dir, "square_results.csv"))
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. CSV 저장을 건너뜁니다.")
        
        # JSON 저장
        if attack_utils_module is not None:
            attack_utils_module.save_results_to_json(statistics, os.path.join(out_dir, "square_statistics.json"))
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. JSON 저장을 건너뜁니다.")
    
    # 시각화 차트 생성
    if create_visualizations:
        if attack_utils_module is not None:
            attack_utils_module.create_result_visualization(unified_results, out_dir=out_dir, attack_name="Square")
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 시각화 차트 생성을 건너뜁니다.")

    return unified_results, statistics


def run_square_full_attack(
    model,
    test_loader,
    class_names,
    eps=2.0,
    n_iters=2000,
    p_init=0.1,
    save_results=True,
    create_visualizations=True,
    out_dir="./square_results",
    amp_heat=4.0
):
    """
    Full-image Square Attack 실행 (통일된 인터페이스)
    
    Args:
        model: 타겟 모델
        test_loader: 데이터 로더
        class_names: 클래스 이름 리스트
        eps: L2 제약 크기
        n_iters: 최대 반복 횟수
        p_init: 초기 패치 크기 비율
        save_results: True면 결과를 파일로 저장
        create_visualizations: True면 시각화 차트 생성
        out_dir: 결과 저장 디렉토리
        amp_heat: 섭동 히트맵 증폭 계수
    
    Returns:
        results: 결과 딕셔너리 리스트
        statistics: 통계 딕셔너리
    """
    global attack_utils_module
    
    device = next(model.parameters()).device
    
    # 전체 데이터를 수집하여 랜덤 샘플링 적용
    all_data = []
    for batch_data in test_loader:
        if len(batch_data) == 3:  # ROI 마스크 포함
            x_pix, y_true, masks_2d = batch_data
            for i in range(x_pix.size(0)):
                all_data.append((x_pix[i:i+1], y_true[i:i+1], masks_2d[i]))
        else:  # ROI 마스크 없음
            x_pix, y_true = batch_data[:2]
            for i in range(x_pix.size(0)):
                all_data.append((x_pix[i:i+1], y_true[i:i+1], None))
    
    # 랜덤 샘플링으로 편향 방지 (다른 공격들과 동일한 방식)
    n_total = len(all_data)
    n_samples = min(100, n_total)  # 기본 100개 샘플
    
    # 시드 고정하여 재현 가능한 랜덤 샘플링
    np.random.seed(42)
    sampled_indices = np.random.choice(n_total, size=n_samples, replace=False)
    sampled_data = [all_data[i] for i in sampled_indices]
    
    print(f"\n===== Square Full Image Attack (eps={eps}, n_iters={n_iters}) =====")
    print(f"공격 대상: {n_samples}개 샘플 (전체 이미지, 랜덤 샘플링)\n")
    
    # 결과 저장용 리스트들
    x_adv_list = []
    elapsed_list = []
    success_list = []
    l2_list = []
    changed_list = []
    total_list = []
    dmargin_list = []
    conf_drop_list = []

    for idx, (xi, yi, mi) in enumerate(sampled_data):
        xi = xi.to(device)
        yi = yi.to(device)
        
        # 전체 이미지 공격 (ROI 무시)
        mi_eff = _full_ones_mask_like(xi)  # 전체 이미지 공격 (올-원 마스크)
        mask_for_vis = None

        with torch.no_grad():
            src_idx = predict_logits(model, xi).argmax(1)  # untargeted: 현재 예측을 소스 라벨로

        # Square Attack 실행
        x_adv_pix, n_queries, elapsed, final_margin = square_attack_l2_single_roi(
            model, xi, y_true=src_idx, mask_2d=mi_eff,
            eps=eps, n_iters=n_iters, p_init=p_init,
            targeted=False, use_ce_for_targeted=True, seed=0
        )

        # 단일 샘플 통계
        S = stats_table_like(model, xi, x_adv_pix, elapsed, y_true=int(src_idx.item()), 
                            count_mode='element', tau_255=1.0)

        with torch.no_grad():
            pred_orig = model(xi).argmax(1).item()
            pred_adv = model(x_adv_pix).argmax(1).item()
        
        success = (pred_orig != pred_adv)
        conf_drop = S.get('conf_drop', 0.0)
        
        # 진행 상황 출력 (통일된 형식)
        success_mark = '✅' if success else '❌'
        print(f"[{idx+1:3d}/{n_samples}] {success_mark} {class_names[pred_orig]} → {class_names[pred_adv]} "
              f"| L2={S['L2_255']:.2f} | Δconf={conf_drop:.3f} | {S['time']:.3f}s")
        
        # 첫 장 시각화 (통일된 함수 사용)
        if idx == 0 and create_visualizations:
            if attack_utils_module is not None:
                attack_utils_module.visualize_attack_result(
                    model, xi, x_adv_pix,
                    class_names=class_names,
                    roi_mask=mi,
                    amp_heat=amp_heat,
                    out_dir=out_dir,
                    filename_prefix="square_full_sample0",
                    display=False,
                    save_file=save_results
                )
            else:
                print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 시각화를 건너뜁니다.")

        # 누적
        x_adv_list.append(x_adv_pix)
        elapsed_list.append(S['time'])
        success_list.append(success)
        l2_list.append(S['L2_255'])
        changed_list.append(S['changed'])
        total_list.append(S['total'])
        dmargin_list.append(S['dmargin'])
        conf_drop_list.append(conf_drop)

    # 통계 계산
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

    # 결과를 통일된 형식으로 변환
    unified_results = []
    for i, (success, l2, changed_ratio, elapsed, dmargin, conf_drop) in enumerate(zip(
        success_list, l2_list, 
        np.array(changed_list) / np.array(total_list), 
        elapsed_list, dmargin_list, conf_drop_list
    )):
        unified_results.append({
            'success': success,
            'pred_original': model(x_adv_list[i]).argmax(1).item() if success else model(x_adv_list[i]).argmax(1).item(),
            'pred_adv': model(x_adv_list[i]).argmax(1).item(),
            'l2_norm': l2,
            'conf_drop': conf_drop,
            'elapsed_time': elapsed,
            'eps': eps,
            'n_iters': n_iters,
            'sample_idx': i,
            'use_roi': False  # 전체 이미지 공격
        })

    # 통계 계산 (통일된 함수 사용)
    if attack_utils_module is not None:
        statistics = attack_utils_module.calculate_batch_statistics(unified_results, attack_name="Square_Full")
        attack_utils_module.print_statistics(statistics)
    else:
        print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 기본 통계만 출력합니다.")
        mean_l2 = float(np.mean(l2_list))
        mean_time = float(np.mean(elapsed_list))
        print(f"\n[Square Full Attack Results]")
        print(f"ASR: {ASR:.2f}%")
        print(f"Mean L2: {mean_l2:.2f}")
        print(f"Mean Time: {mean_time:.3f}s")
        statistics = {'ASR': ASR, 'mean_l2': mean_l2, 'mean_time': mean_time}
    
    # 결과 저장 (통일된 형식)
    if save_results:
        import os
        os.makedirs(out_dir, exist_ok=True)
        
        # CSV 저장
        if attack_utils_module is not None:
            attack_utils_module.save_results_to_csv(unified_results, os.path.join(out_dir, "square_full_results.csv"))
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. CSV 저장을 건너뜁니다.")
        
        # JSON 저장
        if attack_utils_module is not None:
            attack_utils_module.save_results_to_json(statistics, os.path.join(out_dir, "square_full_statistics.json"))
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. JSON 저장을 건너뜁니다.")
    
    # 시각화 차트 생성
    if create_visualizations:
        if attack_utils_module is not None:
            attack_utils_module.create_result_visualization(unified_results, out_dir=out_dir, attack_name="Square_Full")
        else:
            print("[WARNING] attack_utils 모듈이 로드되지 않았습니다. 시각화 차트 생성을 건너뜁니다.")

    return unified_results, statistics


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
    print("Square Attack 실행")
    print("="*70)
    
    # 1. 환경 설정
    base_path, is_colab = setup_environment()
    
    # 2. 시드 고정
    seed_everything(42)
    
    # 3. 로깅 설정
    log_file = setup_logging("square_results")
    
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
        test_loader = DataLoader(test_set, batch_size=100, shuffle=False)
        class_names = test_set.class_names
        print(f"[INFO] 테스트 데이터 로드 완료: {len(test_set)}개 샘플")
        
        # 6. 공격 파라미터 설정
        eps = 2.0
        n_iters = 10000  # integrated.py와 동일하게 수정
        p_init = 0.1
        
        print("\n" + "="*70)
        print("Square Attack 실행")
        print("="*70)
        
        # 7. Square 전체 이미지 공격 실행
        results_full, statistics_full = run_square_full_attack(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            eps=eps,
            n_iters=n_iters,
            p_init=p_init,
            save_results=True,
            create_visualizations=True,
            out_dir="./square_results"
        )
        
        # 통계에서 필요한 값들 추출
        ASR_full = statistics_full.get('ASR', 0.0)
        mean_L2_full = statistics_full.get('mean_l2', 0.0)
        mean_changed_pct_full = statistics_full.get('mean_changed_ratio', 0.0)
        mean_time_full = statistics_full.get('mean_time', 0.0)
        success_list_full = [r['success'] for r in results_full]
        
        # 8. Square ROI 공격 실행
        results_roi, statistics_roi = run_square_roi_attack(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            eps=eps,
            n_iters=n_iters,
            p_init=p_init,
            save_results=True,
            create_visualizations=True,
            out_dir="./square_roi_results"
        )
        
        # 통계에서 필요한 값들 추출
        ASR_roi = statistics_roi.get('ASR', 0.0)
        mean_L2_roi = statistics_roi.get('mean_l2', 0.0)
        mean_changed_pct_roi = statistics_roi.get('mean_changed_ratio', 0.0)
        mean_time_roi = statistics_roi.get('mean_time', 0.0)
        success_list_roi = [r['success'] for r in results_roi]
        
        # 9. 최종 결과 요약
        print("\n" + "="*70)
        print("Square Attack 최종 결과 요약")
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



