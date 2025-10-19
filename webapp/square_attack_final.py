import time
import math
import torch


def to_pixel(x_norm, mean, std):
    """정규화된 이미지를 픽셀 공간으로 변환"""
    return torch.clamp(x_norm * std + mean, 0.0, 1.0)


def to_norm(x_pix, mean, std):
    """픽셀 공간 이미지를 정규화"""
    return (torch.clamp(x_pix, 0.0, 1.0) - mean) / std


@torch.no_grad()
def predict_logits(model, x_pix, mean, std):
    """픽셀 공간 이미지로부터 logits 예측"""
    return model(to_norm(x_pix, mean, std))


def margin_loss_untargeted(logits, y):
    """Untargeted 공격을 위한 margin loss 계산"""
    bs = logits.size(0)
    correct_logit = logits[torch.arange(bs, device=logits.device), y]
    tmp = logits.clone()
    tmp[torch.arange(bs, device=logits.device), y] = -1e9
    max_other = tmp.max(dim=1).values
    return correct_logit - max_other


def p_selection(p_init, it, n_iters):
    """Square Attack에서 사용하는 동적 p 값 선택"""
    it = int(it / n_iters * 10000)
    if 10 < it <= 50: p = p_init / 2
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
    """가우시안 유사 섭동 생성"""
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
    """정사각형 섭동 생성"""
    delta = torch.zeros(s, s, device=device)
    delta[:s//2] = pseudo_gaussian_pert_rectangles(s//2, s, device)
    delta[s//2:] = -pseudo_gaussian_pert_rectangles(s - s//2, s, device)
    delta = delta / (delta.pow(2).sum().sqrt() + 1e-12)
    if torch.rand(1, device=device).item() > 0.5:
        delta = delta.t()
    return delta


def square_attack_l2_single(model, x_pix, y_true, mean, std, eps=0.5, n_iters=2000, p_init=0.1, seed=0, roi_mask=None):
    """
    Square Attack (L2) - 단일 이미지
    
    Args:
        model: 대상 모델
        x_pix: 픽셀 공간 이미지
        y_true: 실제 레이블
        mean: 정규화 평균값
        std: 정규화 표준편차
        eps: L2 섭동 예산
        n_iters: 최대 반복 횟수
        p_init: 초기 p 값
        seed: 난수 시드
        roi_mask: ROI 마스크 (H,W) bool, None이면 전체 영역
    
    Returns:
        x_best: 최적 공격 이미지
        n_queries: 쿼리 횟수
        elapsed: 소요 시간
        margin_min: 최종 margin 값
    """
    torch.manual_seed(seed)
    device = x_pix.device
    _, c, h, w = x_pix.shape
    n_features = c * h * w

    # ROI 마스크 처리
    if roi_mask is not None:
        roi_mask_expanded = roi_mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        roi_mask_expanded = roi_mask_expanded.expand(1, c, -1, -1)  # (1,C,H,W)

    # 초기화
    delta_init = torch.zeros_like(x_pix)
    s = max(min(h, w) // 5, 3)
    if s % 2 == 0: s += 1
    for ch in range(0, h, s):
        for cw in range(0, w, s):
            hs = min(s, h - ch); ws = min(s, w - cw)
            if hs < 3 or ws < 3: continue
            
            # ROI 체크: 해당 영역에 ROI가 있는지 확인
            if roi_mask is not None:
                roi_region = roi_mask[ch:ch+hs, cw:cw+ws]
                if not roi_region.any():  # ROI 영역이 아니면 스킵
                    continue
            
            rect = pseudo_gaussian_pert_rectangles(hs, ws, device).view(1,1,hs,ws).repeat(1, c, 1, 1)
            sign = torch.randint(0, 2, (1, c, 1, 1), device=device).float() * 2 - 1
            delta_init[:, :, ch:ch+hs, cw:cw+ws] += rect * sign

    norm = delta_init.view(1, -1).norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    x_best = torch.clamp(x_pix + delta_init * (eps / norm.view(1,1,1,1)), 0.0, 1.0)

    logits = predict_logits(model, x_best, mean, std)
    margin_min = margin_loss_untargeted(logits, y_true)
    loss_min = margin_min

    n_queries = 1
    time_start = time.time()

    for i in range(n_iters):
        if (margin_min <= 0).all():
            break

        x_curr = x_pix.clone()
        x_best_ = x_best.clone()
        delta = (x_best_ - x_curr)

        p = p_selection(p_init, i, n_iters)
        s = max(int(round(math.sqrt(p * n_features / c))), 3)
        s = min(s, min(h, w) - 1)
        if s % 2 == 0: s += 1
        s2 = s
        if h - s <= 0 or w - s <= 0: break

        # ROI 제약이 있을 때는 ROI 영역 내에서만 선택
        if roi_mask is not None:
            # ROI 영역에서 가능한 위치들 찾기
            roi_coords = torch.nonzero(roi_mask, as_tuple=False)
            if len(roi_coords) == 0:
                break  # ROI 영역이 없으면 중단
            
            # ROI 영역 내에서 랜덤 선택
            valid_coords = []
            for coord in roi_coords:
                y, x = coord[0].item(), coord[1].item()
                if x + s <= w and y + s <= h:
                    valid_coords.append((y, x))
            
            if len(valid_coords) == 0:
                break  # 유효한 좌표가 없으면 중단
            
            coord = valid_coords[torch.randint(0, len(valid_coords), (1,)).item()]
            ch, cw = coord
            
            # 두 번째 좌표도 ROI 영역에서 선택
            valid_coords2 = []
            for coord in roi_coords:
                y, x = coord[0].item(), coord[1].item()
                if x + s2 <= w and y + s2 <= h:
                    valid_coords2.append((y, x))
            
            if len(valid_coords2) == 0:
                ch2, cw2 = ch, cw  # 대체 좌표 사용
            else:
                coord2 = valid_coords2[torch.randint(0, len(valid_coords2), (1,)).item()]
                ch2, cw2 = coord2
        else:
            ch = torch.randint(0, h - s + 1, (1,), device=device).item()
            cw = torch.randint(0, w - s + 1, (1,), device=device).item()
            ch2 = torch.randint(0, h - s2 + 1, (1,), device=device).item()
            cw2 = torch.randint(0, w - s2 + 1, (1,), device=device).item()

        mask1 = torch.zeros_like(x_curr); mask1[:, :, ch:ch+s, cw:cw+s] = 1.0
        mask2 = torch.zeros_like(x_curr); mask2[:, :, ch2:ch2+s2, cw2:cw2+s2] = 1.0
        mask_union = torch.maximum(mask1, mask2)

        curr_norm_window = ((x_best_ - x_curr) * mask1).view(1, -1).norm(p=2, dim=1, keepdim=True).view(1,1,1,1)
        curr_norm_img = (x_best_ - x_curr).view(1, -1).norm(p=2, dim=1, keepdim=True).view(1,1,1,1)
        norms_windows = (delta * mask_union).view(1, -1).norm(p=2, dim=1, keepdim=True).view(1,1,1,1)

        new_deltas = meta_pseudo_gaussian_pert_square(s, device).view(1,1,s,s).repeat(1, c, 1, 1)
        sign = torch.randint(0, 2, (1, c, 1, 1), device=device).float() * 2 - 1
        new_deltas = new_deltas * sign
        old_deltas = delta[:, :, ch:ch+s, cw:cw+s] / (curr_norm_window + 1e-10)
        new_deltas = new_deltas + old_deltas

        scale = ((torch.clamp(eps**2 - curr_norm_img**2, min=0.0) / c) + norms_windows**2).sqrt()
        new_deltas = new_deltas / (new_deltas.view(1, -1).norm(p=2, dim=1, keepdim=True).view(1,1,1,1) + 1e-12) * scale

        delta[:, :, ch2:ch2+s2, cw2:cw2+s2] = 0.0
        delta[:, :, ch:ch+s, cw:cw+s] = new_deltas

        new_norm = delta.view(1, -1).norm(p=2, dim=1, keepdim=True).view(1,1,1,1).clamp(min=1e-12)
        x_new = torch.clamp(x_curr + delta * (eps / new_norm), 0.0, 1.0)

        logits_new = predict_logits(model, x_new, mean, std)
        margin_new = margin_loss_untargeted(logits_new, y_true)
        loss_new = margin_new

        if (loss_new < loss_min).item():
            x_best = x_new
            loss_min = loss_new
            margin_min = margin_new

        n_queries += 1

    elapsed = time.time() - time_start
    return x_best, n_queries, elapsed, margin_min.item()
