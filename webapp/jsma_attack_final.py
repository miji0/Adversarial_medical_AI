'''
Adversarial Attack2: jsma_fianl.py
**JSMA(Jacobian-based Saliency Map Attak) 기반의 momentum 강화 버전 알고리즘**
- 
'''

import torch
import torch.nn.functional as F


def to_pixel(x_norm, mean, std):
    """정규화된 이미지를 픽셀 공간([0, 1])으로 변환"""
    return torch.clamp(x_norm * std + mean, 0.0, 1.0)


def to_norm(x_pix, mean, std):
    """픽셀 공간 이미지를 정규화(JSMA는 픽셀 단위로 연산하지만, 모델은 정규화된 입력을 받기 때문.)"""
    return (torch.clamp(x_pix, 0.0, 1.0) - mean) / std

@torch.no_grad()
def _predict_logits(model, x):
    """로짓 예측 (gradient 없이)"""
    return model(x)


def _margin_and_grad(model, x, y_idx):
    """
    margin(x) = logit[y] - max_{j!=y} logit[j]
    반환: margin(float), grad(d margin / d x), pred_lbl(int)
    """
    with torch.enable_grad():
        x = x.clone().detach().requires_grad_(True)
        logits = model(x)  # foward. 한 번만 수행
        y = int(y_idx.item() if torch.is_tensor(y_idx) else y_idx)

        c = logits.size(1)
        if c == 2:
            j_star = 1 - y
        else:
            tmp = logits[0].clone()
            tmp[y] = -1e9
            j_star = int(tmp.argmax().item())

        margin_t = logits[0, y] - logits[0, j_star]
        grad = torch.autograd.grad(margin_t, x, retain_graph=False, create_graph=False, allow_unused=False)[0]

    pred_lbl = int(logits.detach().argmax(dim=1).item())
    return float(margin_t.item()), grad.detach(), pred_lbl


def jsma_attack_margin_mom(
    model,
    x,                # 공격 대상 이미지 (1,3,H,W) in [0,1]
    y_true,           # 정답 레이블 (1,)
    theta=0.08,       # 픽셀 변화 기본 step
    max_pixels_percentage=0.05,   # 변경 허용 픽셀 비율(공간 픽셀 기준)
    k_small=2,        # 초기 k
    k_big=12,         # 정체 시 k 확대
    patience=3,       # margin 개선 정체 허용 스텝
    momentum=0.75,    # gradient EMA 모멘텀
    restarts=4,       # 재시도 횟수
    topk_pool=5000,   # saliency 상위 후보 픽셀 pool 크기
    allowed_masks=None,   # ROI 마스크 (H,W) bool
    clamp=(0.0, 1.0),   # 픽셀 값 제한 범위
    early_success=True  # 조기 성공시 반복 종료
):
    """
    반환
    - x_best(1,3,H,W) : 공격 후 이미지
    - changed_spatial(int) : 변경 픽셀 수
    - l1_total(float) : 전체 L1 섭동량
    - success(0/1 tensor) : 공격 성공 여부
    """
    device = x.device
    _, C, H, W = x.shape
    budget = int(max_pixels_percentage * H * W)
    best = None  # (success, changed, -final_margin, x_adv, l1)

    if allowed_masks is not None:
        roi = allowed_masks.bool().to(device)
    else:
        roi = torch.ones((H, W), dtype=torch.bool, device=device)

    for _ in range(restarts):
        x_adv = x.clone().detach()
        v = torch.zeros_like(x_adv)           # momentum buffer
        changed_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        last_margins = []
        changed_spatial = 0

        step_theta = theta
        k = k_small
        success = False

        while changed_spatial < budget:
            # 현재 margin / grad / pred (중복 forward 제거)
            margin, g, pred_lbl = _margin_and_grad(model, x_adv, y_true)
            last_margins.append(margin)
            if len(last_margins) > patience + 1:
                last_margins.pop(0)

            # 조기 성공 시 종료
            if early_success and pred_lbl != int(y_true.item()):
                success = True
                break

            # momentum 업데이트 (EMA)
            v = momentum * v + (1.0 - momentum) * g

            # saliency(공간) = Σ_c |v_c|
            sal = v.abs().sum(dim=1, keepdim=False)[0]  # (H,W)

            # 업데이트 불가 위치 마스킹 - 이미 변경된 픽셀이거나 경계 (0,1) 도달한 픽셀 제외
            eligible = roi & (~changed_mask)
            with torch.no_grad():
                sign_v = v.sign()[0]  # (3,H,W)
                up_block   = (sign_v > 0) & (x_adv[0] <= clamp[0] + 1e-6)
                down_block = (sign_v < 0) & (x_adv[0] >= clamp[1] - 1e-6)
                blocked = up_block.any(dim=0) | down_block.any(dim=0)
                eligible = eligible & (~blocked)

            if not eligible.any():
                break

            # 마스킹 + topk (커널 호출 수↓)
            sal_masked = sal.masked_fill(~eligible, float('-inf'))
            flat = sal_masked.view(-1)
            k_pool = min(topk_pool, flat.numel())
            idx_pool = flat.topk(k_pool, largest=True).indices
            if idx_pool.numel() == 0 or torch.isneginf(flat[idx_pool[0]]):
                break

            k_eff = min(k, idx_pool.numel(), budget - changed_spatial)
            idx_pick = idx_pool[:k_eff] # 상위 k개의 픽셀만 변경 후보로
            ys = (idx_pick // W).long()
            xs = (idx_pick %  W).long()

            # 벡터화(픽셀) 업데이트 + 누적 카운트
            # gradient 부호 반대로 스텝만큼 이동
            with torch.no_grad():
                upd = step_theta * v[0, :, ys, xs].sign()
                x_adv[0, :, ys, xs] = (x_adv[0, :, ys, xs] - upd).clamp_(clamp[0], clamp[1])
                changed_mask[ys, xs] = True
                changed_spatial += k_eff

            # margin 정체시 k/θ 조절
            if len(last_margins) >= patience + 1:
                if last_margins[0] - last_margins[-1] < 1e-4:  # 개선 거의 없음
                    k = min(k_big, k * 2)
                    step_theta = min(step_theta * 1.25, 0.2)
                else:
                    k = max(k_small, k // 2)
                    step_theta = max(step_theta * 0.95, theta * 0.5)

        # 결과 집계
        with torch.no_grad():
            diff = (x_adv - x).abs()
            l1_total = float(diff.sum().item())
            # 최종 성공 판정
            pred_final = _predict_logits(model, x_adv).argmax(dim=1)
            success = success or (int(pred_final.item()) != int(y_true.item()))
            # 최종 margin(작을수록 좋음)
            fin_margin, _, _ = _margin_and_grad(model, x_adv, y_true)

        cand = (success, changed_spatial, -fin_margin, x_adv.detach(), l1_total)
        if best is None:
            best = cand
        else:
            if (cand[0] and not best[0]) \
               or (cand[0] and best[0] and cand[1] < best[1]) \
               or (cand[0] == best[0] and cand[1] == best[1] and cand[2] > best[2]):
                best = cand

    success, changed_spatial, _neg_margin, x_best, l1_total = best
    return x_best, torch.tensor(changed_spatial), torch.tensor(l1_total, dtype=torch.float32), torch.tensor(1 if success else 0)


def jsma_attack(model, x_pix, y_true, mean, std, theta=0.08, max_pixels_pct=0.05, 
                k_small=2, restarts=4, topk_pool=5000, roi_mask=None):
    """
    **JSMA 공격 - 웹 애플리케이션용 간소화 인터페이스**
    
    Args:
        model: 대상 모델
        x_pix: 공격 대상 이미지 (1,3,H,W) in [0,1]
        y_true: 정답 레이블 (1,)
        mean: 정규화 평균값
        std: 정규화 표준편차
        theta: 픽셀 변화 step 크기 (기본 0.08)
        max_pixels_pct: 최대 변경 픽셀 비율 (기본 0.05 = 5%)
        k_small: 초기 동시 변경 픽셀 수 (기본 2)
        restarts: 재시도 횟수 (기본 4)
        topk_pool: saliency 상위 후보 픽셀 pool 크기 (기본 5000)
        roi_mask: ROI 마스크 (H,W) bool, None이면 전체 영역

    Returns:
        x_adv_pix: 공격된 픽셀 공간 이미지
        changed_spatial: 변경된 픽셀 수
        l1_total: L1 섭동 총합
        success: 공격 성공 여부
    """

    # 픽셀 공간 이미지를 그대로 사용 (이미 [0,1] 범위)
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