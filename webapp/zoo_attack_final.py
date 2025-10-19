import time
import torch
import torch.nn.functional as F
import numpy as np


def to_pixel(x_norm, mean, std):
    """정규화된 이미지를 픽셀 공간으로 변환"""
    return torch.clamp(x_norm * std + mean, 0.0, 1.0)


def to_norm(x_pix, mean, std):
    """픽셀 공간 이미지를 정규화"""
    return (torch.clamp(x_pix, 0.0, 1.0) - mean) / std


@torch.no_grad()
def predict_logits(model, x):
    """로짓 예측 (gradient 없이)"""
    return model(x)


def zoo_total_loss_untargeted(model, x, x0, y_true, c, kappa=0.0):
    """
    ZOO Untargeted 공격을 위한 총 손실 함수
    
    Args:
        model: 대상 모델
        x: 현재 이미지 (B,3,H,W)
        x0: 원본 이미지 (1,3,H,W)
        y_true: 실제 레이블 (B,)
        c: 정규화 상수
        kappa: 마진 파라미터
    
    Returns:
        total_loss: 총 손실
        hinge_loss: 힌지 손실
        l2_loss: L2 손실
        logits: 예측 로짓
        margin: 마진 값
    """
    # L2 손실
    diff = x - x0
    l2_loss = (diff ** 2).sum(dim=(1, 2, 3))
    
    # 예측
    logits = model(x)
    
    # 마진 계산 (correct - max_other)
    correct_logit = logits.gather(1, y_true.unsqueeze(1))
    tmp = logits.clone()
    tmp.scatter_(1, y_true.unsqueeze(1), -1e9)
    max_other = tmp.max(dim=1)[0]
    margin = correct_logit.squeeze() - max_other
    
    # 힌지 손실 (margin이 kappa보다 클 때만 손실)
    hinge_loss = torch.clamp(margin + kappa, min=0.0)
    
    # 총 손실
    total_loss = c * hinge_loss + l2_loss
    
    return total_loss, hinge_loss, l2_loss, logits, margin


def adam_update(m, grad, mt, vt, idx, lr, beta1, beta2, epoch, up, down, project=True):
    """
    Adam 옵티마이저 업데이트
    
    Args:
        m: 현재 파라미터 벡터
        grad: 그래디언트
        mt: 1차 모멘텀
        vt: 2차 모멘텀
        idx: 업데이트할 인덱스
        lr: 학습률
        beta1, beta2: Adam 파라미터
        epoch: 에포크 카운터
        up, down: 상한/하한 경계
        project: 경계 투영 여부
    """
    for i, j in enumerate(idx):
        epoch[j] += 1
        
        # Adam 업데이트
        mt[j] = beta1 * mt[j] + (1 - beta1) * grad[i]
        vt[j] = beta2 * vt[j] + (1 - beta2) * (grad[i] ** 2)
        
        # 편향 보정
        mt_hat = mt[j] / (1 - beta1 ** epoch[j])
        vt_hat = vt[j] / (1 - beta2 ** epoch[j])
    
    # 파라미터 업데이트
        update = lr * mt_hat / (np.sqrt(vt_hat) + 1e-8)
        m[j] = m[j] - update
    
        # 경계 투영
    if project:
            m[j] = np.clip(m[j], down[j], up[j])
    

def zoo_attack_l2_adam_untargeted(
    model,
    x0_pix,                 # (1,3,H,W) in [0,1] - 원본 이미지
    src_idx_tensor,         # (1,) 현재 예측(원본의 source class)
    roi_mask=None,          # (1,3,H,W) ROI 마스크 (0 또는 1)
    max_iterations=320,     # 최대 반복 횟수
    batch_coords=256,       # 배치당 좌표 수
    step_eps=0.1,           # 유한 차분을 위한 스텝 크기
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
    """
    ZOO L2 공격 (Adam 옵티마이저, Untargeted, ROI 기능 포함)
    
    Args:
        model: 대상 모델
        x0_pix: 원본 픽셀 이미지 (1,3,H,W) [0,1]
        y_true: 실제 레이블
        mean: 정규화 평균값
        std: 정규화 표준편차
        roi_mask: ROI 마스크 (H,W) bool, None이면 전체 영역
        max_iterations: 최대 반복 횟수
        batch_coords: 배치당 좌표 수
        step_eps: 유한 차분 스텝 크기
        lr: Adam 학습률
        beta1, beta2: Adam 모멘텀 파라미터
        kappa: 마진 파라미터
        initial_const_c: 초기 정규화 상수
        binary_search_steps: 이진 탐색 단계
        print_every: 로그 출력 주기
        early_stop_window: 조기 종료 윈도우
        abort_early: 조기 종료 여부
        early_success: 초기 성공 체크
    
    Returns:
        x_adv_pix: 공격된 픽셀 공간 이미지
        n_queries: 쿼리 횟수
        l2_norm: L2 노름
        success: 공격 성공 여부
        margin: 최종 마진
        elapsed: 소요 시간
    """
    device = x0_pix.device
    _, C, H, W = x0_pix.shape
    N = C * H * W  # 총 픽셀 수
    h = step_eps   # 유한 차분 스텝

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
        roi_indices = np.where(roi_flat > 0.5)[0]  # ROI 영역
        
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

    # 이진 탐색을 위한 경계값
    lower_c, upper_c = 0.001, 1e6
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
        m = np.random.normal(0, 0.01, N).astype(np.float32)
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

            # 조기 종료 체크
            if L0 > prev_total * 0.999:
                no_improve += 1
                if abort_early and (no_improve >= max(1, int(early_stop_window))):
                    break
            else:
                no_improve = 0
                prev_total = L0

            # 성공 판정: margin <= 0.1
            if margin[0].item() <= 0.1:
                success_this_c = True
                if l2[0].item() < best_l2_this_c:
                    best_l2_this_c = l2[0].item()
                    # 최종 이미지 저장
                    m_best = m.copy()
                    m_best_tensor = torch.from_numpy(m_best.reshape(1,C,H,W)).to(device)
                    best_img = torch.clamp(x0_pix + m_best_tensor, 0.0, 1.0)

        # 이진 탐색 업데이트
        if success_this_c:
            upper_c = min(upper_c, CONST)
            if upper_c < 1e9:
                CONST = (lower_c + upper_c) / 2.0
        else:
            lower_c = max(lower_c, CONST)
            if upper_c < 1e9:
                CONST = (lower_c + upper_c) / 2.0
            else:
                CONST *= 10.0

    elapsed = time.time() - t0
    
    # 최종 성공 판정
    with torch.no_grad():
        final_logits = predict_logits(model, best_img)
        final_pred = final_logits.argmax(1).item()
        success = (final_pred != int(src_idx_tensor.item()))
        
        # 최종 마진 계산
        correct_logit = final_logits[0, int(src_idx_tensor.item())]
        tmp = final_logits[0].clone()
        tmp[int(src_idx_tensor.item())] = -1e9
        max_other = tmp.max()
        final_margin = float((correct_logit - max_other).item())
        
        # L2 노름 계산
        diff = (best_img - x0_pix).view(1, -1)
        l2_norm = float(diff.norm(p=2).item())

    return best_img, n_queries, l2_norm, success, final_margin, elapsed


def zoo_attack(model, x_pix, y_true, mean, std, max_iterations=320, batch_coords=256,
               step_eps=0.1, lr=2e-3, roi_mask=None):
    """
    ZOO 공격 - 웹 애플리케이션용 간소화 인터페이스
    
    Args:
        model: 대상 모델
        x_pix: 픽셀 공간 이미지 (1,3,H,W) [0,1]
        y_true: 실제 레이블
        mean: 정규화 평균값
        std: 정규화 표준편차
        max_iterations: 최대 반복 횟수
        batch_coords: 배치당 좌표 수
        step_eps: 유한 차분 스텝 크기
        lr: Adam 학습률
        roi_mask: ROI 마스크 (H,W) bool, None이면 전체 영역
    
    Returns:
        x_adv_pix: 공격된 픽셀 공간 이미지
        n_queries: 쿼리 횟수
        l2_norm: L2 노름
        success: 공격 성공 여부
    """
    # y_true를 텐서로 변환
    if not torch.is_tensor(y_true):
        y_true = torch.tensor([y_true], device=x_pix.device)
    elif y_true.dim() == 0:
        y_true = y_true.unsqueeze(0)
    
    x_adv, n_queries, l2_norm, success, margin, elapsed = zoo_attack_l2_adam_untargeted(
        model=model,
        x0_pix=x_pix,
        src_idx_tensor=y_true,
        roi_mask=roi_mask,
        max_iterations=max_iterations,
            batch_coords=batch_coords,
        step_eps=step_eps,
        lr=lr,
            beta1=0.9, 
            beta2=0.999,
            kappa=0.0, 
        initial_const_c=6.0,
            binary_search_steps=3, 
        print_every=100,
        early_stop_window=200,
            abort_early=True,
            early_success=True
        )

    return x_adv, n_queries, l2_norm, success