import torch
import torch.nn as nn


def to_pixel(x_norm, mean, std):
    """정규화된 이미지를 픽셀 공간으로 변환"""
    return torch.clamp(x_norm * std + mean, 0.0, 1.0)


def to_norm(x_pix, mean, std):
    """픽셀 공간 이미지를 정규화"""
    return (torch.clamp(x_pix, 0.0, 1.0) - mean) / std


def fgsm_attack_pixel_space(model, x_norm, y, mean, std, eps=0.01, roi_mask=None):
    """
    FGSM (Fast Gradient Sign Method) 공격
    
    Args:
        model: 대상 모델
        x_norm: 정규화된 입력 이미지
        y: 타겟 레이블
        mean: 정규화 평균값
        std: 정규화 표준편차
        eps: 섭동 크기
        roi_mask: ROI 마스크 (H,W) bool, None이면 전체 영역
    
    Returns:
        x_pix: 원본 픽셀 이미지
        x_pix_adv: 공격된 픽셀 이미지
        x_adv_norm: 공격된 정규화 이미지
    """
    x_norm = x_norm.clone().detach().requires_grad_(True)
    logits = model(x_norm)
    loss = nn.CrossEntropyLoss()(logits, y)
    model.zero_grad()
    loss.backward()
    grad_norm = x_norm.grad.detach()
    
    # ROI 마스크 적용
    if roi_mask is not None:
        # ROI 마스크를 (1,3,H,W) 형태로 확장
        roi_mask_expanded = roi_mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        roi_mask_expanded = roi_mask_expanded.expand(1, 3, -1, -1)  # (1,3,H,W)
        grad_norm = grad_norm * roi_mask_expanded.float()
    
    # 픽셀 공간으로 이동
    x_pix = x_norm * std + mean
    # 정규화된 입력에 대한 그래디언트의 부호는 픽셀 입력에 대한 부호와 비례
    x_pix_adv = torch.clamp(x_pix + eps * grad_norm.sign(), 0.0, 1.0)
    # 모델을 위해 다시 정규화
    x_adv_norm = (x_pix_adv - mean) / std
    return x_pix, x_pix_adv, x_adv_norm
