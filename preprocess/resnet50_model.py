"""
ResNet50 모델 정의 및 유틸 함수
- ImageNet 정규화 내장 (입력 [0,1] → 내부에서 정규화)
- 사전학습(backbone) 및 학습된 가중치 체크포인트 로드 지원

주요 함수/클래스:
  - ResNet50WithNorm: 내부적으로 ImageNet 정규화 포함한 ResNet50 (torchvision)
  - load_trained_model: 학습된 모델 가중치(pth) 불러오기
  - get_model_info: 체크포인트에서 클래스명 등 부가 정보 추출
"""

import torch
import torch.nn as nn
from torchvision import models

class ResNet50WithNorm(nn.Module):
    """ImageNet 정규화를 모델 내부에 포함한 ResNet50"""
    def __init__(self, num_classes=2):
        super().__init__()
        
        # ResNet50 백본
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # 마지막 FC 레이어 교체
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        # ImageNet 정규화 파라미터 (사전학습 가중치와 정확히 매칭)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x):
        # ImageNet 정규화 (사전학습 가중치 입력 분포와 매칭)
        x = (x - self.mean) / self.std
        x = self.backbone(x)
        return x

def load_trained_model(checkpoint_path, device='cpu'):
    """훈련된 모델 로드"""
    # 모델 인스턴스 생성
    model = ResNet50WithNorm(num_classes=2)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 체크포인트 구조에 따라 다르게 처리
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 직접 state_dict인 경우
            model.load_state_dict(checkpoint)
    else:
        # 체크포인트가 직접 state_dict인 경우
        model.load_state_dict(checkpoint)
    
    # 모델을 지정된 디바이스로 이동
    model = model.to(device)
    model.eval()  # 평가 모드로 설정
    
    return model

def get_model_info(checkpoint_path):
    """체크포인트에서 모델 정보 추출"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 체크포인트 구조에 따라 다르게 처리
    if isinstance(checkpoint, dict):
        # 기본값 설정
        info = {
            'class_names': checkpoint.get('class_names', ['notumor', 'tumor']),
            'mean': checkpoint.get('mean', [0.485, 0.456, 0.406]),
            'std': checkpoint.get('std', [0.229, 0.224, 0.225]),
            'input_size': checkpoint.get('input_size', [224, 224]),
            'backbone_name': checkpoint.get('backbone_name', 'ResNet50')
        }
    else:
        # 체크포인트가 직접 state_dict인 경우 기본값 반환
        info = {
            'class_names': ['notumor', 'tumor'],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'input_size': [224, 224],
            'backbone_name': 'ResNet50'
        }
    
    return info