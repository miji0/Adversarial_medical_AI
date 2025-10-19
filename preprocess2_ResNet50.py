'''
preprocess2_ResNet50.py
--------------------------------------
-  3채널 뇌종양 데이터셋 로드 → ResNet50 학습 → Binary 분류 (정규화 포함)

**기능 요약**
  1. 데이터셋 클래스 (BrainTumorDataset) - npy 3채널 변환, 라벨/마스크 지원
  2. 학습/검증/테스트 데이터로더 빌드 (split, seed 고정)
  3. ResNet50(backbone) + 정규화 래퍼 로드 및 최적화 세팅
  4. 학습/검증/평가 루프 (AMP 지원, 체크포인트 관리)
  5. 상세 평가(Confusion matrix, 메트릭, 시각화)
  6. Config 관리 및 진입점

** 생성되는 폴더 구조 **
Adversarial_mdedical_AI/
├── models/
│   ├── resnet50_binary_best.pth      # 최고 정확도 모델
│   └── resnet50_binary_final.pth     # 최종 epoch 모델
│   ├── training.log                   # 학습 로그 (Epoch별 Loss/Acc 기록)
│   ├── training_curves.png            # 학습/검증 Loss & Accuracy 곡선 시각화
└── confusion_matrix.png              # 테스트셋 성능 시각화

'''

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import json
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# =======================
# Colab 환경 자동 감지 및 설정
# =======================
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
    
    # resnet50_model.py 임포트를 위해 sys.path에 작업 디렉토리 추가
    if work_dir not in sys.path:
        sys.path.insert(0, work_dir)
        print(f"[INFO] {work_dir}를 sys.path에 추가.")
    
except ImportError:
    IN_COLAB = False
    print("[INFO] 로컬 환경에서 실행")

# =======================
# Utils (시드 고정 함수)
# =======================
def seed_everything(seed: int = 42):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_dir="models"):
    """
    로깅 설정: 콘솔과 파일에 동시 출력
    로그 파일: {log_dir}/training.log
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"
    
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

# =================================================
# 1. BrainTumorDataset: 1채널 -> 3채널 변환 및 로드
# =================================================
class BrainTumorDataset(Dataset):
    """뇌종양 이미지 데이터셋(np.ndarray 기반, 3채널 복제, 라벨/마스크 동반)"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # npy 파일들 로드 (이미지: float32, 라벨: int64, 마스크: bool)
        self.images = np.load(self.data_dir / "images.npy")  # (N, 224, 224, 1) [0,1]
        self.labels = np.load(self.data_dir / "labels.npy")  # (N,) int64
        self.masks = np.load(self.data_dir / "masks.npy")    # (N, 224, 224) bool

        # 클래스 이름 목록 로드
        with open(self.data_dir / "class_names.txt", "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # 메타데이터 로드 (ex: 클래스 비율 등)
        with open(self.data_dir / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        logging.info(f"데이터 로드 완료: {len(self.images)}개 이미지 from {self.data_dir}")
        logging.info(f"클래스: {self.class_names}")
        logging.info(f"메타 정보: {self.meta}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1채널 흑백 이미지를 3채널로 복제
        img = self.images[idx]              # (224, 224, 1) [0,1]
        img = np.repeat(img, 3, axis=2)     # (224, 224, 3) [0,1]
        label = self.labels[idx]
        mask = self.masks[idx]

        # 필요 시 transform 적용 (예: ToTensor, 증강 등)
        if self.transform:
            img = self.transform(img)

        return img, label, mask

# =====================================================
# 2. 데이터 분할 및 DataLoader 빌드 (train/val/test 분할)
# =====================================================
def build_loaders(train_dir, test_dir, batch_size=32, val_ratio=0.2, num_workers=0):
    """
    train_dir에서 train/val 데이터셋 분할, test_dir 전체를 test set으로 사용
    Training, Testing 폴더가 구분되어 있을 때 사용
    """
    # --- 훈련 데이터용 이미지 변환(증강 없음, ToTensor만) ---
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # (정규화는 모델 내부에서 수행)
    ])
    # --- 테스트용 변환 ---
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # (정규화는 모델 내부에서 수행)
    ])

    # --- Dataset 객체 생성 ---
    full_train = BrainTumorDataset(train_dir, transform=train_transform)
    test_set = BrainTumorDataset(test_dir, transform=test_transform)

    # --- 훈련셋 크기 체크 ---
    if len(full_train) < 2:
        raise ValueError(f"Training dataset too small: {len(full_train)} samples")

    n_val = max(1, int(len(full_train) * val_ratio))
    n_train = len(full_train) - n_val

    # --- 최소 분할 보장 ---
    if n_train < 1:
        logging.warning("Training set too small, adjusting val_ratio")
        n_val = 1
        n_train = len(full_train) - 1
        val_ratio = n_val / len(full_train)
        logging.warning(f"Adjusted val_ratio to {val_ratio:.3f}")

    # --- random_split 시드 고정 (재현성 확보) ---
    generator = torch.Generator()
    generator.manual_seed(42)

    train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)

    # --- pin_memory: CUDA 사용시 성능 향상 ---
    pin_memory = torch.cuda.is_available()

    # --- DataLoader 생성 (배치 단위 데이터 로딩) ---
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, full_train.class_names

# ========================================================
# 3. ResNet50 모델 + 입력 정규화 래퍼 정의 및 생성 함수
# ========================================================
def build_model(num_classes=2, lr=3e-4, weight_decay=1e-4):
    """
    모델 및 optimizer/criterion 생성 함수
    (ResNet50WithNorm: 정규화 내장, AMP/AdamW 지원)
    """
    # resnet50_model.py에서 래퍼 import (내부적으로 입력 정규화까지 처리)
    from resnet50_model import ResNet50WithNorm
    
    model = ResNet50WithNorm(num_classes=num_classes)

    # AdamW 옵티마이저 사용 (L2 페널티 포함)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

# =========================================================
# 4. 학습/검증 (train_one_epoch, evaluate) AMP 지원
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    """한 epoch의 모델 학습(AMP 지원), loss와 정확도 반환"""
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for batch_idx, batch in enumerate(loader):
        # DataLoader가 (img, label, mask) 튜플을 반환하므로 mask는 무시
        if len(batch) == 2:
            x, y = batch
        else:
            x, y = batch[0], batch[1]

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        # ----- Automatic Mixed Precision (AMP) 지원 -----
        with autocast(device.type, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 누적 손실/정확도 계산
        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    """모델 평가 (loss, acc, 예측값, 정답, 확률 반환)"""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        # (img, label, mask) 형식 -> mask는 무시
        if len(batch) == 2:
            x, y = batch
        else:
            x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)

        with autocast(device.type, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)

        # 누적 메트릭
        loss_sum += loss.item() * x.size(0)
        probs = torch.softmax(out, dim=1)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

        # 상세 평가를 위한 값 저장
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # tumor 클래스 확률 (1번 클래스)

    return (loss_sum / total, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))

# ========================================================
# 5. 평가 메트릭 계산 및 Confusion Matrix/ROC-AUC 지원
# ========================================================
def calculate_metrics(y_true, y_pred, y_probs, class_names):
    """
    precision/recall/f1, confusion matrix, ROC-AUC 등 평가 지표 계산
      - 이진 분류 기준으로 tumor를 positive로 간주
    """
    # tumor 클래스 인덱스 자동 선택 (ex: ["normal", "tumor"])
    tumor_idx = 1 if len(class_names) == 2 and 'tumor' in class_names[1].lower() else 1

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=tumor_idx
    )

    cm = confusion_matrix(y_true, y_pred)

    # ROC-AUC: 여러 클래스 등장시에만 계산; 그렇지 않으면 None 반환
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_probs)
        else:
            roc_auc = None
            logging.warning(f"ROC-AUC 계산 불가 (단일 클래스: {np.unique(y_true)})")
    except ValueError as e:
        roc_auc = None
        logging.warning(f"ROC-AUC 계산 오류: {e}")

    # 평가 결과 및 시각화(Confusion Matrix)
    logging.info("=" * 50)
    logging.info("상세 평가 결과")
    logging.info("=" * 50)
    logging.info(f"Precision (tumor): {precision:.4f}")
    logging.info(f"Recall (tumor): {recall:.4f}")
    logging.info(f"F1-Score (tumor): {f1:.4f}")
    if roc_auc is not None:
        logging.info(f"ROC-AUC: {roc_auc:.4f}")
    else:
        logging.info("ROC-AUC: 계산 불가")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path="training_curves.png"):
    """
    학습 곡선 그래프 생성 및 저장
    - 상단: Loss 곡선 (Train vs Val)
    - 하단: Accuracy 곡선 (Train vs Val)
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    epochs = range(1, len(train_losses) + 1)
    
    # Loss 그래프
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 그래프
    axes[1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"학습 곡선 그래프 저장: {save_path}")
    plt.close()

# ===========================================
# 6. Checkpoint Management (저장/복원)
# ===========================================
def save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc,
                   class_names, mean, std, input_size, backbone_name,
                   filepath):
    """
    모델, optimizer, scheduler 등 학습 상태 체크포인트 저장
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'class_names': class_names,
        'mean': mean.cpu().tolist() if torch.is_tensor(mean) else mean,
        'std': std.cpu().tolist() if torch.is_tensor(std) else std,
        'input_size': input_size,
        'backbone_name': backbone_name
    }
    torch.save(checkpoint, filepath)
    logging.info(f"체크포인트 저장: {filepath}")

def load_checkpoint(filepath, model, optimizer, scheduler, device):
    """
    저장된 체크포인트 로드 (모델 및 옵티마이저, 스케줄러 상태 복원)
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    logging.info(f"체크포인트 로드: {filepath}")
    logging.info(f"최고 검증 정확도: {checkpoint['best_val_acc']:.4f}")
    logging.info(f"클래스: {checkpoint['class_names']}")

    return checkpoint

# ========================================================================
# Main Training Loop (훈련-검증-테스트 전체 파이프라인)
# ========================================================================
def main(config):
    """메인 훈련 루프: 시드 설정, 디바이스 할당, 데이터로더, 모델, 학습, 평가, 모델 저장"""
    # --------- 로깅 설정 ---------
    setup_logging(config['out_dir'])
    
    # --------- 시드 고정 ---------
    seed_everything(config['seed'])

    # --------- 디바이스 선택 (우선순위: config -> GPU 가용성) ---------
    if config.get('device') == 'cpu':
        device = torch.device("cpu")
        logging.info("강제로 CPU 사용")
    elif config.get('device') == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info("강제로 CUDA 사용")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA 사용 불가, CPU로 대체")
    else:  # 'auto' 또는 기본값 : GPU 우선
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"자동 디바이스 선택: {device}")

    # --------- AMP 사용여부 (GPU일 때만 True) ---------
    use_amp = (device.type == "cuda")
    logging.info(f"AMP 사용: {use_amp}")

    # --------- 데이터 로더 생성 (훈련/밸리데이션/테스트) ---------
    train_loader, val_loader, test_loader, class_names = build_loaders(
        config['train_dir'], config['test_dir'],
        batch_size=config['batch_size'],
        val_ratio=config['val_ratio'],
        num_workers=config['num_workers']
    )

    logging.info(f"Dataset split - Train: {len(train_loader.dataset)}, "
                 f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # --------- 모델, optimizer, loss 함수 세팅 ---------
    model, optimizer, criterion = build_model(
        num_classes=len(class_names),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    model = model.to(device)

    # --------- 러닝레이트 스케줄러 ---------
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # --------- AMP 스케일러 ---------
    scaler = GradScaler(device.type, enabled=use_amp)

    # --------- 체크포인트 경로 준비 (best/final) ---------
    out_dir = Path(config['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "resnet50_binary_best.pth"
    final_path = out_dir / "resnet50_binary_final.pth"

    # --------- 학습 곡선 기록을 위한 리스트 ---------
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    # --------- 훈련 루프 진입 ---------
    best_val_acc = 0.0
    logging.info(f"학습 시작 - 총 {config['epochs']} epochs")
    
    for epoch in range(1, config['epochs'] + 1):
        # 1. 한 epoch 학습
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, use_amp
        )
        # 2. 검증
        va_loss, va_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device, use_amp
        )
        # 3. 스케줄러 스텝 실행 (lr 갱신)
        scheduler.step()

        # 4. 메트릭 저장
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(va_loss)
        val_accs.append(va_acc)

        # 5. 로그 출력
        current_lr = scheduler.get_last_lr()[0]
        logging.info(f"Epoch {epoch:02d}/{config['epochs']} | "
                     f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | "
                     f"Val Loss: {va_loss:.4f}, Acc: {va_acc:.4f} | "
                     f"LR: {current_lr:.6f}")

        # 6. 최고 검증정확도 모델 저장
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_acc,
                class_names, model.mean, model.std, (224, 224), "ResNet50",
                best_path
            )
            logging.info(f"새로운 최고 검증 정확도: {best_val_acc:.4f}")

    # --------- 학습 곡선 그래프 생성 ---------
    plot_training_curves(
        train_losses, train_accs, val_losses, val_accs,
        save_path=str(out_dir / "training_curves.png")
    )

    # --------- 최고 검증 성능 모델 불러와 테스트 ---------
    if best_path.exists():
        checkpoint = load_checkpoint(best_path, model, optimizer, scheduler, device)
        logging.info("최고 성능 모델 로드 완료 - 테스트 시작")

    # --------- 테스트셋 평가 ---------
    te_loss, te_acc, te_preds, te_labels, te_probs = evaluate(
        model, test_loader, criterion, device, use_amp
    )

    logging.info(f"테스트 결과 - Loss: {te_loss:.4f}, Accuracy: {te_acc:.4f}")

    # --------- 상세 메트릭 및 Confusion Matrix 시각화 ---------
    metrics = calculate_metrics(te_labels, te_preds, te_probs, class_names)

    # --------- 최종 모델 저장 ---------
    save_checkpoint(
        model, optimizer, scheduler, config['epochs'], best_val_acc,
        class_names, model.mean, model.std, (224, 224), "ResNet50",
        final_path
    )

    logging.info("=" * 60)
    logging.info("학습 완료!")
    logging.info(f"최고 검증 정확도: {best_val_acc:.4f}")
    logging.info(f"테스트 정확도: {te_acc:.4f}")
    logging.info(f"Precision: {metrics['precision']:.4f}")
    logging.info(f"Recall: {metrics['recall']:.4f}")
    logging.info(f"F1-Score: {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        logging.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    logging.info("=" * 60)

# =================================================
# 7. 설정값 Config: 경로, 하이퍼파라미터 등 딕셔너리화
# =================================================
def get_default_config():
    """기본 설정 반환(경로/하이퍼파라미터 등, 필요시 수정하여 사용)"""
    return {
        'train_dir': "processed_data_np224/Training",  # Train 데이터 경로(필요시 변경)
        'test_dir': "processed_data_np224/Testing",    # Test 데이터 경로(필요시 변경)
        'out_dir': "models",                           # 모델, 체크포인트 저장 폴더
        'batch_size': 32,
        'epochs': 50,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'val_ratio': 0.2,
        'num_workers': 0,
        'seed': 42,
        'device': 'auto'  # 'auto', 'cuda', 'cpu' 중 선택 가능
    }

# =================================================
# 8. Entrypoint (main)
# =================================================
if __name__ == "__main__":
    ## 선택1, 2 중 하나만 선택 후 나머지는 주석 처리 후 실행

    # (선택1): Colab에서 실행할 때 경로 직접 지정
    # config = get_default_config()
    # config['train_dir'] = "/content/drive/MyDrive/Adversarial_medical_AI/processed_data_np224/Training"
    # config['test_dir'] = "/content/drive/MyDrive/Adversarial_medical_AI/processed_data_np224/Testing"
    # config['out_dir'] = "/content/drive/MyDrive/Adversarial_medical_AI/models"
    # main(config)

    # (선택2): 로컬 환경에서 기본 경로 사용
    config = get_default_config()
    main(config)
