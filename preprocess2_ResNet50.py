'''
preprocess2_ResNet50.py
--------------------------------------
-  3채널 뇌종양 데이터셋 로드 → ResNet50 학습 → Binary 분류 (정규화 포함)

**기능 요약**
  1. 데이터셋 클래스 (BrainTumorDataset) - npy 3채널 변환, 라벨/마스크 지원
  2. 학습/검증/테스트 데이터로더 빌드 (split, seed 고정)
  3. ResNet50(backbone)+정규화 래퍼 로드 및 최적화 세팅
  4. 학습/검증/평가 루프 (AMP 지원, 체크포인트 관리)
  5. 상세 평가(Confusion matrix, 메트릭, 시각화)
  6. Config 관리 및 진입점

** 생성되는 폴더 구조 **
Adversarial_AI/
├── models/
│   ├── resnet50_binary_best.pth      # Best validation accuracy checkpoint
│   └── resnet50_binary_final.pth     # Final epoch checkpoint
└── confusion_matrix.png              # Confusion matrix
'''

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import json
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

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

# =================================================
# 1. BrainTumorDataset: 1채널 -> 3채널 변환 및 로드
# =================================================
class BrainTumorDataset(Dataset):
    """뇌종양 이미지 데이터셋(np.ndarray 기반, 3채널 복제, 라벨/마스크 동반)"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # npy 파일들 로드 (float32, int64, bool)
        self.images = np.load(self.data_dir / "images.npy")  # (N, 224, 224, 1) [0,1]
        self.labels = np.load(self.data_dir / "labels.npy")  # (N,) int64
        self.masks = np.load(self.data_dir / "masks.npy")    # (N, 224, 224) bool

        # 클래스명 (텍스트)
        with open(self.data_dir / "class_names.txt", "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # 메타데이터 (예: 클래스 비율 등)
        with open(self.data_dir / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        print(f"[INFO] Loaded {len(self.images)} images from {self.data_dir}")
        print(f"[INFO] Classes: {self.class_names}")
        print(f"[INFO] Meta: {self.meta}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1채널 흑백 이미지 -> 3채널 복제
        img = self.images[idx]              # (224, 224, 1) [0,1]
        img = np.repeat(img, 3, axis=2)     # (224, 224, 3) [0,1]
        label = self.labels[idx]
        mask = self.masks[idx]

        # transform (예: ToTensor, 기타 증강)
        if self.transform:
            img = self.transform(img)

        return img, label, mask

# =====================================================
# 2. 데이터 분할 및 DataLoader 빌드 (train/val/test 분할)
# =====================================================
def build_loaders(train_dir, test_dir, batch_size=32, val_ratio=0.2, num_workers=0):
    """
    train:train_dir에서 -> (train/val 분할)
    val:train_dir에서 -> 분할
    test:test_dir 전체

    Training, Testing 폴더가 따로 있을 때만 사용
    """
    # --- 훈련용 증강 transform ---
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # (정규화는 모델 내부에서 수행)
    ])
    # --- 테스트용 transform (증강X) ---
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # (정규화는 모델 내부에서 수행)
    ])

    # --- 데이터셋 생성 ---
    full_train = BrainTumorDataset(train_dir, transform=train_transform)
    test_set = BrainTumorDataset(test_dir, transform=test_transform)

    # --- 데이터셋 분할 크기 검증 ---
    if len(full_train) < 2:
        raise ValueError(f"Training dataset too small: {len(full_train)} samples")

    n_val = max(1, int(len(full_train) * val_ratio))
    n_train = len(full_train) - n_val

    # --- 분할 최소 크기 보장 ---
    if n_train < 1:
        print(f"[WARNING] Training set too small, adjusting val_ratio")
        n_val = 1
        n_train = len(full_train) - 1
        val_ratio = n_val / len(full_train)
        print(f"[WARNING] Adjusted val_ratio to {val_ratio:.3f}")

    # --- random_split 시드 고정 (재현성 위함) ---
    generator = torch.Generator()
    generator.manual_seed(42)

    train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)

    # --- pin_memory: CUDA 사용시 True ---
    pin_memory = torch.cuda.is_available()

    # --- DataLoader 생성 ---
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
from resnet50_model import ResNet50WithNorm  # 모델 래퍼 (정규화 내장형)

def build_model(num_classes=2, lr=3e-4, weight_decay=1e-4):
    """
    모델 및 optimizer/criterion 생성 함수
    (ResNet50WithNorm: 정규화 내장, AMP/AdamW 지원)
    """
    model = ResNet50WithNorm(num_classes=num_classes)

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
        # img, label, (mask는 사용 안 함) - 튜플 타입 대응
        if len(batch) == 2:
            x, y = batch
        else:
            x, y = batch[0], batch[1]

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        # ----- Automatic Mixed Precision -----
        with autocast(enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 메트릭 계산
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
        if len(batch) == 2:
            x, y = batch
        else:
            x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)

        with autocast(enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)

        # 메트릭 계산
        loss_sum += loss.item() * x.size(0)
        probs = torch.softmax(out, dim=1)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

        # 상세 평가를 위한 데이터 수집
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
    precision/recall/f1 및 confusion matrix, ROC-AUC 계산
      - binary classification: tumor를 positive로 처리
    """
    # tumor의 인덱스 자동 탐지(예: ["normal", "tumor"])
    tumor_idx = 1 if len(class_names) == 2 and 'tumor' in class_names[1].lower() else 1

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=tumor_idx
    )

    cm = confusion_matrix(y_true, y_pred)

    # ROC-AUC(값이 둘 다 있을때만), 아니면 None
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_probs)
        else:
            roc_auc = None
            print(f"[WARNING] ROC-AUC 계산 불가 (단일 클래스: {np.unique(y_true)})")
    except ValueError as e:
        roc_auc = None
        print(f"[WARNING] ROC-AUC 계산 오류: {e}")

    # 상세 결과 출력 및 시각화 저장
    print(f"\n{'='*50}")
    print("상세 평가 결과")
    print(f"{'='*50}")
    print(f"Precision (tumor): {precision:.4f}")
    print(f"Recall (tumor): {recall:.4f}")
    print(f"F1-Score (tumor): {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.4f}")
    else:
        print("ROC-AUC: 계산 불가")

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
    print(f"[INFO] Checkpoint saved: {filepath}")

def load_checkpoint(filepath, model, optimizer, scheduler, device):
    """
    저장된 체크포인트 로드 (모델/옵티마이저/스케줄러 상태)
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"[INFO] Checkpoint loaded: {filepath}")
    print(f"[INFO] Best val acc: {checkpoint['best_val_acc']:.4f}")
    print(f"[INFO] Classes: {checkpoint['class_names']}")

    return checkpoint

# ========================================================================
# Main Training Loop (훈련-검증-테스트 전체 파이프라인)
# ========================================================================
def main(config):
    """메인 훈련 루프: 시드, 디바이스, 로더, 모델, 학습, 평가, 저장"""
    # --------- 시드 고정 ---------
    seed_everything(config['seed'])

    # --------- 디바이스 선택 ---------
    if config.get('device') == 'cpu':
        device = torch.device("cpu")
        print("[INFO] 강제로 CPU 사용")
    elif config.get('device') == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("[INFO] 강제로 CUDA 사용")
        else:
            device = torch.device("cpu")
            print("[WARNING] CUDA 사용 불가, CPU로 대체")
    else:  # 'auto' 또는 기본값
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] 자동 디바이스 선택: {device}")

    # --------- AMP 사용 여부 판단 ---------
    use_amp = (device.type == "cuda")
    print(f"[INFO] AMP 사용: {use_amp}")

    # --------- 데이터 로더 생성 ---------
    train_loader, val_loader, test_loader, class_names = build_loaders(
        config['train_dir'], config['test_dir'],
        batch_size=config['batch_size'],
        val_ratio=config['val_ratio'],
        num_workers=config['num_workers']
    )

    print(f"[INFO] Dataset split - Train: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # --------- 모델, optimizer 등 생성 ---------
    model, optimizer, criterion = build_model(
        num_classes=len(class_names),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    model = model.to(device)

    # --------- 스케줄러 ---------
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # --------- AMP 스케일러 ---------
    scaler = GradScaler(enabled=use_amp)

    # --------- 체크포인트 경로 세팅 ---------
    out_dir = Path(config['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "resnet50_binary_best.pth"
    final_path = out_dir / "resnet50_binary_final.pth"

    # --------- 훈련 루프 시작 ---------
    best_val_acc = 0.0
    for epoch in range(1, config['epochs'] + 1):
        # 1. 학습
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, use_amp
        )
        # 2. 검증
        va_loss, va_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device, use_amp
        )
        # 3. 스케줄러 스텝(매 에폭)
        scheduler.step()

        # 4. 상태 출력
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{config['epochs']} | "
              f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"Val {va_loss:.4f}/{va_acc:.4f} | "
              f"LR {current_lr:.6f}")

        # 5. 최고 검증 정확도 모델 저장
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_acc,
                class_names, model.mean, model.std, (224, 224), "ResNet50",
                best_path
            )

    # --------- 최고 모델 로드 & 테스트 ---------
    if best_path.exists():
        checkpoint = load_checkpoint(best_path, model, optimizer, scheduler, device)
        print(f"[INFO] Loaded best weights for testing")

    # --------- 테스트 평가 ---------
    te_loss, te_acc, te_preds, te_labels, te_probs = evaluate(
        model, test_loader, criterion, device, use_amp
    )

    print(f"\n[TEST] Loss: {te_loss:.4f} | Accuracy: {te_acc:.4f}")

    # --------- 상세 메트릭 및 confusion matrix 출력 ---------
    metrics = calculate_metrics(te_labels, te_preds, te_probs, class_names)

    # --------- 마지막 모델 저장(최종) ---------
    save_checkpoint(
        model, optimizer, scheduler, config['epochs'], best_val_acc,
        class_names, model.mean, model.std, (224, 224), "ResNet50",
        final_path
    )

    print(f"\n[SUCCESS] Training completed!")
    print(f"[SUCCESS] Best val acc: {best_val_acc:.4f}")
    print(f"[SUCCESS] Test acc: {te_acc:.4f}")

# =================================================
# 7. 설정값 Config: 경로, 하이퍼파라미터 등 딕셔너리화
# =================================================
def get_default_config():
    """기본 설정 반환(수정해서 변형 사용 가능)"""
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
    config = get_default_config()
    main(config)
