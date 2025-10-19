# Adversarial Attacks 모듈

`integrated.py`에서 분리된 공격별 모듈입니다.

## 파일 구조

```
attacks/
├── README.md                      # 이 파일
├── attack_square_l2.py           # Square Attack L2 (ROI 버전)
├── attack_fgsm_full.py           # FGSM Full Image
├── attack_fgsm_roi.py            # FGSM ROI
├── attack_jsma.py                # JSMA
└── attack_jsma_jpeg_defense.py   # JSMA + JPEG Defense 평가
```

## 공격 모듈 설명

### 1. Square Attack L2 (`attack_square_l2.py`)
- **설명**: ROI 영역 내에서만 L2 제약 공격 수행
- **특징**: 
  - Margin 기반 공격
  - 동적 p 선택 스케줄
  - 가우시안 타일 초기화
- **주요 함수**:
  - `square_attack_l2_single_roi()`: 단일 샘플 공격
  - `run_square_attack_batch()`: 배치 실행

### 2. FGSM Full Image (`attack_fgsm_full.py`)
- **설명**: 전체 이미지에 대해 FGSM 적용 (ROI 무시)
- **특징**:
  - 단일 스텝 그래디언트 기반 공격
  - 빠른 실행 속도
- **주요 함수**:
  - `fgsm_attack_full()`: FGSM 공격
  - `run_fgsm_full_attack()`: 배치 실행

### 3. FGSM ROI (`attack_fgsm_roi.py`)
- **설명**: ROI 영역 내에서만 섭동 적용
- **특징**:
  - 픽셀 공간 기준 공격
  - ROI 마스크 활용
- **주요 함수**:
  - `fgsm_attack_roi_pixelspace()`: FGSM ROI 공격
  - `run_fgsm_roi_attack()`: 배치 실행

### 4. JSMA (`attack_jsma.py`)
- **설명**: Jacobian 기반 Saliency Map 공격
- **특징**:
  - Margin 기반 공격
  - Adam 스타일 모멘텀
  - 동적 k 조절
  - ROI 마스크 지원
- **주요 함수**:
  - `jsma_attack_margin_mom()`: 핵심 JSMA 알고리즘
  - `jsma_attack()`: Flask 애플리케이션용 인터페이스
  - `run_jsma_full_attack()`: 배치 실행

### 5. JSMA + JPEG Defense (`attack_jsma_jpeg_defense.py`)
- **설명**: JSMA 공격 후 JPEG 압축 방어의 효과 측정
- **특징**:
  - 방어 전후 ASR 비교
  - 마진/신뢰도 변화 분석
- **주요 함수**:
  - `jpeg_compress_tensor_pil()`: JPEG 압축
  - `eval_example_with_jpeg()`: 샘플별 평가
  - `run_jsma_and_eval_jpeg_defense()`: 배치 실험

## 사용 예제

### Square Attack L2

```python
from attacks.attack_square_l2 import run_square_attack_batch, BrainTumorDatasetWithROI
from torchvision import transforms

# 데이터셋 준비
test_set = BrainTumorDatasetWithROI(
    "path/to/Testing",
    transform=transforms.ToTensor()
)

# 공격 실행
results = run_square_attack_batch(
    model=model,
    test_set=test_set,
    device='cuda',
    eps=2.0,
    n_iters=10000,
    p_init=0.1,
    use_roi=False,
    n_samples=100
)
```

### FGSM Full Image

```python
from attacks.attack_fgsm_full import run_fgsm_full_attack, BrainTumorDatasetWithROI
from torchvision import transforms

test_set = BrainTumorDatasetWithROI(
    "path/to/Testing",
    transform=transforms.ToTensor()
)

ASR, mean_L2, mean_changed_pct, mean_time, success_list = run_fgsm_full_attack(
    model, test_set, class_names, eps=8/255.0, n_samples=100
)
```

### FGSM ROI

```python
from attacks.attack_fgsm_roi import run_fgsm_roi_attack, BrainTumorDatasetWithROI
from torchvision import transforms

test_set = BrainTumorDatasetWithROI(
    "path/to/Testing",
    transform=transforms.ToTensor()
)

ASR, mean_L2, mean_changed_pct, mean_time, success_list = run_fgsm_roi_attack(
    model, test_set, class_names, eps=8/255.0, n_samples=100
)
```

### JSMA

```python
from attacks.attack_jsma import run_jsma_full_attack, BrainTumorDatasetWithROI
from torchvision import transforms

test_set = BrainTumorDatasetWithROI(
    "path/to/Testing",
    transform=transforms.ToTensor()
)

# 정규화 파라미터 (모델이 내부에서 정규화하는 경우)
mean = [0.0, 0.0, 0.0]
std  = [1.0, 1.0, 1.0]

ASR, mean_L1, mean_changed_count, mean_time, success_list = run_jsma_full_attack(
    model=model,
    test_set=test_set,
    class_names=class_names,
    mean=mean,
    std=std,
    theta=0.08,
    max_pixels_pct=0.05,
    k_small=2,
    restarts=4,
    topk_pool=5000,
    n_samples=100
)
```

### JSMA + JPEG Defense

```python
from attacks.attack_jsma_jpeg_defense import run_jsma_and_eval_jpeg_defense
from attacks.attack_jsma import BrainTumorDatasetWithROI
from torchvision import transforms

test_set = BrainTumorDatasetWithROI(
    "path/to/Testing",
    transform=transforms.ToTensor()
)

mean = [0.0, 0.0, 0.0]
std  = [1.0, 1.0, 1.0]

jsma_params = {
    'theta': 0.12,
    'max_pixels_pct': 0.10,
    'k_small': 4,
    'restarts': 4,
    'topk_pool': 5000
}

results = run_jsma_and_eval_jpeg_defense(
    model=model,
    test_set=test_set,
    class_names=class_names,
    mean=mean,
    std=std,
    n_samples=50,
    jsma_params=jsma_params,
    jpeg_quality=60,
    visualize_first=True
)
```

## 공통 데이터셋 클래스

모든 모듈에서 사용하는 `BrainTumorDatasetWithROI` 클래스:

```python
class BrainTumorDatasetWithROI(Dataset):
    """
    뇌종양 이미지 데이터셋 (+ ROI mask 포함)
    
    필요한 파일:
    - images.npy : (N,224,224,1) [0,1]
    - labels.npy : (N,) int64
    - masks.npy  : (N,224,224) bool or {0,1}
    - class_names.txt : 클래스명
    - meta.json  : (옵션) 메타데이터
    """
    def __init__(self, data_dir, transform=None):
        ...
    
    def __getitem__(self, idx):
        # Returns: (img, label, mask)
        # img: (3,224,224) tensor
        # label: int
        # mask: (224,224) tensor
        ...
```

## 메트릭

모든 공격 모듈은 다음 메트릭을 반환합니다:

- **ASR (Attack Success Rate)**: 공격 성공률 (%)
- **L2 Norm**: 섭동 L2 노름 (0-255 스케일)
- **Changed Pixels**: 변경된 픽셀 비율/개수
- **Generation Time**: 공격 생성 시간 (초)
- **ΔMargin**: 마진 변화량 (confidence gap drop)
- **ΔConfidence**: 신뢰도 변화량 (softmax)

## 주의사항

1. **경로 설정**: 
   - Colab 환경의 경로(`/content/drive/...`)는 로컬 환경에 맞게 수정 필요
   
2. **모델 정규화**:
   - 모델이 내부에서 정규화하는 경우: `mean=[0,0,0]`, `std=[1,1,1]`
   - 별도 정규화가 필요한 경우: ImageNet 값 사용 `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

3. **ROI 사용**:
   - Square Attack, JSMA: `use_roi` 또는 `roi_mask` 파라미터로 제어
   - FGSM: Full/ROI 별도 모듈

4. **GPU 메모리**:
   - 배치 크기가 큰 경우 GPU 메모리 부족 가능
   - `n_samples` 파라미터로 조절

## 라이선스

이 코드는 연구 목적으로만 사용되어야 합니다.

