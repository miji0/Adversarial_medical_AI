'''
preprocess1_preprocess_image.py
--------------------------------------
- 최초 1회만 실행
- (선택 1, 2) 중 하나를 선택하여 실행 (나머지는 주석처리)
    (1) Colab 환경 → Google Drive에 저장
    (2) 로컬 환경 → 현재 디렉토리 내에 processed_data_np224 생성

** 기능 요약 **
- processed_data/ 내의 이미지를 불러와서
  1채널(흑백), 224x224 크기로 리사이즈 및 패딩, [0,1] float32로 정규화하여 npy로 저장
- Otsu 이진화 방법을 이용한 ROI(관심영역) 마스크 생성, npy로 저장
- Training과 Testing 데이터 모두 처리

** 생성되는 폴더 구조 **
Adversarial_mdedical_AI/
├─ processed_data/                 # 입력 (preprocess0_prepare_binary_dataset.py 출력)
├─ processed_data_np224/           # 출력 (1채널, (N,224,224,1), [0,1], ROI 마스크 포함)
│   ├─ Training/
│   │   ├─ images.npy      # (N,224,224,1) float32, [0,1] (흑백)
│   │   ├─ labels.npy      # (N,) int64, 0=notumor, 1=tumor
│   │   ├─ masks.npy       # (N,224,224) bool (ROI)
│   │   ├─ class_names.txt # 클래스명
│   │   └─ meta.json       # 메타데이터
│   └─ Testing/
│       ├─ images.npy
│       ├─ labels.npy
│       ├─ masks.npy
│       ├─ class_names.txt
│       └─ meta.json

** npy 설명 **
- images.npy : 이미지들을 224x224로 리사이즈해서 흑백 1채널로 저장, (N,224,224,1) float32, 값 범위 [0,1]
- labels.npy : notumor=0, tumor=1의 int64 레이블, (N,)
- masks.npy  : Otsu 이진화를 통한 ROI 마스크, (N,224,224) bool
- class_names.txt : 라벨-클래스명 매핑 텍스트 (notumor, tumor)
- meta.json : 이미지 정보, 마스크 포함 여부 등 메타데이터

'''

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

# =======================
# Colab 환경 자동 감지 및 설정
# =======================
try:
    import google.colab
    IN_COLAB = True
    print("[INFO] Google Colab 환경 감지")
    
    # Colab에서만 Drive 마운트
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Colab에서만 경로 이동
    os.chdir('/content/drive/MyDrive/Adversarial_medical_AI/Adversarial_medical_AI')
    print(f"[INFO] 작업 디렉토리 변경: {os.getcwd()}")
    
except ImportError:
    IN_COLAB = False
    print("[INFO] 로컬 환경에서 실행")

# ---------------------------
# 1. 224x224 리사이즈 및 패딩 (흑백 1채널)
# ---------------------------
def _resize_with_pad(img, size=(224,224), pad=0):
    w, h = img.size
    tw, th = size
    s = min(tw/w, th/h)
    nw, nh = int(w*s), int(h*s)
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("L", size, pad)
    canvas.paste(img, ((tw-nw)//2, (th-nh)//2))
    return canvas

# ---------------------------
# 2. Otsu 이진화로 ROI 마스크 생성
# ---------------------------
def _make_roi_mask(gray01):
    g = (np.clip(gray01,0,1)*255).astype(np.uint8)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    p = hist/hist.sum() if hist.sum() > 0 else np.zeros_like(hist)
    omega = p.cumsum()
    mu = (p*np.arange(256)).cumsum()
    mt = mu[-1]
    denom = omega*(1-omega)
    denom[denom==0] = np.nan
    t = np.nanargmax((mt*omega-mu)**2/denom)    # 최적 Otsu 임계값
    thr = float((t+0.5)/255.0)
    thr = float(np.clip(thr,0.02,0.30))
    return (gray01 > thr)

# ---------------------------
# 3. 폴더 내 모든 이미지 로드 및 전처리
# ---------------------------
def split_and_preprocess_images(src_dir, img_size=(224,224)):
    images, labels, masks = [], [], []
    class_names = ["notumor", "tumor"]
    c2i = {c:i for i,c in enumerate(class_names)}

    for cls in sorted(d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir,d))):
        norm = cls.strip().lower().replace("_","").replace("-","")
        target = "notumor" if "notumor" in norm else "tumor"
        dpath = os.path.join(src_dir, cls)
        files = [f for f in os.listdir(dpath) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff"))]
        for f in tqdm(files, desc=f"{cls}→{target}"):
            p = os.path.join(dpath, f)
            try:
                img = Image.open(p).convert("L")
                img = _resize_with_pad(img, img_size)
                arr = np.asarray(img, dtype=np.float32)/255.0          # (H,W)
                msk = _make_roi_mask(arr)                              # (H,W) bool
                images.append(arr[...,None].astype(np.float32))        # (H,W,1)
                labels.append(c2i[target])
                masks.append(msk)
            except Exception as e:
                print("skip:", p, "|", e)

    X = np.stack(images) if images else np.empty((0,img_size[0],img_size[1],1),np.float32)
    y = np.array(labels, dtype=np.int64)
    M = np.stack(masks) if masks else np.empty((0,img_size[0],img_size[1]), np.bool_)
    return X, y, M, class_names     # (N,H,W,1) in [0,1], (N,) int64, (N,H,W) bool, [class_names]

# ---------------------------
# 4. 전처리 결과 저장 (npy 및 class/meta)
# ---------------------------
def save_split(src_dir, dst_dir, img_size=(224,224)):
    os.makedirs(dst_dir, exist_ok=True)
    X, y, M, names = split_and_preprocess_images(src_dir, img_size=img_size)

    np.save(os.path.join(dst_dir, "images.npy"), X)   # (N,H,W,1) in [0,1]
    np.save(os.path.join(dst_dir, "labels.npy"), y)   # (N,)
    np.save(os.path.join(dst_dir, "masks.npy"), M)    # (N,H,W) bool

    with open(os.path.join(dst_dir, "class_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names))

    meta = {"img_size": list(img_size), "channels": 1, "value_range": "[0,1]", "roi_mask": True}
    with open(os.path.join(dst_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"저장 완료 → {dst_dir}")
    print(f"  images: {X.shape}, labels: {y.shape}, masks: {M.shape}, classes: {names}")
    return X, y, M, names

# ---------------------------
# 메인 실행: Colab/로컬 중 택1
# ---------------------------
if __name__ == "__main__":
    ## 선택1, 2 중 하나만 선택 후 나머지는 주석 처리 후 실행

    # (선택1): Colab 실행 시 경로 지정 예시
    # processed_root = "/content/drive/MyDrive/Adversarial_medical_AI/processed_data"   ## 경로 수정 가능
    # out_root = "/content/drive/MyDrive/Adversarial_medical_AI/processed_data_np224"  ## 경로 수정 가능
    # SRC_TRAIN = os.path.join(processed_root, "Training")
    # SRC_TEST  = os.path.join(processed_root, "Testing")
    # TRAIN_DATA = os.path.join(out_root, "Training")
    # TEST_DATA  = os.path.join(out_root, "Testing")
    # os.makedirs(TRAIN_DATA, exist_ok=True)
    # os.makedirs(TEST_DATA, exist_ok=True)
    # print("="*50)
    # print("1채널(흑백) 224x224 리사이즈 및 npy 저장을 시작합니다.")
    # print("="*50)
    # Xtr, ytr, Mtr, names = save_split(SRC_TRAIN, TRAIN_DATA, img_size=(224,224))
    # Xte, yte, Mte, _     = save_split(SRC_TEST,  TEST_DATA,  img_size=(224,224))

    # (선택2): 로컬 실행 시 기본 경로 사용
    processed_root = "processed_data"
    out_root = "processed_data_np224"
    SRC_TRAIN = os.path.join(processed_root, "Training")
    SRC_TEST  = os.path.join(processed_root, "Testing")
    TRAIN_DATA = os.path.join(out_root, "Training")
    TEST_DATA  = os.path.join(out_root, "Testing")
    os.makedirs(TRAIN_DATA, exist_ok=True)
    os.makedirs(TEST_DATA, exist_ok=True)
    print("="*50)
    print("1채널(흑백) 224x224 리사이즈 및 npy 저장을 시작합니다.")
    print("="*50)
    Xtr, ytr, Mtr, names = save_split(SRC_TRAIN, TRAIN_DATA, img_size=(224,224))
    Xte, yte, Mte, _     = save_split(SRC_TEST,  TEST_DATA,  img_size=(224,224))
    
    
# (선택) Colab 실행 후 아래 명령어로 정상 저장 여부 확인 가능
# !ls -la "/content/drive/MyDrive/Adversarial_medical_AI/processed_data_np224" || echo "processed_data_np224 없음"
# !ls -la "/content/drive/MyDrive/Adversarial_medical_AI/processed_data_np224/Training" || echo "Training 없음"
