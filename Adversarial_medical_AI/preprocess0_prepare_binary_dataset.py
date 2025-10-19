'''
preprocess0_prepare_binary_dataset.py
--------------------------------------
- 최초 1회만 실행
- (선택 1, 2) 중 하나를 선택하여 실행 (나머지는 주석처리)
    (1) Colab 환경 → Google Drive에 저장
    (2) 로컬 환경 → 현재 디렉토리 내에 processed_data 생성

** 기능 요약 **
- KaggleHub로 원본 데이터셋(Brain Tumor MRI Dataset) 자동 다운로드
- 이진 분류용 (tumor / notumor) 폴더 구조로 재분할: Notumor(notumor) vs Tumor(glioma, meningioma, pituitary)
- Training과 Testing 데이터 모두 처리 (최초 1회만 실행)

** 생성되는 폴더 구조 **
processed_data/
├─ Training/
│   ├─ notumor/    # 정상 뇌 MRI 이미지
│   └─ tumor/      # 종양 뇌 MRI 이미지 (glioma, meningioma, pituitary 통합)
└─ Testing/
    ├─ notumor/
    └─ tumor/

** 이미지 개수 정보 (Brain Tumor MRI Dataset) **
- Training: notumor 1595개, tumor 4117개
- Testing: notumor 105개, tumor 906개
- 총 ~4,270개 이미지
'''
import os
import shutil
from pathlib import Path
import sys

def download_kaggle_dataset():
    '''Kaggle에서 뇌종양 MRI 데이터셋을 자동으로 다운로드'''
    try:
        import kagglehub
        
        print("[INFO] Kaggle 데이터셋 다운로드 시작...")
        print("[INFO] 데이터셋: masoudnickparvar/brain-tumor-mri-dataset")
        
        # 최신 버전 다운로드
        path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
        
        print(f"[SUCCESS] 데이터셋 다운로드 완료!")
        print(f"[INFO] 다운로드 경로: {path}")
        
        # 다운로드된 경로의 실제 구조 확인
        path_obj = Path(path)
        print(f"[INFO] 다운로드된 폴더 내용:")
        for item in path_obj.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}/")
                # 하위 폴더도 확인
                try:
                    for subitem in item.iterdir():
                        if subitem.is_dir():
                            print(f"    📁 {subitem.name}/")
                        else:
                            print(f"    📄 {subitem.name}")
                except PermissionError:
                    print(f"    [권한 오류로 하위 폴더 접근 불가]")
            else:
                print(f"  📄 {item.name}")
        
        return path
        
    except ImportError:
        print("[ERROR] kagglehub가 설치되지 않았습니다.")
        print("[INFO] 설치 방법: pip install kagglehub")
        return None
    except Exception as e:
        print(f"[ERROR] 데이터셋 다운로드 실패: {e}")
        return None

# 이진분류 데이터셋 생성 (notumor vs tumor)
def create_binary_classification_dataset(raw_data_root="raw_data", processed_data_root="processed_data", 
                                       download_from_kaggle=False):
    ''' [생성되는 폴더 구조]
    processed_data/  # 이진분류 데이터 경로
    ├─ Training/
    │   ├─ notumor/
    │   └─ tumor/
    └─ Testing/
        ├─ notumor/
        └─ tumor/
    '''
    
    # Kaggle에서 데이터셋 다운로드
    if download_from_kaggle:
        kaggle_path = download_kaggle_dataset()
        if kaggle_path:
            # Kaggle에서 다운로드된 경로를 raw_data_root로 사용
            raw_data_root = kaggle_path
            print(f"[INFO] Kaggle 데이터셋 경로를 사용합니다: {raw_data_root}")
            
            # 실제 폴더 구조 확인 및 경로 조정
            path_obj = Path(raw_data_root)
            
            # 만약 바로 Training/Testing 폴더가 있다면 그대로 사용
            if (path_obj / "Training").exists() and (path_obj / "Testing").exists():
                print("[INFO] Training/Testing 폴더를 직접 찾았습니다.")
            # 만약 하위 폴더에 Training/Testing이 있다면 경로 조정
            elif any((path_obj / subdir / "Training").exists() and (path_obj / subdir / "Testing").exists() 
                     for subdir in path_obj.iterdir() if subdir.is_dir()):
                for subdir in path_obj.iterdir():
                    if subdir.is_dir() and (subdir / "Training").exists() and (subdir / "Testing").exists():
                        raw_data_root = subdir
                        print(f"[INFO] 하위 폴더에서 Training/Testing을 찾았습니다: {raw_data_root}")
                        break
            else:
                print("[WARNING] Training/Testing 폴더를 찾을 수 없습니다. 수동으로 경로를 확인해주세요.")
                print(f"[INFO] 다운로드된 폴더 내용을 확인했습니다: {path_obj}")
        else:
            print("[WARNING] Kaggle 다운로드 실패, 기본 경로를 사용합니다.")
    
    # 경로를 Path 객체로 변환
    raw_data_root = Path(raw_data_root)
    processed_data_root = Path(processed_data_root)
    
    print(f"[INFO] 원본 데이터 루트 경로: {raw_data_root.absolute()}")
    print(f"[INFO] 처리된 데이터 루트 경로: {processed_data_root.absolute()}")
    
    # Training과 Testing 데이터 모두 처리
    datasets = ["Training", "Testing"]
    
    for dataset in datasets:
        print(f"\n[INFO] {dataset} 데이터셋 처리 시작...")
        
        # 경로 설정
        raw_data_path = raw_data_root / dataset
        processed_data_path = processed_data_root / dataset
        
        notumor_path = processed_data_path / "notumor"
        tumor_path = processed_data_path / "tumor"
        
        # 원본 폴더 확인
        if not raw_data_path.exists():
            print(f"[WARNING] {dataset} 원본 데이터 폴더가 존재하지 않습니다: {raw_data_path}")
            continue
        
        # processed_data 폴더 생성
        processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # 기존 폴더가 있다면 삭제 후 재생성
        if notumor_path.exists():
            shutil.rmtree(notumor_path)
        if tumor_path.exists():
            shutil.rmtree(tumor_path)
        
        notumor_path.mkdir()
        tumor_path.mkdir()
        
        print(f"[INFO] {dataset} 이진분류 데이터셋 생성 시작...")
        print(f"[INFO] 원본 경로: {raw_data_path}")
        print(f"[INFO] 처리된 데이터 경로: {processed_data_path}")
        
        # 파일 카운터
        notumor_count = 0
        tumor_count = 0
        
        # notumor -> notumor 폴더로 복사
        source_notumor_path = raw_data_path / "notumor"
        if source_notumor_path.exists():
            for img_file in source_notumor_path.glob("*"):
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    dest_file = notumor_path / img_file.name
                    shutil.copy2(img_file, dest_file)
                    notumor_count += 1
            print(f"[INFO] {dataset} Notumor 이미지 {notumor_count}개를 복사했습니다.")
        else:
            print(f"[WARNING] {dataset} Notumor 폴더가 존재하지 않습니다: {source_notumor_path}")
        
        # glioma, meningioma, pituitary -> tumor 폴더로 복사
        tumor_types = ["glioma", "meningioma", "pituitary"]
        
        for tumor_type in tumor_types:
            tumor_type_path = raw_data_path / tumor_type
            if tumor_type_path.exists():
                type_count = 0
                for img_file in tumor_type_path.glob("*"):
                    if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        # 파일명에 종양 타입을 prefix로 추가하여 구분
                        dest_file = tumor_path / f"{tumor_type}_{img_file.name}"
                        shutil.copy2(img_file, dest_file)
                        type_count += 1
                        tumor_count += 1
                print(f"[INFO] {dataset} {tumor_type} 이미지 {type_count}개를 복사했습니다.")
            else:
                print(f"[WARNING] {dataset} {tumor_type} 폴더가 존재하지 않습니다: {tumor_type_path}")
        
        print(f"\n[SUCCESS] {dataset} 이진분류 데이터셋 생성 완료!")
        print(f"[INFO] {dataset} Notumor: {notumor_count}개")
        print(f"[INFO] {dataset} Tumor: {tumor_count}개")
        print(f"[INFO] {dataset} 총 {notumor_count + tumor_count}개 이미지가 처리되었습니다.")
    
    print(f"\n[SUCCESS] 모든 데이터셋 처리 완료!")
    return True

if __name__ == "__main__":
    ## 선택1, 2 중 하나만 선택 후 나머지는 주석 처리 후 실행

    #(선택1): Colab 실행 시 자동 다운로드 지원
    # processed_root = Path("/content/drive/MyDrive/Adversarial_AI/processed_data")   ## 경로 수정 가능
    # processed_root.parent.mkdir(parents=True, exist_ok=True)
    # create_binary_classification_dataset(
    #     processed_data_root=processed_root,
    #     download_from_kaggle=True
    # )

    # (선택2): 로컬 실행 시 기본값 사용 및 kaggle 자동 다운로드 지원
    create_binary_classification_dataset(download_from_kaggle=True)
    
# (선택) Colab 실행 후 아래 명령어로 정상 저장 여부 확인 가능
# !ls -la "/content/drive/MyDrive/Adversarial_AI" || echo "Adversarial_AI 없음"
# !ls -la "/content/drive/MyDrive/Adversarial_AI/processed_data" || echo "processed_data 없음"
# !ls -la "/content/drive/MyDrive/Adversarial_AI/processed_data/Training" || echo "Training 없음"