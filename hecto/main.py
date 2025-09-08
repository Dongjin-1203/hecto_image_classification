import os
import ssl
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

ssl._create_default_https_context = ssl._create_unverified_context

# 모듈 불러오기
from module.data_set import data_split, ImageFolderWithPaths
from module.train import train_model
from module.utils import (
    evaluate,
    save_softmax_submission,
    #plot_confusion_matrix,
    analyze_noisy_samples_with_filenames,
    TestImageDataset
)
from module.resNet_18 import get_resnet18_model

def load_noise_list(noise_txt='noise_list.txt'):
    if not os.path.exists(noise_txt):
        print("노이즈 리스트 파일이 없습니다. 전체 데이터 사용.")
        return set()
    with open(noise_txt, 'r') as f:
        noise_set = set(line.strip() for line in f)
    return noise_set

def filter_noise_from_imagefolder(dataset, noise_set):
    original_len = len(dataset.samples)
    dataset.samples = [s for s in dataset.samples if s[0] not in noise_set]
    print(f"노이즈 {original_len - len(dataset.samples)}장 제외, 최종 학습용 {len(dataset.samples)}장")
    return dataset

if __name__ == "__main__":
    print("모듈을 성공적으로 불러왔습니다.")

    # 1. Transform 정의
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # 2. 노이즈 리스트 불러오기
    noise_set = load_noise_list('noise_list.txt')

    # 3. 학습 데이터 ImageFolder + 노이즈 제외
    train_path = 'data/train'
    dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    dataset = filter_noise_from_imagefolder(dataset, noise_set)
    class_names = dataset.classes

    # 4. train/val split
    train_loader, val_loader = data_split(dataset, class_names, visualize=False)
    val_loader.dataset.dataset.transform = val_transform

    # 5. 테스트셋 경로
    test_path = 'data/test'

    # 6. 모델 리스트
    model_list = {
        "ResNet18": lambda: get_resnet18_model(num_classes=len(class_names), pretrained=True)
    }

    for model_name, model_fn in model_list.items():
        print(f"\n📌 Training: {model_name}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_fn().to(device)

        cce = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        save_path = f"{model_name}_best_model.pt"
        epoch = 10

        # 7. 학습
        train_model(model, cce, optimizer, train_loader, val_loader, device, epoch, scheduler, save_path=save_path)

        # 8. 모델 로드 및 평가
        model.load_state_dict(torch.load(save_path))
        val_loss, val_acc = evaluate(model, val_loader, device, cce)
        print(f"✅ {model_name} → Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")

        # 9. 테스트셋 예측
        output_csv = f"{model_name}_submission.csv"
        save_softmax_submission(model, test_path, val_transform, device, class_names, output_csv=output_csv)

        # 10. (선택) 노이즈 샘플 분석 - 파일명 포함
        # 분석 시에는 파일명 정보를 반환하는 ImageFolderWithPaths와 DataLoader 사용!
        analysis_dataset = ImageFolderWithPaths(root=train_path, transform=val_transform)
        analysis_dataset = filter_noise_from_imagefolder(analysis_dataset, noise_set)
        analysis_loader = DataLoader(analysis_dataset, batch_size=32, shuffle=False)

        noisy_df = analyze_noisy_samples_with_filenames(
            model,
            analysis_loader,
            device,
            class_names,
            out_dir=f"{model_name}_noisy_samples",
            top_k=50
        )

        # 혼동행렬 등 다른 분석은 필요시 주석 해제!
        # plot_confusion_matrix(model, val_loader, device, class_names, save_path=f"{model_name}_confusion_matrix.png")
