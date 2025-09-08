from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder



def evaluate(model, data_loader, device, cce):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = cce(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    avg_loss = running_loss / len(data_loader)
    return avg_loss, acc

def test(model, test_path, transform, device):
    test_data = datasets.ImageFolder(root=test_path, transform=transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model.eval()
    predictions = []     # 분류 결과

    with torch.no_grad():
        for inputs, _ in test_loader:  # 테스트셋 라벨은 무시
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    return test_data, predictions

def save_softmax_submission(model, test_path, transform, device, class_names, output_csv="submission.csv"):
    # test 데이터셋 (클래스 폴더 없이)
    test_dataset = TestImageDataset(test_path, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()
    results = []
    ids = []
    filenames = []

    all_probs = []

    with torch.no_grad():
        for imgs, fnames in tqdm(test_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)  # 확률값
            all_probs.append(probs.cpu().numpy())
            filenames.extend(fnames)

    all_probs = np.concatenate(all_probs, axis=0)
    
    # 파일명을 ID로 바꾸는 부분 (예: "TEST_00000" 형태)
    ids = [os.path.splitext(f)[0] for f in filenames]  # 파일명에서 .jpg 등 제거
    # 만약 ID가 특정 형식이면 여기서 조정

    df = pd.DataFrame(all_probs, columns=class_names)
    df.insert(0, "ID", ids)

    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"저장 완료: {output_csv}")

def plot_training(train_losses, val_losses, val_accuracies, save_path="img/"):
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12, 5))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📈 학습 그래프 저장 완료: {save_path}")
    plt.show()

# 조기 종료
class EarlyStopping:
    def __init__(self, patience=5, verbose=True, save_path="best_model.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f"💤 EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            print(f"✅ Best model saved to {self.save_path}")

# 혼동 행렬
def plot_confusion_matrix(model, data_loader, device, class_names, normalize='true', title="Confusion Matrix", save_path="img/confusion_matrix.png"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format=".2f")
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Confusion Matrix 저장 완료: {save_path}")
    plt.show()

class TestImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(self.paths[idx])



def analyze_noisy_samples_with_filenames(model, data_loader, device, class_names, out_dir="noisy_samples", top_k=50):
    model.eval()
    results = []
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(data_loader):
            # (이미지, 라벨, 파일명)
            imgs, labels, filenames = batch
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            losses = F.cross_entropy(outputs, labels, reduction='none')
            probs = F.softmax(outputs, dim=1)

            for i in range(len(labels)):
                results.append({
                    "img_tensor": imgs[i].cpu(),
                    "filename": filenames[i],
                    "true_label": class_names[labels[i].item()],
                    "true_label_idx": labels[i].item(),
                    "pred_label": class_names[probs[i].argmax().item()],
                    "pred_label_idx": probs[i].argmax().item(),
                    "loss": losses[i].item(),
                    "probs": ";".join(map(str, probs[i].cpu().numpy()))
                })

    # loss 기준 내림차순
    results = sorted(results, key=lambda x: x["loss"], reverse=True)[:top_k]

    # 이미지 및 CSV 저장 (원본 파일명 사용)
    csv_rows = []
    for sample in results:
        save_path = os.path.join(image_dir, sample["filename"])
        save_image(sample["img_tensor"], save_path)
        row = {
            "img_path": save_path,
            "filename": sample["filename"],
            "true_label": sample["true_label"],
            "true_label_idx": sample["true_label_idx"],
            "pred_label": sample["pred_label"],
            "pred_label_idx": sample["pred_label_idx"],
            "loss": sample["loss"],
            "probs": sample["probs"]
        }
        csv_rows.append(row)

    # CSV 저장
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(out_dir, f"noisy_samples_top{top_k}.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV 및 이미지 저장 완료: {csv_path}")

    return df

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)  # (image, label)
        path = self.samples[index][0]
        filename = os.path.basename(path)
        return original_tuple + (filename,)