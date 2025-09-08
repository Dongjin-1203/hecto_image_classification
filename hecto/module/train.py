import matplotlib.pyplot as plt

from module.utils import evaluate, plot_training, EarlyStopping


def train_model(model, cce, optimizer, train_loader, val_loader, device, epoch, scheduler, save_path="best_model.pt"):
    early_stopping = EarlyStopping(patience=5, save_path=save_path)
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epoch):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = cce(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_loss, val_acc = evaluate(model, val_loader, device, cce)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)  # 매 epoch 끝에 호출
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("⛔ Early stopping triggered.")
            # 학습 그래프 저장
            plot_training(train_losses, val_losses, val_accuracies, save_path="img/training_plot.png")
            break

    # 정상 종료 시도 그래프 저장
    if not early_stopping.early_stop:
        plot_training(train_losses, val_losses, val_accuracies, save_path="img/training_plot.png")
