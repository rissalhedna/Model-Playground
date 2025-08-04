import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score, precision_recall_fscore_support, confusion_matrix, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from model import create_lightweight_detector

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TrailerDataModule:
    def __init__(self, annotations_file, image_dir, transform=None, target_transform=None):
        self.annotations = pd.read_parquet(annotations_file)
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.annotations.iloc[idx, 1])
        image = Image.open(img_path).convert("RGB")
        label = np.digitize(
            self.annotations.iloc[idx, 3], bins=[0, 0.1, 0.5]) - 1

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def train_with_scheduler(train_dataloader, test_dataloader, model, loss_fn, optimizer, scheduler, epochs):
    train_losses = []
    test_losses = []
    test_accuracies = []
    test_f1_scores = []
    test_balanced_accuracies = []
    test_kappa_scores = []
    learning_rates = []

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        print(f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.2e}")

        model.train()
        train_loss = 0

        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            if batch % 10 == 0:
                current = (batch + 1) * len(X)
                print(
                    f"  Batch {batch}: loss: {loss.item():.6f} [{current}/{len(train_dataloader.dataset)}]")

        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        test_loss, correct = 0, 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()

                pred_classes = pred.argmax(1)
                correct += (pred_classes == y).type(torch.float).sum().item()

                all_preds.extend(pred_classes.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        avg_test_loss = test_loss / len(test_dataloader)
        accuracy = correct / len(test_dataloader.dataset)

        f1 = f1_score(all_targets, all_preds, average='weighted')
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        kappa = cohen_kappa_score(all_targets, all_preds)

        test_losses.append(avg_test_loss)
        test_accuracies.append(accuracy)
        test_f1_scores.append(f1)
        test_balanced_accuracies.append(balanced_acc)
        test_kappa_scores.append(kappa)

        scheduler.step(avg_test_loss)

        print(
            f"Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        print(
            f"Accuracy: {accuracy*100:.2f}%, F1: {f1:.4f}, Balanced Acc: {balanced_acc:.4f}, Kappa: {kappa:.4f}")

        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f"LR reduced from {current_lr:.2e} to {new_lr:.2e}")

    return train_losses, test_losses, test_accuracies, test_f1_scores, test_balanced_accuracies, test_kappa_scores, learning_rates, all_targets, all_preds


def plot_results(train_losses, test_losses, test_accuracies, test_f1_scores, test_balanced_accuracies, test_kappa_scores, learning_rates, all_targets, all_preds):
    epochs = range(1, len(train_losses) + 1)

    fig = plt.figure(figsize=(20, 12))

    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, [acc * 100 for acc in test_accuracies],
             'g-', label='Accuracy')
    ax2.plot(epochs, [f1 * 100 for f1 in test_f1_scores],
             'orange', label='F1-Score')
    ax2.plot(epochs, [ba * 100 for ba in test_balanced_accuracies],
             'purple', label='Balanced Acc')
    ax2.set_title('Classification Metrics (%)')
    ax2.legend()
    ax2.grid(True)

    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, test_kappa_scores, 'red', marker='o', markersize=3)
    ax3.set_title('Cohen\'s Kappa')
    ax3.grid(True)

    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(epochs, learning_rates, 'purple', marker='o', markersize=3)
    ax4.set_title('Learning Rate')
    ax4.set_yscale('log')
    ax4.grid(True)

    ax5 = plt.subplot(2, 3, 5)
    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5)
    ax5.set_title('Confusion Matrix')
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('Actual')

    ax6 = plt.subplot(2, 3, 6)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None)
    class_names = ['Low', 'Medium', 'High']
    x = np.arange(len(class_names))
    width = 0.25
    ax6.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax6.bar(x, recall, width, label='Recall', alpha=0.8)
    ax6.bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)
    ax6.set_title('Per-Class Metrics')
    ax6.set_xticks(x)
    ax6.set_xticklabels(class_names)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    torch.cuda.empty_cache()
    gc.collect()

    transform = v2.Compose([
        v2.ToTensor(),
        v2.Resize([512, 512]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data = TrailerDataModule(
        annotations_file="trailer-53ft-data-large/data.parquet",
        image_dir="trailer-53ft-data-large",
        transform=transform
    )

    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, shuffle=True)

    batch_size = 4
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True)

    model = create_lightweight_detector().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        verbose=True,
        min_lr=1e-9
    )

    print(f"Starting training with batch size {batch_size}")

    train_losses, test_losses, test_accuracies, test_f1_scores, test_balanced_accuracies, test_kappa_scores, learning_rates, all_targets, all_preds = train_with_scheduler(
        train_dataloader, test_dataloader, model, loss_fn, optimizer, scheduler, epochs=30
    )

    print(f"\nTraining completed!")
    print(f"Final Accuracy: {test_accuracies[-1]*100:.2f}%")
    print(f"Final F1-Score: {test_f1_scores[-1]:.4f}")
    print(f"Final Balanced Accuracy: {test_balanced_accuracies[-1]:.4f}")
    print(f"Final Cohen's Kappa: {test_kappa_scores[-1]:.4f}")

    torch.save(model.state_dict(), 'trained_model.pth')
    print("Model saved as 'trained_model.pth'")

    plot_results(train_losses, test_losses, test_accuracies, test_f1_scores,
                 test_balanced_accuracies, test_kappa_scores, learning_rates, all_targets, all_preds)


if __name__ == "__main__":
    main()
