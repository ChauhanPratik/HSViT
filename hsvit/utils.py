import matplotlib.pyplot as plt
import torch
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

def visualize_batch(batch, title="Batch Samples"):
    images, labels = batch
    images = images[:8]  # show up to 8
    labels = labels[:8]

    plt.figure(figsize=(16, 4))
    for i in range(len(images)):
        plt.subplot(1, 8, i + 1)
        img = images[i].squeeze().cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def compute_classification_metrics(y_true, y_pred, average='macro'):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average=average),
        "Precision": precision_score(y_true, y_pred, average=average),
        "Recall": recall_score(y_true, y_pred, average=average),
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }

def compute_iou(box_pred, box_true):
    """
    box: [B, 4] in format (x, y, w, h)
    """
    eps = 1e-6
    x1 = torch.max(box_pred[:, 0], box_true[:, 0])
    y1 = torch.max(box_pred[:, 1], box_true[:, 1])
    x2 = torch.min(box_pred[:, 0] + box_pred[:, 2], box_true[:, 0] + box_true[:, 2])
    y2 = torch.min(box_pred[:, 1] + box_pred[:, 3], box_true[:, 1] + box_true[:, 3])

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area_pred = box_pred[:, 2] * box_pred[:, 3]
    area_true = box_true[:, 2] * box_true[:, 3]
    union = area_pred + area_true - inter + eps
    return (inter / union).cpu().numpy()
