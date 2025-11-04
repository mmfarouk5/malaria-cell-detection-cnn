import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from baseline_model import BaselineCNN
from data import test_loader


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = BaselineCNN().to(device)

model_path = "../outputs/models/baseline_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Trained model not found at: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

criterion = nn.BCEWithLogitsLoss()

plot_dir = "../outputs/figures"
os.makedirs(plot_dir, exist_ok=True)


y_true, y_pred, y_scores = [], [], []
total_loss = 0.0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating on test set"):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_scores.extend(probs.cpu().numpy())

avg_loss = total_loss / len(test_loader)


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n✅ Test Results:")
print(f"Average Loss   : {avg_loss:.4f}")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Uninfected", "Parasitized"],
            yticklabels=["Uninfected", "Parasitized"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

cm_path = os.path.join(plot_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"📊 Confusion matrix saved at: {cm_path}")


fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()

roc_path = os.path.join(plot_dir, "roc_curve.png")
plt.savefig(roc_path)
plt.close()
print(f"📈 ROC curve saved at: {roc_path}")

print("\n✅ Evaluation complete! All plots saved in '../outputs/figures/'")
