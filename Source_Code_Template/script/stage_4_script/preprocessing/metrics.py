import torch
from sklearn.metrics import f1_score, precision_score, recall_score

def compute_metrics(model, data_loader, device):
    with torch.no_grad():
        num_correct = 0
        num_instances = 0

        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_instances += targets.size(0)
            num_correct += (predicted_labels == targets).sum()

    
    f1 = f1_score(targets.cpu(), predicted_labels.cpu())
    recall = recall_score(targets.cpu(), predicted_labels.cpu())
    precision = precision_score(targets.cpu(), predicted_labels.cpu())
    
    return f1, recall, precision