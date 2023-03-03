import torch
from metrics import compute_metrics
import torch.nn.functional as F

def model_training(model, optimizer, num_epochs, dataloader, device):
    for epoch in range(num_epochs):
        print("Epoch: ", epoch+1)
        model.train()
        for batch_idx, batch_data in enumerate(dataloader):
            text = batch_data.REVIEW_TEXT.to(torch.device(0))
            labels = batch_data.isPos.to(torch.device(0))

            logits = model(text)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if batch_idx % 50 == 0:
                print("Batch: ", batch_idx, " of ", len(dataloader), end=" | ")
                print("Loss: ", loss)
    
        with torch.set_grad_enabled(False):
            f1, recall, precision = compute_metrics(model, dataloader, device)
            print("Epoch ", epoch+1, " Metrics:")
            print("Precision: %f | Recall: %f | F1-Score: %f", precision, recall, f1)
            
