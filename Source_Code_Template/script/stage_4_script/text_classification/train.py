import torch
from metrics import compute_metrics
import torch.nn.functional as F

def model_training(model, optimizer, num_epochs, train_dataloader, test_dataloader, device):
    losses = []
    for epoch in range(num_epochs):
        print("Epoch: ", epoch+1)
        model.train()
        for batch_idx, batch_data in enumerate(train_dataloader):
            text = batch_data.REVIEW_TEXT.to(device)
            labels = batch_data.isPos.to(device)

            logits = model(text)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if batch_idx % 50 == 0:
                print("Batch: ", batch_idx, " of ", len(train_dataloader), end=" | ")
                #print(type(loss))
                print(f"Loss: ", loss.item())
            
            if batch_idx == 199:
                losses.append(loss.item())
                print(losses)
    
        with torch.set_grad_enabled(False):
            f1, recall, precision = compute_metrics(model, train_dataloader, device)
            print("\n")
            print("Epoch ", epoch+1, " Metrics:")
            print("Precision:", precision, "| Recall: ", recall, "| F1-Score:", f1)
            print("\n")
    
    print("Training complete, now testing...")
    f1, recall, precision = compute_metrics(model, test_dataloader, device)
    print("Test Model Results:")
    print("Precision:", precision, "| Recall: ", recall, "| F1-Score:", f1)