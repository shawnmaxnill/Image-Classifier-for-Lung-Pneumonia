import torch

def test(dataloader, model, loss_fn, device = "cuda"):

    validation_loss = []
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Perform prediction
            pred = model(X)
            loss = loss_fn(pred, y)
            _, predicted = torch.max(pred, 1)

            # Computing statistics
            total += y.size(0)
            correct += (predicted == y).sum().item()
            total_loss += loss.item()

    # Computing statistics
    avg_loss = total_loss/len(dataloader)
    validation_loss.append(avg_loss)
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")

    return accuracy, avg_loss