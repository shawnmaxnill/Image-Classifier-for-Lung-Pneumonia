def train(dataloader, model, loss_fn, optimizer, device="cuda"):

    train_loss_list = []
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0

    for batch, (X, y) in enumerate(dataloader, 0):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # Computing prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        
    avg_train_loss = total_loss / len(dataloader)
    train_loss_list.append(avg_train_loss)
    return train_loss_list