import matplotlib.pyplot as plt

def plot_all_folds(epochs, history_list):
    print("Generating Loss Graph...")

    plt.figure(figsize=(8,5))
    plot_epoch = list(range(1, epochs + 1))

    for i, fold in enumerate(history_list):
        train_loss = flatten_loss(fold["train_loss"])
        plt.plot(plot_epoch, train_loss, label=f"Fold {i+1}")

    plt.title("Training Loss Across All Folds")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(plot_epoch)
    plt.legend()
    plt.grid(True)
    plt.savefig("validation_all_folds.png")
    print("Graph Generation Completed")

def flatten_loss(loss_list):
    # If elements are lists, flatten them
    if isinstance(loss_list[0], list) or isinstance(loss_list[0], tuple):
        return [x[0] for x in loss_list]
    return loss_list
