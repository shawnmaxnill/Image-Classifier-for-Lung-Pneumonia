def print_metrics(history):
    for fold, history in enumerate(history):
        print(f"\n--- Fold {fold+1} Summary ---")
        print(f"Best Val Accuracy: {max(history['val_accuracy']):.2f}%")
        print(f"Final Train Loss: {history['train_loss'][-1]}")
        print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")