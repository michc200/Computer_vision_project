"""Training script for the bonus model."""
import torch
from torch import nn, optim
from bonus_model import BonusModel
from utils import load_dataset, get_nof_params
from trainer import Trainer

def main():
    """Train the bonus model on the fakes_dataset."""
    print("=" * 60)
    print("Training MobileNetV2-inspired Bonus Model")
    print("=" * 60)

    # Load datasets
    print("\n==> Preparing data: fakes_dataset..")
    train_dataset = load_dataset(dataset_name='fakes_dataset', dataset_part='train')
    val_dataset = load_dataset(dataset_name='fakes_dataset', dataset_part='val')
    test_dataset = load_dataset(dataset_name='fakes_dataset', dataset_part='test')

    # Create model
    model = BonusModel()
    nof_params = get_nof_params(model)
    print(f"Number of parameters: {nof_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=32,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset
    )

    best_val_acc = 0
    epochs = 15
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}/{epochs}')
        trainer.train_one_epoch()
        val_loss, val_acc = trainer.validate()
        test_loss, test_acc = trainer.test()

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            print(f'*** New best validation accuracy: {val_acc:.2f}% ***')
            # Save to the path required by the PDF
            state = {
                'model': model.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc,
                'epoch': epoch,
            }
            torch.save(state, 'checkpoints/bonus.pt')
            best_val_acc = val_acc

    print(f"\nTraining completed! Model saved to: checkpoints/bonus.pt")

if __name__ == '__main__':
    main()