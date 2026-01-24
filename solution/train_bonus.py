"""Training script for the bonus model."""
import torch
from torch import nn, optim
from bonus_model import BonusModel
from utils import load_dataset, get_nof_params
from trainer import Trainer, LoggingParameters

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

    optimizer_params = optimizer.param_groups[0].copy()
    del optimizer_params['params']

    # Training Logging Parameters
    logging_parameters = LoggingParameters(model_name='BonusModel',
                                           dataset_name='fakes_dataset',
                                           optimizer_name='Adam',
                                           optimizer_params=optimizer_params,)

    # Initialize output data for logging
    output_data = {
        "model": logging_parameters.model_name,
        "dataset": logging_parameters.dataset_name,
        "optimizer": {
            "name": logging_parameters.optimizer_name,
            "params": logging_parameters.optimizer_params,
        },
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

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
        train_loss, train_acc = trainer.train_one_epoch()
        val_loss, val_acc = trainer.validate()
        test_loss, test_acc = trainer.test()

        scheduler.step(val_acc)

        # Collect data for logging
        output_data["train_loss"].append(train_loss)
        output_data["train_acc"].append(train_acc)
        output_data["val_loss"].append(val_loss)
        output_data["val_acc"].append(val_acc)
        output_data["test_loss"].append(test_loss)
        output_data["test_acc"].append(test_acc)

        if val_acc > best_val_acc:
            print(f'*** New best validation accuracy: {val_acc:.2f}% ***')
            state = {
                'model': model.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc,
                'epoch': epoch,
            }
            torch.save(state, 'checkpoints/bonus.pt')
            best_val_acc = val_acc

    print(f"\nTraining completed! Model saved to: checkpoints/bonus.pt")

    Trainer.write_output(logging_parameters, output_data)

if __name__ == '__main__':
    main()