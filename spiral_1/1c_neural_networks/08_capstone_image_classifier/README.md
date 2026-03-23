# 08 -- Capstone: Image Classifier

## Goal

Build, train, and compare two image classifiers on CIFAR-10:

1. **SimpleCNN** -- a CNN trained from scratch (target: >70% accuracy)
2. **TransferModel** -- a fine-tuned pretrained ResNet (target: >85% accuracy)

## What You Will Build

### `model.py`

- `SimpleCNN(nn.Module)`: A 3-block CNN (Conv-ReLU-Pool each block), followed
  by fully connected layers. Input: 3x32x32 CIFAR-10 images.
- `TransferModel(nn.Module)`: A pretrained ResNet-18 with a replaced
  classification head for 10 classes.

### `train.py`

- `load_data(batch_size)`: Download CIFAR-10, apply transforms (resize for
  transfer model, normalize), return train/test DataLoaders.
- `train(model, train_loader, epochs, lr)`: The training loop. Returns a
  history dict with loss and accuracy per epoch.
- `evaluate_and_report(model, test_loader)`: Compute test accuracy, per-class
  accuracy, and a confusion matrix.
- `compare_models(simple_results, transfer_results)`: Print a comparison table.

## Architecture Details

### SimpleCNN

```
Conv2d(3, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)
Conv2d(64, 128, 3, padding=1) -> ReLU -> MaxPool2d(2)
Flatten -> Linear(128*4*4, 256) -> ReLU -> Dropout(0.5) -> Linear(256, 10)
```

### TransferModel

```
ResNet-18 (pretrained, backbone frozen)
Replace fc: Linear(512, 10)
Optionally unfreeze last residual block for fine-tuning
```

## Success Criteria

| Model         | Test Accuracy | Training Time |
|---------------|--------------|---------------|
| SimpleCNN     | > 70%        | ~5 min (GPU)  |
| TransferModel | > 85%        | ~3 min (GPU)  |

## How to Run

```bash
# Train both models and compare
python spiral_1/1c_neural_networks/08_capstone_image_classifier/train.py

# Run tests
python -m pytest spiral_1/1c_neural_networks/08_capstone_image_classifier/tests/ -v
```

## Reflection Questions

1. Why does the transfer model reach higher accuracy with less training?
2. Which classes does SimpleCNN struggle with? Why?
3. What would happen if you unfroze the entire ResNet and trained with a
   high learning rate?
4. How would you improve SimpleCNN further? (Hint: data augmentation, batch
   norm, deeper architecture.)

## Style Notes

- Keep model definitions in `model.py` and training logic in `train.py`.
- Use `argparse` or simple constants at the top of `train.py` for
  hyperparameters.
- Save trained model weights with `torch.save(model.state_dict(), path)`.
