# 07 -- Transfer Learning

## Learning Objectives

- Load a pretrained model (ResNet) from torchvision
- Freeze the backbone and replace the classification head
- Count trainable vs frozen parameters
- Selectively unfreeze layers for fine-tuning
- Train a fine-tuned model on a new dataset

## The Big Idea

Training a CNN from scratch requires millions of images. **Transfer learning**
lets you reuse a model trained on ImageNet (1.2 million images, 1000 classes)
and adapt it to your task with far less data. The early layers learn generic
features (edges, textures) that transfer across domains; only the later layers
need task-specific training.

### Go Parallel

Think of a pretrained model as a well-tested library. You would not rewrite
`net/http` for every project -- you import it and customize the handler. Transfer
learning is the same idea: import the feature extractor, customize the
classifier.

## Key Concepts

### 1. Loading a Pretrained Model

```python
model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
```

This gives you a ResNet-18 with weights trained on ImageNet.

### 2. Freezing the Backbone

```python
for param in model.parameters():
    param.requires_grad = False
```

Frozen parameters are not updated during training. This is fast and prevents
catastrophic forgetting.

### 3. Replacing the Head

```python
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

Only the new head has `requires_grad=True`, so only it gets trained.

### 4. Selective Unfreezing

For better accuracy, unfreeze the last N layers and fine-tune with a small
learning rate. The pretrained features adapt slightly to your domain.

> **Math Callout:** Fine-tuning with a high learning rate destroys the
> pretrained features because large gradient steps push the weights far from
> their pretrained values. Use 1/10th to 1/100th of the LR you would use for
> training from scratch. See 1D-04 for learning rate intuition.

## "Wrong Way" Challenge: High Learning Rate

Fine-tune a pretrained ResNet with `lr=0.1` (way too high). Watch the
accuracy drop *below* a randomly initialized model. The pretrained features
are destroyed in the first few gradient steps.

## Style Notes

- Always normalize inputs with the same mean/std the pretrained model was
  trained with (ImageNet stats for torchvision models).
- Use separate parameter groups for backbone and head so you can use different
  learning rates.
- Log both training and validation loss to detect overfitting early.

## Exercises

See [exercises.py](exercises.py) for 5 stubs. Tests in
`tests/test_exercises.py`.
