# Module 08: Capstone -- Transformer from Scratch

> Build a complete, well-documented Transformer implementation in PyTorch.
> Support encoder-only (BERT-style) and decoder-only (GPT-style) configurations.
> Train a small GPT-style model on text and generate coherent output.

## Requirements

1. **`transformer.py`**: Complete implementation with:
   - Multi-head attention, positional encoding, layer normalization
   - Residual connections, causal masking
   - Encoder-only and decoder-only configurations

2. **`train.py`**: Training script with:
   - Character-level or BPE tokenization
   - Configurable hyperparameters
   - Loss logging and checkpointing

3. **Visualization notebook**:
   - Attention heatmaps for generated text
   - Loss curves
   - Generated samples at different training checkpoints

4. **1-page analysis**: Compare your implementation against PyTorch's nn.Transformer.

## Acceptance Criteria

- Model generates recognizable text after training
- Attention patterns show meaningful structure
- All tests pass
