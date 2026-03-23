# Module: Semantic Segmentation

## References: Ronneberger et al. 2015 U-Net (arXiv:1505.04597).

## Exercises
1. `unet_encoder_block` -- Conv-BN-ReLU-Conv-BN-ReLU-MaxPool.
2. `unet_decoder_block` -- Upsample-Concat-Conv-BN-ReLU-Conv-BN-ReLU.
3. `dice_loss` -- Dice loss for segmentation.
4. `mean_iou` -- Mean IoU across classes.
