# SimCLR & DeepCluster on CIFAR-10

Topics in ML, Assignment 2: SimCLR and DeepCluster trained from scratch on the same CIFAR ResNet-18 and evaluated with a linear probe on frozen features (accuracies below are the best linear-probe test accuracy). All code is in [`SimCLR_DeepCluster_CIFAR10.ipynb`](SimCLR_DeepCluster_CIFAR10.ipynb), plots are in [`results/`](results/) and per-run results in `checkpoints/*/results.json`.

## Experiments

1. **SimCLR, batch size** (cells 36–40): batch sizes 128 / 256 / 512 / 1024, 40 epochs each: 77.4% / 76.1% / 71.6% / 67.7%.
2. **SimCLR, augmentation composition** (cells 41–43, 45): crop, color, cutout and rotation, alone and in pairs, 20 epochs each. Best pair: crop + color, 62.1%.
3. **SimCLR, color distortion strength** (cells 44–45): crop + color with s = 0 / 0.25 / 0.5 / 1.0: 52.4% / 61.5% / 62.1% / 59.3%.
4. **SimCLR, why color matters** (cell 46): pixel-intensity histograms of crops of one image, without and with color distortion.
5. **DeepCluster, batch size** (cells 47–51): same backbone, augmentations and batch sizes as SimCLR, 45 epochs each: 60.0% / 56.5% / 57.2% / 54.7%.
6. **DeepCluster, training duration** (cell 49): the same models after 1, 15 and 45 epochs, e.g. batch size 128: 41.6% → 51.9% → 60.0%.
7. **DeepCluster, augmentations** (cells 52–56): SimCLR's composition and color-strength experiments repeated with DeepCluster, 20 epochs each.
8. **DeepCluster, why color matters** (cell 57): how closely each run's clusters follow raw color histograms versus the true labels.
9. **SimCLR vs DeepCluster** (cells 50 and 56): accuracy per batch size and per augmentation configuration, side by side.

All runs are logged to Weights & Biases. Model weights are not included.
