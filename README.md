# HBS-Former

Code for **Hierarchical Bilinear Scattering Transformer for Remote-Sensing Image Scene Classification**.

HBS-Former combines a dual-stream scattering block (DWT and DTCWT), global–local dual-path feature aggregation, and hierarchical bilinear fusion for remote-sensing scene classification.

> **Release note:** This README is a starting template. Before describing the repository as fully reproducible, verify that the uploaded model and training scripts match the experiments reported in the manuscript, and complete the setup checks below.

## Project structure

```text
HBS-Former/
├── configs/svt/          # Model and training configurations
├── dwt_dtcwt.py          # HBS-Former model used by the uploaded training entry point
├── datasets.py           # Dataset loaders
├── engine.py             # Training and evaluation loops
├── loss/                 # Loss functions
├── mcloader/             # Optional data-loading components
├── util/                 # Utilities
├── main.py               # Training/evaluation entry point
├── losses.py
├── samplers.py
├── utils.py
└── requirements.txt
```

## Datasets

The code contains dataset-loading branches for:

| Dataset | Train split | Configuration name | Classes |
| --- | ---: | --- | ---: |
| NWPU-RESISC45 | 10% / 20% | `nwpu19` / `nwpu28` | 45 |
| AID | 20% / 50% | `aid28` / `aid55` | 30 |
| UC Merced | 50% / 80% | `ucm55` / `ucm82` | 21 |

Download these datasets from their respective providers. Prepare the train/test splits independently; the dataset loader expects ImageFolder-style class subdirectories. For example, AID 50%:

```text
/path/to/aid55/
├── train_50/
│   ├── airport/
│   ├── bareland/
│   └── ...
└── val/
    ├── airport/
    ├── bareland/
    └── ...
```

The same class names and class-to-index mappings must be used for training and evaluation. Check that no image appears in both subsets. To reproduce the manuscript's five-run statistics, preserve or publish the exact split-generation procedure and random seeds `0, 1, 2, 3, 4`.

## Environment

Create an environment with a Python and CUDA/PyTorch combination compatible with your GPU. Install PyTorch and torchvision according to your CUDA setup, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

**Dependency to resolve before a clean installation:** the current `main.py` and `engine.py` still import `tlt.data` for token labeling, but the uploaded code does not include a `tlt/` package. Supply the corresponding upstream source with its license, or remove the unused token-labeling imports and branches if the published RSSC code does not use this feature. The optional `mcloader/` Memcached backend also requires its own `mc` environment if enabled.

## Pretrained weights

The manuscript uses an ImageNet-1K pretrained SVT checkpoint for initialization. Download the checkpoint from an authorized source and pass its local path using `--finetune`. Model weights and datasets are **not** included in this GitHub repository.

## Training

**Before running:** confirm the public repository's `main.py` imports the model file that is actually included (`dwt_dtcwt.py`). Update the `--data-set` argument's `choices` to include `nwpu19`, `nwpu28`, `aid28`, `aid55`, `ucm55`, and `ucm82`. The current `configs/svt/svt_b.py` contains a machine-specific `output_dir` that overrides the command-line value; change it to your desired experiment directory. Resolve the `tlt` dependency described above.

Example: AID 50%, one GPU (after the above setup steps):

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 \
  --master_port=29501 \
  main.py \
  --config configs/svt/svt_b.py \
  --data-set aid55 \
  --data-path /path/to/aid55 \
  --finetune /path/to/svt_pre_imagenet.pth.tar \
  --seed 0
```

Change the dataset name, dataset path, output directory, and seed for other experiments. Check the actual command-line and config defaults before launching training.

## Evaluation

The training entry point supports `--eval` and `--resume`. After confirming the saved checkpoint format and evaluation split, you can adapt the training command with these arguments:

```bash
# Add to the command above, with your trained checkpoint:
--eval --resume /path/to/trained_checkpoint.pth.tar
```

Record the exact checkpoint, dataset split, and seed used for each reported result.

## Paper

*Hierarchical Bilinear Scattering Transformer for Remote-Sensing Image Scene Classification*.

Citation details (DOI, year, volume, and page numbers) should be added after the final publication metadata is confirmed.

## Acknowledgments and license

This project builds on existing PyTorch and vision-transformer components. Identify any reused upstream code and comply with its applicable licenses before adding a license to this repository.
