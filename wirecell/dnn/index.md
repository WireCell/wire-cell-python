---
generated: 2026-04-29
source-hash: 901023f6498682ae
children-hash: 71434ab650cf2d54
---

# wirecell/dnn/

PyTorch-based deep neural network training framework for Wire-Cell signal processing, providing a generic CLI harness for training, evaluating, and inspecting DNN models. Sub-packages cover dataset loading, model definitions, and application-specific pipelines (ROI finding, regression, vision-transformer architectures). Experiment tracking is supported via mlflow or a filesystem fallback.

## Modules

| Module | Purpose | Key Symbols |
|---|---|---|
| `__init__` | Package entry point re-exporting sub-packages | `data`, `models`, `io`, `apps` |
| `__main__` | Click CLI harness for training and inference workflows | `cli`, `train`, `dump`, `extract`, `vizmod`, `viztrain`, `run_one` |
| `io` | Checkpoint save/load utilities wrapping `torch.save`/`torch.load` | `save_checkpoint`, `load_checkpoint`, `load_checkpoint_raw` |
| `train` | Generic supervised training loop (`Classifier`) with epoch/evaluate methods | `Classifier` |
| `tracker` | Filesystem-based experiment tracker mimicking the mlflow API; falls back if mlflow absent | `fsflow`, `flow` |
| `apps` | Per-application DNN pipelines (dnnroi, uvitrio, …) | see child index |
| `data` | HDF5-backed PyTorch datasets and train/eval splitting | see child index |
| `models` | U-Net and ViT-UNet neural network model definitions | see child index |
| `test` | Integration tests and utility scripts | see child index |

## CLI Commands

| Command | Description |
|---|---|
| `dump-config` | Print resolved configuration |
| `train` | Train a model over epochs with checkpointing and eval loss |
| `dump` | Dump run and epoch summary from a checkpoint file |
| `extract` | Extract individual samples from a dataset to `.npz` |
| `plot3p1` | Plot 3 input layers and 1 truth image per sample |
| `vizmod` | Print torchsummary and optionally write a GraphViz diagram of a model |
| `run_one` | Run a single reco/true pair through a saved model |
| `viztrain` | Plot training and eval loss curves from a checkpoint file |

## Dependencies

| Import | Role |
|---|---|
| `wirecell.dnn.io` | Checkpoint persistence used by `train` and `run_one` commands |
| `wirecell.dnn.data` | `train_eval_split` for partitioning datasets during training |
| `wirecell.dnn.apps` | Application-specific `Network`, `Dataset`, `Trainer`, `Criterion`, `Optimizer` |
| `wirecell.util.cli` | `context`, `log`, `jsonnet_loader`, `anyconfig_file` decorators |
| `wirecell.util.paths` | `unglob`, `listify` for file argument handling |
