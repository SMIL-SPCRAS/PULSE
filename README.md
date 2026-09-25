# PULSE

PULSE is an audio-based multi-task learning framework for joint affective and psychological state estimation from pre-extracted speech embeddings. A single shared encoder is trained with four task-specific heads on CMU-MOSEI, RESD, First Impressions V2, and BAH.

The repository also contains TRACE, a phase-based multi-task optimization strategy with validation-driven task constraints and recovery-aware updates.

The repository is organized into two main parts:

- `app/` — source code of the developed PULSE application. The application entry point is `app/app.py`.
- `trainer/` — research and training code for the multi-task learning experiments described below.

## Tasks and datasets

| Dataset | Task | Output | Metrics |
| --- | --- | --- | --- |
| CMU-MOSEI | Emotion recognition | 7-dimensional multi-label prediction | mF1, mUAR |
| RESD | Emotion recognition | 7-class classification | mF1, mUAR |
| First Impressions V2 | Personality estimation | 5-dimensional regression | ACC, CCC |
| BAH | Ambivalence / hesitancy recognition | Binary classification | Macro-F1, UAR |

## Model

The model consists of one shared audio encoder and four task-specific heads.

Supported encoders:

- `mamba`
- `transformer`
- `mhla`
- `zeros`

Supported multi-task optimization methods:

- `trace`
- `autolambda`
- `stch`
- `taskgroup`
- `ntkmtl`
- `fairgrad`
- `none`

TRACE divides training into warm-up, conflict-aware, refinement, and late-refinement phases. Validation/control performance is used to update task constraints and activate recovery-aware learning-rate multipliers when required.

## Repository structure

```text
.
├── app/
│   ├── app.py
│   ├── config.toml
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── pulse/
│   │   ├── audio/
│   │   ├── events/
│   │   ├── inference/
│   │   ├── settings/
│   │   ├── transcription/
│   │   └── ui/
│   ├── scripts/
│   └── static/
└── trainer/
    ├── config.toml
    ├── requirements.txt
    ├── run_training.py
    ├── build_data.py
    ├── data/
    │   ├── collate.py
    │   ├── dataloaders.py
    │   └── dataset_builders.py
    ├── metrics/
    │   └── metrics.py
    ├── models/
    │   ├── mamba_encoder.py
    │   ├── mhla_encoder.py
    │   ├── multitask_model.py
    │   ├── transformer_encoder.py
    │   ├── zeros_attention.py
    │   └── zeros_encoder.py
    ├── mtl/
    │   ├── autolambda.py
    │   ├── fairgrad.py
    │   ├── ntkmtl.py
    │   └── stch.py
    └── training/
        ├── hyper_search.py
        └── train_loop.py
```

The remainder of this README describes the research and training code located in `trainer/`. For the developed application itself, see the source code in `app/`.

## Requirements

The training code in `trainer/` requires Python 3.11 or newer because the configuration is loaded with the standard-library `tomllib` module.

Install the training dependencies from the repository root with:

```bash
pip install -r trainer/requirements.txt
```

Alternatively:

```bash
cd my_code
pip install -r requirements.txt
```

The versions used in the reference training environment are pinned in `trainer/requirements.txt`.

## Input embeddings

Each dataset is expected to use an `.npz` file with the following structure:

```text
keys
v_0
v_1
...
v_n
```

`keys` contains sample identifiers. Each `v_i` contains the sequence of embeddings for the corresponding sample and must have shape:

```text
(T, H)
```

or:

```text
(1, T, H)
```

where `T` is the sequence length and `H` is the embedding dimension.

## Label files

### CMU-MOSEI

The label CSV must contain:

```text
video_name
Neutral
Anger
Disgust
Fear
Happiness
Sadness
Surprise
```

Separate train, validation, and test embedding/CSV pairs are configured in `trainer/config.toml`.

### RESD

The CSV must contain:

```text
name
emotion
```

The `emotion` values must use the following class mapping:

```text
anger: 0
disgust: 1
fear: 2
happiness: 3
neutral: 4
sadness: 5
enthusiasm: 6
```

### First Impressions V2

The labels CSV must contain:

```text
NAME_VIDEO
Subset
openness
conscientiousness
extraversion
agreeableness
non-neuroticism
```

`Subset` is used to select `train`, `validation`, and `test` samples. Train, validation, and test embeddings are stored separately, while the same labels CSV is used for all three splits.

### BAH

BAH uses one shared embeddings file and separate split files:

```text
train.txt
val.txt
test.txt
```

By default, each line is expected to contain:

```text
sample_id,label
```

The separator and field indices can be changed in `trainer/config.toml`.

## Configuration

All training and experiment settings are stored in `trainer/config.toml`.

The main sections are:

```text
[paths]
[data]
[training]
[model]
[mamba]
[mhla]
[mtl]
[ntkmtl]
[taskgroup]
[fairgrad]
[stch]
[autolambda]
[trace]
[grid]
```

Before running an experiment, replace all `/path/to/...` placeholders in the `[paths]` section of `trainer/config.toml` with the actual dataset locations.

Example model configuration:

```toml
[model]
encoder_type = "mamba" # "mamba", "zeros", "transformer", "mhla"
d_model = 256
n_heads = 4
num_layers = 4
dim_feedforward = 1024
dropout = 0.1
max_len = 15000
```

Example MTL configuration:

```toml
[mtl]
mt_weight_method = "trace" # "trace", "stch", "autolambda", "taskgroup", "ntkmtl", "fairgrad", "none"
mt_max_norm = 1.0
```

## Hyperparameter search

`trainer/run_training.py` always reads the `[grid]` section from `trainer/config.toml` and performs grid search over all listed values.

For a single run, keep one value for every grid parameter:

```toml
[grid]
mt_weight_method = ["trace"]
encoder_type = ["mamba"]
num_layers = [4]
d_model = [256]
batch_size = [16]
```

For multiple configurations, add more values:

```toml
[grid]
d_model = [128, 256]
batch_size = [8, 16]
```

Every combination is trained independently. The best combination is selected using the metric specified by:

```toml
[training]
selection_metric = "mean_all"
```

## Validation and model selection

Validation/control data are always used for model selection.

The protocol is:

```text
CMU-MOSEI -> validation split
RESD      -> test split used as validation/control
FIv2      -> validation split
BAH       -> validation split
```

The validation/control metric is used for:

- TRACE feedback and task constraints;
- Auto-Lambda meta-updates;
- early stopping;
- best-epoch selection;
- best-checkpoint selection;
- hyperparameter selection.

For CMU-MOSEI, FIv2, and BAH, the test split is evaluated only after the best validation checkpoint has been loaded.

## Training

After editing `trainer/config.toml`, run:

```bash
cd my_code
python run_training.py
```

No command-line arguments are required. `run_training.py` resolves `config.toml` relative to its own location.

The training script:

1. reads `config.toml`;
2. creates train, validation/control, and test loaders;
3. generates all configurations from `[grid]`;
4. trains each configuration;
5. selects checkpoints using validation/control performance;
6. evaluates the best validation checkpoint on the test splits;
7. reports the best hyperparameter combination.

## Outputs

When training is launched from `trainer/` as shown above, checkpoints are written by default to:

```text
trainer/checkpoints/
```

and grid-search logs to:

```text
trainer/grid_search.log
```

Each training run creates its own directory containing files such as:

```text
best_model.pt
config_snapshot.json
run_meta.json
```

The checkpoint stores the model state together with selection, validation, and final test metrics.
