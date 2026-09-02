# Named Entity Recognition with BERT and T5-Small

This project trains and evaluates two Named Entity Recognition (NER) systems on UniversalNER data:

- **BERT-Base Cased**, an encoder-only model with a token-classification head.
- **T5-Small**, an encoder-decoder model trained with word-level prompting.

The script supports both the seven-label entity-tagging task and a three-label version. It reports token-level accuracy, per-label precision/recall/F1 where implemented, macro F1, micro F1 for BERT, and span accuracy. A span is counted as correct only when every label in the sentence is predicted correctly. The random seed defaults to 42.

## Requirements

Use Python 3.10 or newer. Install the dependencies in the environment used to run the script:

```sh
python -m pip install numpy pandas requests scikit-learn torch tqdm transformers
```

A GPU or Apple Silicon device is recommended. Full training can require substantial memory, particularly for T5. The script automatically selects `mps`, then CUDA, then CPU when no device is supplied.

## Data and model downloads

On the first run, the script downloads the following UniversalNER splits from GitHub:

- `en_ewt`: train, development, and test data
- `en_pud`: an out-of-domain test set

The downloaded data is cached as `ner_data_dict.json` beside `NER.py`. Hugging Face model files are cached by the Transformers library; T5 also uses the local `hf_cache` directory.

The script uses the following pretrained models:

- `google-bert/bert-base-cased`
- `google-t5/t5-small`

Internet access is therefore required for the first data/model download.

## Usage

Run commands from the `CL2` directory:

```sh
python NER.py --model bert --labels 7
python NER.py --model t5 --labels 7
```

The default settings are:

- BERT: 8 epochs, learning rate `1e-5`
- T5: 4 epochs, learning rate `1e-4`
- Labels: 7
- T5 batch size: 256
- Seed: 42
- Device: automatically selected as `mps`, `cuda`, or `cpu`

For a quick smoke test, limit the number of sentences in every split and train for one epoch:

```sh
python NER.py --model bert --labels 3 --epochs 1 --max-examples 4 --device mps
```

Use `--device cuda` on an NVIDIA GPU or `--device cpu` when a GPU is unavailable. The device must be supported by the installed PyTorch build.

## Command-line options

| Option | Values/default | Description |
| --- | --- | --- |
| `--model` | `bert` or `t5`; default `bert` | Select the model architecture. |
| `--labels` | `3` or `7`; default `7` | Use B/I/O labels or entity-specific B/I/O labels. |
| `--epochs` | Optional | Override the model's default number of epochs. |
| `--max-examples` | Optional | Limit the number of examples taken from each split. |
| `--batch-size` | Default `256` | T5 batch size for word-level prompts. |
| `--seed` | Default `42` | Set the Python, NumPy, and PyTorch random seeds. |
| `--device` | Auto-selected | Choose `mps`, `cuda`, or `cpu`. |
| `--force-download` | Flag | Re-download UniversalNER data instead of using `ner_data_dict.json`. |
| `--metrics-path` | `ner_metrics.csv` beside the script | Set the output path for the metrics CSV. |

## Label schemes

With seven labels, the script uses:

```text
B-PER, B-ORG, B-LOC, I-PER, I-ORG, I-LOC, O
```

With three labels, entity types are collapsed into:

```text
B, I, O
```

## Outputs

Training uses early stopping based on development-set macro F1:

- BERT stops after three epochs without improvement.
- T5 stops after two epochs without improvement.

The best model encountered during training is saved beside `NER.py` as one of:

- `best_encoder.pt` and `best_clf_head.pt` for BERT
- `best_t5_model.pt` for T5

Evaluation results are written to the path supplied by `--metrics-path`, which defaults to `ner_metrics.csv`. The CSV contains the model, label count, dataset (`dev`, `test`, or `ood`), label, metric, and value columns.

The `ood` results are calculated on the UniversalNER English PUD test split, while the regular test results use the English EWT test split.

## Reproducibility notes

The default seed is 42 and can be changed with `--seed`. Results can still vary across devices and PyTorch versions, especially when using GPU or Apple Silicon acceleration. The script evaluates the final in-memory model after training; the saved checkpoint is the model selected by development macro F1.
