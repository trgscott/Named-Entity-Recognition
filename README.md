# Named Entity Recognition with BERT (encoder-only) and T5-Small (encoder-decoder)

<!-- ABOUT THE PROJECT -->
## About The Project

This code relates to an assignment on two Named Entity Recognition (NER) tasks (tagging seven or three classes) using an encoder-only (BERT-Base) and an encoder-decoder (T5-Small) model. The models are evaluated using span accuracy, per label F1 and macro-F1. Options to freeze the lower layers of the encoder model and vary the learning rates on the encoder-decoder model are highlighted below. The code output currently shows the results for the best performing model options (all layers trained for BERT and 10^(-4) learning rate for T5). All random seeds are set to 42 for reproducibility.

<!-- GETTING STARTED -->
## Getting Started

All of the code required to run the project is in the CL2_Assignment.ipynb file, which can be run in Colab. A GPU runtime and 18.3GB of GPU RAM is required. For the exact same results you may need to run on the L4 GPU on Colab though it can be run on others. The code is currently set up to run in Colab. If you would like to run this on Manchester University GPU nodes you will need to amend the code as noted below and follow the instructions [here](https://livemanchesterac-my.sharepoint.com/:w:/g/personal/dmitry_nikolaev_manchester_ac_uk/EQVPI6GKWN5LsYQoHkFOItAB05Nv6EeRyZDhzuNjFwPcuw):

* Set the devices
  ```sh
  # set to 0 when on university system, "cuda" when using Colab:
  encoder_device = "cuda"
  # set to 0 when on university system, "cuda" when using Colab:
  clf_head_device = "cuda"
  # set to 0 when on university system, "cuda" when using Colab:
  device = "cuda" if torch.cuda.is_available() else 'cpu'
  ```

### Prerequisites

These are the libraries that will need to be installed:
* Installations
  ```sh
  !pip install datasets
  ```
  * Libraries
  ```sh
  from collections import defaultdict, Counter
  from urllib import request
  import json
  import pandas as pd
  from random import shuffle
  from math import ceil
  import torch
  import torch.nn as nn
  from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler, BitsAndBytesConfig
  import datasets
  from tqdm.auto import tqdm
  import random
  import numpy as np
  from sklearn.metrics import f1_score, precision_score, recall_score
  from glob import glob
  import os
  ```

<!-- USAGE EXAMPLES -->
## Usage

The code can be run entirely to produce the results for fine-tuning and testing bert-base-cased and T5-small on [Universal NER](https://www.universalner.org) data. The code is currently set up to tag seven labels (B-PER, B-ORG, B-LOC, I-PER, I-ORG, I-LOC, and O). If you would like to run the code on three labels (B, I, and O), amend the block of code below as shown before re-running the training and testing:

* Set the task - three labels or seven labels - reset as needed
  ```sh
  #Change THIS_MANY_LABELS to three or seven depending on the task



  THIS_MANY_LABELS = seven




  ```
The code is currently set up to train all layers of BERT. If you would prefer to freeze encoder layers 0 and 1 for BERT simply uncomment out this code:

* #Experiment with freezing first few layers of the encoder:
 ```sh
 #for name, param in encoder.named_parameters():
 #    if name.startswith("bert.encoder.layer.0") or name.startswith("bert.encoder.layer.1"):
 #        param.requires_grad = False

 #optimizer_parameters = [param for name, param in encoder.named_parameters()
 #    if not (name.startswith("bert.encoder.layer.0") or name.startswith("bert.encoder.layer.1"))
 #] + list(clf_head.parameters())
  ```
... and amend the optimizer_parameters in the training code block from:

```sh
optimiser = torch.optim.AdamW(list(encoder.parameters()) + list(clf_head.parameters()), lr=10**(-5))
  ```
to:

```sh
optimiser = torch.optim.AdamW(optimizer_parameters, lr=10**(-5))
  ```

The code for T5 is currently set up to use a learning rate of 10^(-4). If you would like to vary this rate, simply amend the lr in the below block of code:

* Encoder-Decoder - T5-Small
 ```sh
  model_tag = 'google-t5/t5-small'

  model = AutoModelForSeq2SeqLM.from_pretrained(model_tag, cache_dir='./hf_cache').to(device)
  tokeniser = AutoTokenizer.from_pretrained(model_tag)

  optim = torch.optim.AdamW(model.parameters(), lr=10^(-4))
  ```

<!-- NOTES -->
## Notes

Due to an error on Github - see [here](https://github.com/orgs/community/discussions/155944) - that requires code outputs to be deleted prior to uploading from Colab, the code outputs have been copied into text boxes in the CL2_Assignment.ipynb file for reference.

