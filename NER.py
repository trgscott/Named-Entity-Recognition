"""
Example usage:
    python NER.py --model bert --labels 7 --epochs 8
    python NER.py --model t5 --labels 7 --epochs 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import requests
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_conllu_using_pandas(block: str) -> pd.DataFrame:
    records = []
    for line in block.splitlines():
        if not line.startswith('#') and line.strip():
            records.append(line.strip().split('\t'))
    return pd.DataFrame.from_records(records, columns=['ID', 'FORM', 'TAG', 'Misc1', 'Misc2'])


def tokens_to_labels(df: pd.DataFrame):
    return df.FORM.tolist(), df.TAG.tolist()


def download_ner_data() -> dict:
    prefix = "https://raw.githubusercontent.com/UniversalNER/"
    data_urls = {
        "en_ewt": {
            "train": "UNER_English-EWT/master/en_ewt-ud-train.iob2",
            "dev": "UNER_English-EWT/master/en_ewt-ud-dev.iob2",
            "test": "UNER_English-EWT/master/en_ewt-ud-test.iob2",
        },
        "en_pud": {
            "test": "UNER_English-PUD/master/en_pud-ud-test.iob2",
        },
    }

    def fetch_url(url: str) -> str:
        response = requests.get(url, timeout=60, verify=False)
        response.raise_for_status()
        return response.text

    data_dict = defaultdict(dict)
    for corpus, split_dict in data_urls.items():
        for split, url_suffix in split_dict.items():
            url = prefix + url_suffix
            text = fetch_url(url)
            data_frames = map(parse_conllu_using_pandas, text.split('\n\n'))
            token_label_alignments = list(map(tokens_to_labels, data_frames))
            data_dict[corpus][split] = token_label_alignments

    save_path = Path(__file__).resolve().parent / 'ner_data_dict.json'
    with open(save_path, 'w', encoding='utf-8') as out:
        json.dump(data_dict, out, indent=2, ensure_ascii=False)
    return data_dict


def load_or_download_ner_data(force_refresh: bool = False) -> dict:
    save_path = Path(__file__).resolve().parent / 'ner_data_dict.json'
    if force_refresh or not save_path.exists():
        return download_ner_data()
    with open(save_path, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def pair_data(data):
    paired_data = []
    for tokens, labels in data:
        paired_example = []
        for i in range(len(tokens)):
            paired_example.append([tokens[i], labels[i]])
        paired_data.append(paired_example)

    label_counts = Counter(label for sentence in paired_data for _, label in sentence)
    print(label_counts)
    return paired_data


def substitute_labels(data, old_label, new_label):
    new_data = []
    for sentence in data:
        new_example = []
        for token, label in sentence:
            if label == old_label:
                new_example.append([token, new_label])
            else:
                new_example.append([token, label])
        new_data.append(new_example)
    return new_data


def substitute_all_labels(data):
    new_one = substitute_labels(data, 'B-LOC', 'B')
    new_two = substitute_labels(new_one, 'B-ORG', 'B')
    new_three = substitute_labels(new_two, 'B-PER', 'B')
    new_four = substitute_labels(new_three, 'I-LOC', 'I')
    new_five = substitute_labels(new_four, 'I-ORG', 'I')
    new_six = substitute_labels(new_five, 'I-PER', 'I')
    return new_six


def create_labels_and_classes(data):
    labels = set()
    for example in data:
        labels.update([el[1] for el in example])
    n_classes = len(labels)
    return sorted(labels), n_classes


def get_split_data(data_dict, labels_choice: int, max_examples: int | None = None):
    train_limit = min(12543, max_examples) if max_examples is not None else 12543
    dev_limit = min(2001, max_examples) if max_examples is not None else 2001
    test_limit = min(2077, max_examples) if max_examples is not None else 2077
    ood_limit = min(1000, max_examples) if max_examples is not None else 1000
    paired_data_train = pair_data(data_dict['en_ewt']['train'][0:train_limit])
    paired_data_dev = pair_data(data_dict['en_ewt']['dev'][0:dev_limit])
    paired_data_test = pair_data(data_dict['en_ewt']['test'][0:test_limit])
    paired_data_ood = pair_data(data_dict['en_pud']['test'][0:ood_limit])

    random.shuffle(paired_data_train)
    random.shuffle(paired_data_dev)
    random.shuffle(paired_data_test)
    random.shuffle(paired_data_ood)

    three_data_train = substitute_all_labels(paired_data_train)
    three_data_dev = substitute_all_labels(paired_data_dev)
    three_data_test = substitute_all_labels(paired_data_test)
    three_data_ood = substitute_all_labels(paired_data_ood)

    if labels_choice == 3:
        training_data = three_data_train
        dev_data = three_data_dev
        test_data = three_data_test
        ood_data = three_data_ood
        labels, n_classes = create_labels_and_classes(training_data)
    else:
        training_data = paired_data_train
        dev_data = paired_data_dev
        test_data = paired_data_test
        ood_data = paired_data_ood
        labels, n_classes = create_labels_and_classes(training_data)

    return {
        'train': training_data,
        'dev': dev_data,
        'test': test_data,
        'ood': ood_data,
        'labels': labels,
        'n_classes': n_classes,
    }


class ClassificationHead(nn.Module):
    def __init__(self, model_dim: int = 768, n_classes: int = 7):
        super().__init__()
        self.linear = nn.Linear(model_dim, model_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(model_dim, n_classes)

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.linear2(x)


def process_sentence(sentence, label_to_i, tokenizer, encoder, clf_head, encoder_device, clf_head_device):
    gold_labels = torch.tensor([label_to_i[label] for _, label in sentence]).to(clf_head_device)
    words = [word for word, _ in sentence]
    tokenisation = tokenizer(words, is_split_into_words=True, return_tensors='pt')
    inputs = {k: v.to(encoder_device) for k, v in tokenisation.items()}
    outputs = encoder(**inputs).last_hidden_state[0, 1:-1, :]

    word_ids = tokenisation.word_ids()[1:-1]
    processed_words = set()
    first_subword_embeddings = []
    for i, word_id in enumerate(word_ids):
        if word_id not in processed_words:
            first_subword_embeddings.append(outputs[i])
            processed_words.add(word_id)

    assert len(first_subword_embeddings) == gold_labels.size(0)
    clf_head_inputs = torch.vstack(first_subword_embeddings).to(clf_head_device)
    return clf_head(clf_head_inputs), gold_labels


def train_epoch_bert(data, label_to_i, tokenizer, encoder, clf_head, encoder_device, clf_head_device, loss_fn, optimiser):
    encoder.train()
    epoch_losses = torch.empty(len(data))
    for step_n, sentence in tqdm(enumerate(data), total=len(data), desc='Train', leave=False):
        optimiser.zero_grad()
        logits, gold_labels = process_sentence(sentence, label_to_i, tokenizer, encoder, clf_head, encoder_device, clf_head_device)
        loss = loss_fn(logits, gold_labels)
        loss.backward()
        optimiser.step()
        epoch_losses[step_n] = loss.item()
    return epoch_losses.mean().item()


def validate_epoch_bert(data, label_to_i, tokenizer, encoder, clf_head, encoder_device, clf_head_device, i_to_label, n_classes):
    encoder.eval()
    epoch_accuracies = torch.empty(len(data))
    all_predictions = []
    all_labels = []
    for step_n, sentence in tqdm(enumerate(data), total=len(data), desc='Eval', leave=False):
        with torch.no_grad():
            logits, gold_labels = process_sentence(sentence, label_to_i, tokenizer, encoder, clf_head, encoder_device, clf_head_device)
        predicted_labels = torch.argmax(logits, dim=-1)
        epoch_accuracies[step_n] = (predicted_labels == gold_labels).sum().item() / len(sentence)
        all_predictions.extend(predicted_labels.cpu().numpy())
        all_labels.extend(gold_labels.cpu().numpy())

    f1_scores = {}
    precision_scores = {}
    recall_scores = {}
    for label_index in range(n_classes):
        binary_gold = np.array(all_labels) == label_index
        binary_pred = np.array(all_predictions) == label_index
        f1_scores[i_to_label[label_index]] = f1_score(binary_gold, binary_pred, average='binary')
        precision_scores[i_to_label[label_index]] = precision_score(binary_gold, binary_pred, average='binary')
        recall_scores[i_to_label[label_index]] = recall_score(binary_gold, binary_pred, average='binary')

    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')
    return epoch_accuracies.mean().item(), f1_scores, macro_f1, predicted_labels, micro_f1, precision_scores, recall_scores


def span_accuracy_bert(data, label_to_i, tokenizer, encoder, clf_head, encoder_device, clf_head_device):
    encoder.eval()
    correct_sentences = 0
    total_sentences = len(data)
    for sentence in data:
        with torch.no_grad():
            logits, gold_labels = process_sentence(sentence, label_to_i, tokenizer, encoder, clf_head, encoder_device, clf_head_device)
        predicted_labels = torch.argmax(logits, dim=-1)
        if torch.all(predicted_labels == gold_labels):
            correct_sentences += 1
    return correct_sentences / total_sentences


def prepare_sentence(sentence_array):
    words = []
    labels = []
    for word, label in sentence_array:
        words.append(word)
        labels.append(label)
    prepared_inputs = []
    for i in range(len(words)):
        tmp = words[:i] + ['~', words[i], '~'] + words[i + 1:]
        prepared_inputs.append(' '.join(tmp))
    return prepared_inputs, labels


def process_batch_t5(batch_inputs, batch_labels, tokenizer, model, device, optimiser, max_len=512):
    optimiser.zero_grad()
    tokenisation = tokenizer(batch_inputs, return_tensors='pt', max_length=max_len, padding='longest', truncation=True)
    input_ids = tokenisation.input_ids.to(device)
    attention_mask = tokenisation.attention_mask.to(device)
    labels = tokenizer(batch_labels, return_tensors='pt', max_length=max_len, padding='longest', truncation=True).input_ids.to(device)
    labels[labels == tokenizer.pad_token_id] = -100
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    loss = model(**inputs).loss
    loss.backward()
    optimiser.step()
    return loss.item()


def train_epoch_t5(train_inputs, batch_size, tokenizer, model, device, optimizer):
    model.train()
    epoch_losses = torch.zeros(len(train_inputs))
    for step_n in tqdm(range(len(train_inputs)), leave=False, desc='Train'):
        prepared_inputs, labels = prepare_sentence(train_inputs[step_n])
        n_batches = int(np.ceil(len(prepared_inputs) / batch_size))
        sentence_losses_accum = 0.0
        for batch_index in range(n_batches):
            lo = batch_index * batch_size
            hi = lo + batch_size
            batch_texts = prepared_inputs[lo:hi]
            batch_labels = labels[lo:hi]
            loss = process_batch_t5(batch_texts, batch_labels, tokenizer, model, device, optimizer)
            sentence_losses_accum += loss
        epoch_losses[step_n] = sentence_losses_accum / max(n_batches, 1)
    return epoch_losses.mean().item()


def get_class_prediction(prompt, tokenizer, model, device, max_len=512):
    tokenisation = tokenizer(prompt, return_tensors='pt', max_length=max_len, truncation=True)
    input_ids = tokenisation.input_ids.to(device)
    attention_mask = tokenisation.attention_mask.to(device)
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=4).squeeze()
    output_string = tokenizer.decode(output, skip_special_tokens=True).strip()
    if not output_string:
        return None
    return output_string.split()[0]


def validate_epoch_t5(dev_inputs, tokenizer, model, device, max_len=512):
    model.eval()
    epoch_hits = []
    all_predictions = []
    all_labels = []
    for step_n in tqdm(range(len(dev_inputs)), leave=False, desc='Validate'):
        prepared_inputs, labels = prepare_sentence(dev_inputs[step_n])
        with torch.no_grad():
            for input_sentence, gold_label in zip(prepared_inputs, labels):
                predicted_label = get_class_prediction(input_sentence, tokenizer, model, device, max_len=max_len)
                epoch_hits.append(int(predicted_label == gold_label))
                all_predictions.append(predicted_label)
                all_labels.append(gold_label)

    labels_set = set(all_labels)
    f1_scores = {}
    for label in labels_set:
        f1 = f1_score(np.array(all_labels) == label, np.array(all_predictions) == label, average='binary')
        f1_scores[label] = f1

    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    return sum(epoch_hits) / len(epoch_hits), f1_scores, macro_f1


def calculate_f1_scores_t5(data, tokenizer, model, device):
    model.eval()
    all_predictions = []
    all_labels = []
    for sentence in tqdm(data, leave=False, desc='F1 scores'):
        prepared_inputs, labels = prepare_sentence(sentence)
        with torch.no_grad():
            for input_sentence, gold_label in zip(prepared_inputs, labels):
                predicted_label = get_class_prediction(input_sentence, tokenizer, model, device)
                all_predictions.append(predicted_label)
                all_labels.append(gold_label)

    label_set = set(all_labels)
    f1_scores = {}
    for label in label_set:
        f1 = f1_score(np.array(all_labels) == label, np.array(all_predictions) == label, average='binary')
        f1_scores[label] = f1

    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    return f1_scores, macro_f1


def span_accuracy_t5(data, tokenizer, model, device):
    model.eval()
    correct_sentences = 0
    total_sentences = len(data)
    for sentence in data:
        prepared_inputs, labels = prepare_sentence(sentence)
        sentence_correct = True
        for input_sentence, gold_label in zip(prepared_inputs, labels):
            with torch.no_grad():
                predicted_label = get_class_prediction(input_sentence, tokenizer, model, device)
            if predicted_label != gold_label:
                sentence_correct = False
                break
        if sentence_correct:
            correct_sentences += 1
    return correct_sentences / total_sentences


def save_metrics_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['model', 'label_count', 'dataset', 'label', 'metric', 'value']
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_metric_row(rows, model_name, label_count, dataset_name, label, metric_name, value):
    rows.append({
        'model': model_name,
        'label_count': label_count,
        'dataset': dataset_name,
        'label': label,
        'metric': metric_name,
        'value': value,
    })


def run_bert_experiment(args, data_dict):
    split_data = get_split_data(data_dict, args.labels, args.max_examples)
    training_data = split_data['train']
    dev_data = split_data['dev']
    test_data = split_data['test']
    ood_data = split_data['ood']
    labels = split_data['labels']
    n_classes = split_data['n_classes']

    print(f'You are running this with {n_classes} labels, are you sure?')
    print(f'Your labels are {labels}')

    label_to_i = {label: i for i, label in enumerate(sorted(labels))}
    i_to_label = {i: label for label, i in label_to_i.items()}

    model_tag = 'google-bert/bert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_tag)
    encoder = AutoModel.from_pretrained(model_tag).to(args.device)
    clf_head = ClassificationHead(n_classes=n_classes)
    clf_head.to(args.device)

    n_epochs = args.epochs if args.epochs is not None else 8
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(list(encoder.parameters()) + list(clf_head.parameters()), lr=10 ** (-5))

    best_f1 = 0.0
    last_epoch_with_dev_improvement = 0
    early_stopping_threshold = 3
    metrics_rows = []

    for epoch_n in tqdm(range(n_epochs), desc='BERT training'):
        loss = train_epoch_bert(training_data, label_to_i, tokenizer, encoder, clf_head, args.device, args.device, loss_fn, optimiser)
        print(f'Epoch {epoch_n + 1} training loss: {loss:.2f}')

        accuracy, _, macro_f1, _, micro_f1, _, _ = validate_epoch_bert(dev_data, label_to_i, tokenizer, encoder, clf_head, args.device, args.device, i_to_label, n_classes)
        print(f'Epoch {epoch_n + 1} dev accuracy: {accuracy:.2f}')
        print(f'Epoch {epoch_n + 1} dev macro f1: {macro_f1:.2f}')
        print(f'Epoch {epoch_n + 1} dev micro f1: {micro_f1:.2f}')

        add_metric_row(metrics_rows, 'bert', n_classes, 'dev', None, 'accuracy', accuracy)
        add_metric_row(metrics_rows, 'bert', n_classes, 'dev', None, 'macro_f1', macro_f1)
        add_metric_row(metrics_rows, 'bert', n_classes, 'dev', None, 'micro_f1', micro_f1)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            last_epoch_with_dev_improvement = epoch_n
            for path in glob(str(Path(__file__).resolve().parent / '*.pt')):
                os.remove(path)
            torch.save(encoder.state_dict(), str(Path(__file__).resolve().parent / 'best_encoder.pt'))
            torch.save(clf_head.state_dict(), str(Path(__file__).resolve().parent / 'best_clf_head.pt'))
        else:
            n_epochs_without_improvement = epoch_n - last_epoch_with_dev_improvement
            if n_epochs_without_improvement == early_stopping_threshold:
                print(f'{n_epochs_without_improvement} without improvement; early stopping.')
                break

    # evaluate on test/OoD data using final in-memory model for this notebook-style workflow
    _, f1_scores, macro_f1, _, micro_f1, precision_scores, recall_scores = validate_epoch_bert(test_data, label_to_i, tokenizer, encoder, clf_head, args.device, args.device, i_to_label, n_classes)
    test_span_accuracy = span_accuracy_bert(test_data, label_to_i, tokenizer, encoder, clf_head, args.device, args.device)
    print(f'Test macro F1: {macro_f1:.2f}')
    print('F1 per label on test set:')
    for label, score in f1_scores.items():
        print(f'{label}: {score}')
    print(f'Span accuracy on test set: {test_span_accuracy}')
    add_metric_row(metrics_rows, 'bert', n_classes, 'test', None, 'macro_f1', macro_f1)
    add_metric_row(metrics_rows, 'bert', n_classes, 'test', None, 'micro_f1', micro_f1)
    add_metric_row(metrics_rows, 'bert', n_classes, 'test', None, 'span_accuracy', test_span_accuracy)
    for label, score in f1_scores.items():
        add_metric_row(metrics_rows, 'bert', n_classes, 'test', label, 'f1', score)
    for label, score in precision_scores.items():
        add_metric_row(metrics_rows, 'bert', n_classes, 'test', label, 'precision', score)
    for label, score in recall_scores.items():
        add_metric_row(metrics_rows, 'bert', n_classes, 'test', label, 'recall', score)

    _, f1_scores_ood, macro_f1_ood, _, micro_f1_ood, precision_scores_ood, recall_scores_ood = validate_epoch_bert(ood_data, label_to_i, tokenizer, encoder, clf_head, args.device, args.device, i_to_label, n_classes)
    ood_span_accuracy = span_accuracy_bert(ood_data, label_to_i, tokenizer, encoder, clf_head, args.device, args.device)
    print(f'OoD macro F1: {macro_f1_ood:.2f}')
    print('F1 per label on OoD set:')
    for label, score in f1_scores_ood.items():
        print(f'{label}: {score}')
    print(f'Span accuracy on OoD set: {ood_span_accuracy}')
    add_metric_row(metrics_rows, 'bert', n_classes, 'ood', None, 'macro_f1', macro_f1_ood)
    add_metric_row(metrics_rows, 'bert', n_classes, 'ood', None, 'micro_f1', micro_f1_ood)
    add_metric_row(metrics_rows, 'bert', n_classes, 'ood', None, 'span_accuracy', ood_span_accuracy)
    for label, score in f1_scores_ood.items():
        add_metric_row(metrics_rows, 'bert', n_classes, 'ood', label, 'f1', score)
    for label, score in precision_scores_ood.items():
        add_metric_row(metrics_rows, 'bert', n_classes, 'ood', label, 'precision', score)
    for label, score in recall_scores_ood.items():
        add_metric_row(metrics_rows, 'bert', n_classes, 'ood', label, 'recall', score)

    save_metrics_csv(metrics_rows, args.metrics_path)
    print(f'Metrics saved to {args.metrics_path}')


def run_t5_experiment(args, data_dict):
    split_data = get_split_data(data_dict, args.labels, args.max_examples)
    training_data = split_data['train']
    dev_data = split_data['dev']
    test_data = split_data['test']
    ood_data = split_data['ood']
    labels = split_data['labels']
    n_classes = split_data['n_classes']

    print(f'You are running this with {n_classes} labels, are you sure?')
    print(f'Your labels are {labels}')

    model_tag = 'google-t5/t5-small'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_tag, cache_dir=str(Path(__file__).resolve().parent / 'hf_cache')).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_tag)
    optim = torch.optim.AdamW(model.parameters(), lr=10 ** (-4))

    n_epochs = args.epochs if args.epochs is not None else 4
    batch_size = args.batch_size
    best_f1 = 0.0
    last_epoch_with_dev_improvement = 0
    early_stopping_threshold = 2
    metrics_rows = []

    for epoch_n in tqdm(range(n_epochs), desc='T5 training'):
        epoch_loss = train_epoch_t5(training_data, batch_size, tokenizer, model, args.device, optim)
        print(f'Epoch {epoch_n + 1} loss: {round(epoch_loss, 2)}')
        epoch_dev_accuracy, _, macro_f1 = validate_epoch_t5(dev_data, tokenizer, model, args.device)
        print(f'Epoch {epoch_n + 1} dev accuracy: {epoch_dev_accuracy:.2f}')
        print(f'Epoch {epoch_n + 1} dev macro F1: {macro_f1:.2f}')
        add_metric_row(metrics_rows, 't5', n_classes, 'dev', None, 'accuracy', epoch_dev_accuracy)
        add_metric_row(metrics_rows, 't5', n_classes, 'dev', None, 'macro_f1', macro_f1)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            last_epoch_with_dev_improvement = epoch_n
            print('Saving the model.')
            for path in glob(str(Path(__file__).resolve().parent / '*.pt')):
                os.remove(path)
            torch.save(model.state_dict(), str(Path(__file__).resolve().parent / 'best_t5_model.pt'))
        else:
            n_epochs_without_improvement = epoch_n - last_epoch_with_dev_improvement
            if n_epochs_without_improvement == early_stopping_threshold:
                print(f'{n_epochs_without_improvement} without improvement; early stopping.')
                break

    test_span_accuracy = span_accuracy_t5(test_data, tokenizer, model, args.device)
    print(f'Span Accuracy on test set: {test_span_accuracy}')
    f1_scores, macro_f1 = calculate_f1_scores_t5(test_data, tokenizer, model, args.device)
    print('F1 scores per label on test data:')
    for label, score in f1_scores.items():
        print(f'{label}: {score}')
    print(f'Macro F1 score on test data: {macro_f1}')
    add_metric_row(metrics_rows, 't5', n_classes, 'test', None, 'macro_f1', macro_f1)
    add_metric_row(metrics_rows, 't5', n_classes, 'test', None, 'span_accuracy', test_span_accuracy)
    for label, score in f1_scores.items():
        add_metric_row(metrics_rows, 't5', n_classes, 'test', label, 'f1', score)

    ood_span_accuracy = span_accuracy_t5(ood_data, tokenizer, model, args.device)
    print(f'Span Accuracy on OoD set: {ood_span_accuracy}')
    f1_scores_ood, macro_f1_ood = calculate_f1_scores_t5(ood_data, tokenizer, model, args.device)
    print('F1 scores per label on OoD data:')
    for label, score in f1_scores_ood.items():
        print(f'{label}: {score}')
    print(f'Macro F1 score on OoD data: {macro_f1_ood}')
    add_metric_row(metrics_rows, 't5', n_classes, 'ood', None, 'macro_f1', macro_f1_ood)
    add_metric_row(metrics_rows, 't5', n_classes, 'ood', None, 'span_accuracy', ood_span_accuracy)
    for label, score in f1_scores_ood.items():
        add_metric_row(metrics_rows, 't5', n_classes, 'ood', label, 'f1', score)

    save_metrics_csv(metrics_rows, args.metrics_path)
    print(f'Metrics saved to {args.metrics_path}')


def main():
    parser = argparse.ArgumentParser(description='Run the NER notebook as a script.')
    parser.add_argument('--model', choices=['bert', 't5'], default='bert', help='Which model variant to run.')
    parser.add_argument('--labels', type=int, choices=[3, 7], default=7, help='Use 3 labels (B/I/O) or 7 labels (entity types).')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train for. Defaults follow the notebook settings.')
    parser.add_argument('--max-examples', type=int, default=None, help='Maximum number of examples to use from each data split.')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for T5 word-level prompting.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cpu, mps, or cuda).')
    parser.add_argument('--force-download', action='store_true', help='Re-download the Universal NER data even if a cache file exists.')
    parser.add_argument('--metrics-path', default=str(Path(__file__).resolve().parent / 'ner_metrics.csv'), help='Where to save the evaluation metrics CSV.')
    args = parser.parse_args()

    set_seed(args.seed)
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)

    data_dict = load_or_download_ner_data(force_refresh=args.force_download)

    if args.model == 'bert':
        run_bert_experiment(args, data_dict)
    else:
        run_t5_experiment(args, data_dict)


if __name__ == '__main__':
    main()
