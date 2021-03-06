import datetime, json
import numpy as np
import torch
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import logging
from tabulate import tabulate # https://pypi.org/project/tabulate/


def get_bool_value(str_bool):
    if str_bool.upper() == "TRUE" or str_bool.upper() == "T":
        return True
    else:
        return False


def get_torch_device(verbose=True):
    use_cuda = False
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        use_cuda = True
        if verbose:
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        if verbose: print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device, use_cuda


device, USE_CUDA = get_torch_device(verbose=False)
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


def build_label_vocab(list_of_labels):
    label2index = {"[PAD]": 0, "[UNK]": 1}
    for i, labelset in enumerate(list_of_labels):
        for l in labelset:
            if l not in label2index:
                label2index[l] = len(label2index)
    return label2index


def expand_to_wordpieces(original_sentence, original_labels, tokenizer):
    """
    Also Expands BIO, but assigns the original label ONLY to the Head of the WordPiece (First WP)
    :param original_sentence: List of Full-Words
    :param original_labels: List of Labels corresponding to each Full-Word
    :param tokenizer: To convert it into BERT-model WordPieces
    :return:
    """

    txt_sentence = " ".join(original_sentence)

    txt_sentence = txt_sentence.replace("#", "N")
    word_pieces = tokenizer.tokenize(txt_sentence)

    tmp_labels, lbl_ix = [], 0
    head_tokens = [1] * len(word_pieces)
    for i, tok in enumerate(word_pieces):
        if "##" in tok:
            tmp_labels.append("X")
            head_tokens[i] = 0
        else:
            tmp_labels.append(original_labels[lbl_ix])
            lbl_ix += 1

    word_pieces = ["[CLS]"] + word_pieces + ["[SEP]"]
    labels = ["O"] + tmp_labels + ["O"]
    head_tokens = [0] + head_tokens + [0]
    return word_pieces, labels, head_tokens


def get_data(filepath, tokenizer, include_labels):
    sentences, verb_indicators, all_labels = [], [], []
    with open(filepath) as f:
        for i, line in enumerate(f.readlines()):
            obj = json.loads(line)
            # Get WordPiece Indices
            wordpieces, labelset, head_toks = expand_to_wordpieces(obj["seq_words"], obj["BIO"], tokenizer)
            # print(wordpieces)
            # print(labelset)
            # print("------------")
            input_ids = tokenizer.convert_tokens_to_ids(wordpieces)
            sentences.append(input_ids)
            # Verb Indicator (which predicate to label)
            bio_verb = [1 if label[-2:] == "-V" else 0 for label in labelset]
            verb_indicators.append(bio_verb)
            # Get Gold Labels (For training or for evaluation)
            if include_labels:
                all_labels.append(labelset)

    return sentences, verb_indicators, all_labels


def load_srl_dataset(filepath, tokenizer, max_len, include_labels, label2index):
    sentences, verb_indicators, labels = get_data(filepath, tokenizer, include_labels)
    seq_lengths = [len(s) for s in sentences]
    logging.info(f"MAX SEQ LENGTH IN DATASET IS {max(seq_lengths)}")
    # BUILD VOCABULARY IF NECESSARY
    label_ixs = []
    if not label2index: label2index = build_label_vocab(labels)
    # CONVERT LABELS TO THEIR INDICES
    for i, labelset in enumerate(labels):
        label_ixs.append([label2index.get(l, 1) for l in labelset])
    # PAD ALL SEQUENCES
    input_ids = pad_sequences(sentences, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    input_is_pred = pad_sequences(verb_indicators, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    if include_labels:
        label_ids = pad_sequences(label_ixs, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
        label_ids = LongTensor(label_ids)
    else:
        label_ids = None
    # Create attention masks
    attention_masks = []
    # For each sentence...
    for i, sent in enumerate(input_ids):
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    return label2index, LongTensor(input_ids), LongTensor(attention_masks), label_ids,  LongTensor(seq_lengths), LongTensor(input_is_pred)


def get_metrics(false_pos, false_neg, true_pos):
    _denom1 = true_pos + false_pos
    precision = true_pos / _denom1 if _denom1 else 0
    _denom2 = true_pos + false_neg
    recall = true_pos / _denom2 if _denom2 else 0
    _denom3 = precision + recall
    F1 = 2 * ((precision * recall) / _denom3) if _denom3 else 0
    return precision*100, recall*100, F1*100


def _add_to_eval_dicts(eval_metrics, arg_excess, arg_missed, arg_match):
    for arg in eval_metrics["excess"]:
        arg_excess[arg] += 1
    for arg in eval_metrics["missed"]:
        arg_missed[arg] += 1
    for arg in eval_metrics["match"]:
        arg_match[arg] += 1
    return arg_excess, arg_missed, arg_match


def evaluate_tagset(gold_labels, pred_labels, ignore_verb_label):
    label_filter = ["X", "O", "B-V"] if ignore_verb_label else ["X", "O"]
    gld = set([f"{i}_{y.strip('B-')}" for i, y in enumerate(gold_labels) if y not in label_filter])
    sys = set([f"{i}_{y.strip('B-')}" for i, y in enumerate(pred_labels) if y not in label_filter])

    excess = sys - gld  # False Positives
    missed = gld - sys  # False Negatives
    true_pos = sys.intersection(gld)

    eval_obj = {"excess": [x.split("_")[1] for x in excess],
                "missed": [x.split("_")[1] for x in missed],
                "match": [x.split("_")[1] for x in true_pos]}
    return eval_obj


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def save_losses(losses, filename):
    out = open(filename, "w")
    out.write(json.dumps({"losses": losses})+"\n")


def save_label_dict(label2index, filename):
    out = open(filename, "w")
    out.write(json.dumps(label2index))


def load_label_dict(modelpath):
    fp = open(modelpath)
    label_dict = json.load(fp)
    return label_dict


def get_overall_metrics(arg_excess, arg_missed, arg_match, save_to_file=None, print_metrics=True):
    # for x in arg_match.items():
    #     print(x)
    processed_args = set()
    results = []
    tot_excess, tot_missed, tot_match = 0, 0, 0
    for arg, count in arg_match.items():
        excess = arg_excess.get(arg, 0)
        missed = arg_missed.get(arg, 0)
        p,r,f = get_metrics(false_pos=excess, false_neg=missed, true_pos=count)
        processed_args.add(arg)
        results.append((arg, count, excess, missed, p, r, f))
        tot_excess += excess
        tot_missed += missed
        tot_match += count
    for arg, count in arg_excess.items():
        if arg not in processed_args:
            excess = count
            missed = arg_missed.get(arg, 0)
            correct = arg_match.get(arg, 0)
            p, r, f = get_metrics(false_pos=excess, false_neg=missed, true_pos=correct) # p,r,f = 0,0,0
            processed_args.add(arg)
            results.append((arg, correct, excess, missed, p, r, f))
            tot_excess += excess
            tot_missed += missed
            tot_match += correct
    for arg, count in arg_missed.items():
        if arg not in processed_args:
            excess = arg_excess.get(arg, 0)
            correct = arg_match.get(arg, 0)
            missed = count
            p, r, f = get_metrics(false_pos=excess, false_neg=missed, true_pos=correct) # p,r,f = 0,0,0
            results.append((arg, correct, excess, missed, p, r, f))
            tot_excess += excess
            tot_missed += missed
            tot_match += correct
    results = sorted(results, key= lambda x: x[0])

    prec, rec, F1 = get_metrics(false_pos=tot_excess, false_neg=tot_missed, true_pos=tot_match)

    if print_metrics:
        print("\n--- OVERALL ---\nCorrect: {0}\tExcess: {1}\tMissed: {2}\nPrecision: {3:.2f}\t\tRecall: {4:.2f}\nF1: {5:.2f}\n".format(tot_match, tot_excess, tot_missed, prec, rec, F1))
        print(tabulate(results, headers=["corr.", "excess", "missed", "prec.", "rec.", "F1"], floatfmt=".2f"))
    if save_to_file:
        fout = open(save_to_file, "w")
        fout.write("\n--- OVERALL ---\nCorrect: {0}\tExcess: {1}\tMissed: {2}\nPrecision: {3:.2f}\t\tRecall: {4:.2f}\nF1: {5:.2f}\n".format(tot_match, tot_excess, tot_missed, prec, rec, F1))
        fout.write(tabulate(results, headers=["corr.", "excess", "missed", "prec.", "rec.", "F1"], floatfmt=".2f"))
    return prec, rec, F1


# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
def save_model(output_dir, arg_dict, model, tokenizer):
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Good practice: save your training arguments together with the trained model
    torch.save(arg_dict, os.path.join(output_dir, 'training_args.bin'))


def load_model(model_class, tokenizer_class, model_dir):
    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(model_dir)
    tokenizer = tokenizer_class.from_pretrained(model_dir)
    # Copy the model to the GPU.
    model.to(device)
    return model, tokenizer