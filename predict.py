from collections import defaultdict
import torch
import numpy as np
import utils_srl, argparse
from transformers import BertTokenizer
from transformers import BertForTokenClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


if __name__ == "__main__":
    """
    RUN EXAMPLE:
        python predict.py -m saved_models/TRIAL_BERT_NER --epoch 10 --test_path data/spanish.mini.jsonl
    """

    confusion_dict = defaultdict(list)
    arg_excess, arg_missed, arg_match = defaultdict(int), defaultdict(int), defaultdict(int)

    # =====================================================================================
    #                    GET PARAMETERS
    # =====================================================================================
    # Read arguments from command line
    parser = argparse.ArgumentParser()

    # GENERAL SYSTEM PARAMS
    parser.add_argument('-t', '--test_path', help='Filepath containing the JSON File to Predict', required=True)
    parser.add_argument('-m', '--model_dir', required=True)
    parser.add_argument('-l', '--lang', default="EN")
    parser.add_argument('-e', '--epoch', help="Epoch to Load model from", required=True)
    parser.add_argument('-g', '--gold_labels', default="False")
    parser.add_argument('-v', '--eval_preds', default="True", help="Include the label V in the F1 score computation")
    parser.add_argument('-b', '--batch_size', default=16, help="For BEST results: same value as wen training the Model")
    parser.add_argument('-mx', '--seq_max_len', default=256, help="BEST results: same value as when training the Model")

    args = parser.parse_args()

    EVALUATE_PREDICATES = utils_srl.get_bool_value(args.eval_preds)
    device, USE_CUDA = utils_srl.get_torch_device()
    file_has_gold = utils_srl.get_bool_value(args.gold_labels)
    SEQ_MAX_LEN = int(args.seq_max_len)
    BATCH_SIZE = int(args.batch_size)

    # Load Saved Model
    model, tokenizer = utils_srl.load_model(BertForTokenClassification, BertTokenizer, f"{args.model_dir}/EPOCH_{args.epoch}")
    label2index = utils_srl.load_label_dict(f"{args.model_dir}/label2index.json")
    index2label = {v:k.strip("B-") for k,v in label2index.items()}

    # Load File for Predictions
    _, prediction_inputs, prediction_masks, gold_labels, seq_lens, gold_predicates = utils_srl.load_srl_dataset(args.test_path, tokenizer,
                                                                                                            include_labels=True,
                                                                                                            max_len=SEQ_MAX_LEN,
                                                                                                            label2index=label2index)

    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, gold_labels, seq_lens, gold_predicates)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

    print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []
    total_sents = 0

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_lengths, b_preds = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=b_preds, attention_mask=b_input_mask)

        logits = outputs[0] # [B, S, V]
        class_probabilities = torch.softmax(logits, dim=-1)

        # Move class_probabilities and labels to CPU
        class_probabilities = class_probabilities.detach().cpu().numpy()
        argmax_indices = np.argmax(class_probabilities, axis=-1)

        label_ids = b_labels.to('cpu').numpy()
        seq_lengths = b_lengths.to('cpu').numpy()

        for ix in range(len(label_ids)):
            total_sents += 1
            text = tokenizer.convert_ids_to_tokens(b_input_ids[ix], skip_special_tokens=True)
            # Store predictions and true labels
            pred_labels = [index2label[p] for p in argmax_indices[ix][:seq_lengths[ix]] if p != 0]
            gold_labels = [index2label[g] for g in label_ids[ix] if g != 0]
            predictions += pred_labels[:len(gold_labels)]
            true_labels += gold_labels
            # We have to evaluate ONLY the labels that belong to a Start WordPiece (not contain "##")
            eval_metrics = utils_srl.evaluate_tagset(gold_labels, pred_labels, ignore_verb_label=EVALUATE_PREDICATES)
            arg_excess, arg_missed, arg_match = utils_srl._add_to_eval_dicts(eval_metrics, arg_excess, arg_missed, arg_match)

            for j, gold in enumerate(gold_labels):
                # if "##" not in text[j] and gold not in ["X"]:
                if j < len(pred_labels): confusion_dict[gold].append(pred_labels[j])

            print(f"\n----- {total_sents} -----\n{pred_labels}\n{gold_labels}")

    # Overall Metrics
    metrics_file = f"{args.model_dir}/F1_Results_{args.lang}_{args.epoch}.txt"
    utils_srl.get_overall_metrics(arg_excess, arg_missed, arg_match, save_to_file=metrics_file, print_metrics=True)
