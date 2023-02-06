from collections import defaultdict
import logging, argparse, torch, sys
import utils_srl
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import pipeline
from transformers import BertTokenizer, AutoModelForTokenClassification

if __name__ == "__main__":
    """
    RUN EXAMPLE:
        python3 predict.py -m saved_models/TRIAL_BERT_NER/ --epoch 10 --test_path data/spanish.mini.jsonl --gold_labels
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
    parser.add_argument('-g', '--gold_labels', action='store_true')
    parser.add_argument('-v', '--eval_preds', default="True", help="Include the label V in the F1 score computation")
    parser.add_argument('-b', '--batch_size', default=16, help="For BEST results: same value as wen training the Model")
    parser.add_argument('-mx', '--seq_max_len', default=256, help="BEST results: same value as when training the Model")

    args = parser.parse_args()

    EVALUATE_PREDICATES = utils_srl.get_bool_value(args.eval_preds)
    device, USE_CUDA = utils_srl.get_torch_device()
    file_has_gold = utils_srl.get_bool_value(args.gold_labels)
    SEQ_MAX_LEN = int(args.seq_max_len)
    BATCH_SIZE = int(args.batch_size)
    INPUTS_PATH=f"{args.model_dir}/EPOCH_{args.epoch}/model_inputs.txt"
    OUTPUTS_PATH=f"{args.model_dir}/EPOCH_{args.epoch}/model_outputs.txt"

    # Logging...
    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=f"{args.model_dir}/EPOCH_{args.epoch}/BERT_TokenClassifier_predictions.log")
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])

    # Load Model and Label Info
    model, tokenizer = utils_srl.load_model(AutoModelForTokenClassification, BertTokenizer, f"{args.model_dir}/EPOCH_{args.epoch}")
    label2index = utils_srl.load_label_dict(f"{args.model_dir}/label2index.json")
    index2label = {v:k for k,v in label2index.items()}

    if file_has_gold:
        label2index, prediction_inputs, prediction_masks, gold_labels, gold_lens, gold_preds = utils_srl.load_srl_dataset(args.test_path,
                                                                                                                    tokenizer,
                                                                                                                    max_len=SEQ_MAX_LEN,
                                                                                                                    include_labels=True,
                                                                                                                    label2index=label2index)
        prediction_data = TensorDataset(prediction_inputs, prediction_masks, gold_labels, gold_preds)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

        logging.info('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
        
        results, preds_list = utils_srl.evaluate_bert_model(prediction_dataloader, BATCH_SIZE, model, tokenizer, index2label, full_report=True, prefix="Test Set")
        logging.info("  Test Loss: {0:.2f}".format(results['loss']))
        logging.info("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision']*100, results['recall']*100, results['f1']*100))

        with open(OUTPUTS_PATH, "w") as fout:
            with open(INPUTS_PATH, "w") as fin:
                for sent, pred in preds_list:
                    fin.write(" ".join(sent)+"\n")
                    fout.write(" ".join(pred)+"\n")

    else:
        test_data = utils_srl.get_sentences(args.test_path)
        # https://huggingface.co/transformers/main_classes/pipelines.html#transformers.TokenClassificationPipeline
        logging.info('Predicting labels for {:,} test sentences...'.format(len(test_data)))
        if not USE_CUDA: GPU_IX = -1
        nlp = pipeline('token-classification', model=model, tokenizer=tokenizer, device=GPU_IX)
        nlp.ignore_labels = []
        with open(OUTPUTS_PATH, "w") as fout:
            with open(INPUTS_PATH, "w") as fin:
                for seq_ix, sentence in enumerate(test_data):
                    predicted_labels = []
                    output_obj = nlp(sentence)
                    # [print(o) for o in output_obj]
                    for tok in output_obj:
                        if '##' not in tok['word']:
                            predicted_labels.append(tok['entity'])
                    logging.info(f"\n----- {seq_ix+1} -----\n{sentence.split()}\nPRED:{predicted_labels}")
                    fin.write(sentence+"\n")
                    fout.write(" ".join(predicted_labels)+"\n")
