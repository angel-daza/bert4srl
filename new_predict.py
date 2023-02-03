from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForTokenClassification
import logging, sys
from transformers import BertTokenizer
from transformers import pipeline
from torch.utils.data import SequentialSampler
import utils_srl as utils


GPU_IX=0
device, USE_CUDA = utils.get_torch_device(GPU_IX)
FILE_HAS_GOLD = True
SEQ_MAX_LEN = 256
BATCH_SIZE = 4
TEST_DATA_PATH = "data/spanish.mini.jsonl"
MODEL_DIR = "saved_models/EN_BERT_SRL/"
LOAD_EPOCH = 5
INPUTS_PATH=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_inputs.txt"
OUTPUTS_PATH=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_outputs.txt"
PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index # -100

console_hdlr = logging.StreamHandler(sys.stdout)
file_hdlr = logging.FileHandler(filename=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/BERT_TokenClassifier_predictions.log")
logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])

model, tokenizer = utils.load_model(BertForTokenClassification, BertTokenizer, f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}")
label2index = utils.load_label_dict(f"{MODEL_DIR}/label2index.json")
index2label = {v:k for k,v in label2index.items()}


if FILE_HAS_GOLD:
    label2index, prediction_inputs, prediction_masks, gold_labels, gold_lens, gold_preds = utils.load_srl_dataset(TEST_DATA_PATH,
                                                                                                                tokenizer,
                                                                                                                max_len=SEQ_MAX_LEN,
                                                                                                                include_labels=True,
                                                                                                                label2index=label2index)
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, gold_labels, gold_preds)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

    logging.info('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
    
    results, preds_list = utils.evaluate_bert_model(prediction_dataloader, BATCH_SIZE, model, tokenizer, index2label, 
                                                        PAD_TOKEN_LABEL_ID, full_report=True, prefix="Test Set")
    logging.info("  Test Loss: {0:.2f}".format(results['loss']))
    logging.info("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision']*100, results['recall']*100, results['f1']*100))

    with open(OUTPUTS_PATH, "w") as fout:
        with open(INPUTS_PATH, "w") as fin:
            for sent, pred in preds_list:
                fin.write(" ".join(sent)+"\n")
                fout.write(" ".join(pred)+"\n")

else:
    test_data = utils._get_srl_data(TEST_DATA_PATH, tokenizer, include_labels=False)
    # https://huggingface.co/transformers/main_classes/pipelines.html#transformers.TokenClassificationPipeline
    logging.info('Predicting labels for {:,} test sentences...'.format(len(test_data)))
    if not USE_CUDA: GPU_IX = -1
    nlp = pipeline('token-classification', model=model, tokenizer=tokenizer, device=GPU_IX)
    nlp.ignore_labels = []
    with open(OUTPUTS_PATH, "w") as fout:
        with open(INPUTS_PATH, "w") as fin:
            for seq_ix, seq in enumerate(test_data):
                sentence = " ".join(seq)
                predicted_labels = []
                output_obj = nlp(sentence)
                # [print(o) for o in output_obj]
                for tok in output_obj:
                    if '##' not in tok['word']:
                        predicted_labels.append(tok['entity'])
                logging.info(f"\n----- {seq_ix+1} -----\n{seq}\nPRED:{predicted_labels}")
                fin.write(sentence+"\n")
                fout.write(" ".join(predicted_labels)+"\n")