from CoNLL_Annotations import *
import json, io, argparse


def get_token_type(type_str):
    if type_str == "ZAPToken":
        return ZAPToken
    elif type_str == "CoNLLUP_Token":
        return CoNLLUP_Token
    else:
        return CoNLL09_Token


def make_mono_files(file_props, include_nominals, append_in_file=False):
    print("---------------------\nProcessing {} file...".format(file_props["in"]))
    no_preds, no_verb, written_in_file = 0, 0, 0
    sentences = read_conll(file_props["in"], conll_token=file_props["token_type"], include_nominals=include_nominals)
    file_mode = "a" if append_in_file else "w"
    json_file = io.open(file_props["out"], file_mode, encoding='utf8')
    for sent in sentences:
        seq_obj = {}
        my_sent = sent.get_tokens()
        # print(sent.get_sentence() + "\n")
        # sent.show_pred_args()
        per_predicate = list(sorted(sent.BIO_sequences.items(), key=lambda x: x[0][0]))
        if len(per_predicate) > 0:
            for ix, (pred_sense, seq) in enumerate(per_predicate):
                seq_obj["seq_words"] = my_sent
                seq_obj["BIO"] = seq
                seq_obj["pred_sense"] = sent.predicates[ix]
                if "B-V" not in seq: no_verb += 1
                seq_obj["src_lang"] = file_props["lang"]
                seq_obj["tgt_lang"] = "<" + file_props["lang"][1:-1] + "-SRL>"
                json_file.write(json.dumps(seq_obj) + "\n")
                written_in_file += 1
        else:
            no_preds += 1
            generic_bio = ["O" for x in my_sent]
            seq_obj["seq_words"] = my_sent
            seq_obj["BIO"] = generic_bio
            seq_obj["pred_sense"] = (-1, "-", "<NO-PRED>", "-")
            seq_obj["src_lang"] = file_props["lang"]
            seq_obj["tgt_lang"] = "<" + file_props["lang"][1:-1] + "-SRL>"
            json_file.write(json.dumps(seq_obj) + "\n")
            written_in_file += 1

    print("IN: {} --> OUT: {}\nFound {} sentences in CoNLL --> Wrote {} sentences in JSON".format(file_props["in"],
                                                                                                  file_props["out"],
                                                                                                  len(sentences),
                                                                                                  written_in_file))


if __name__ == "__main__":
    """
    RUN EXAMPLE:
    
        python pre_processing/conll2json.py \
            --source_file data/mini_X-SRL_Test_EN.conll \
            --output_file data/Test_EN.jsonl \
            --src_lang "<EN>" \
            --token_type ZAPToken
        
        python pre_processing/conll2json.py \
            --source_file data/CoNLL2009-ST-English-trial.txt \
            --output_file data/Trial_EN.jsonl \
            --src_lang "<EN>" \
            --token_type CoNLL09_Token
            
                
    """

    # Read arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_file', help='Path to the source text sentences or to the raw source dataset CoNLL file', required=True)
    parser.add_argument('-t', '--target_file', help='Path to the raw target dataset CoNLL file', default=None)
    parser.add_argument('-o', '--output_file', help='Path and filename where the JSON data will be saved', required=True)
    parser.add_argument('-l', '--src_lang',    help="Language of the unlabeled sentences. Options: '<EN>','<DE>','<FR>'", required=True)
    parser.add_argument('-d', '--dataset_type', help="String indicating if it is a 'mono' or 'cross' mode", default="mono")
    parser.add_argument('-tt', '--token_type',   help='String that Indicates the format of the CoNLL file. Options: CoNLL09 | CoNLLUP_Token', default="CoNLL09")
    args = parser.parse_args()

    token_type = get_token_type(args.token_type)

    props = {"in":  args.source_file,
             "out":  args.output_file,
             "lang": args.src_lang,
             "token_type": token_type
             }

    # Convert to JSON Objects repeating sentences to have 1-Predicate-Argument Structure per sentence
    make_mono_files(props, include_nominals=False)

print("DONE!")