from collections import defaultdict

# CoNLL-U Format - https://universaldependencies.org/format.html
# Universal POS - https://universaldependencies.org/u/pos/index.html
POS_EN_PTB_TO_UNIVERSAL = {
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
    "RP": "ADP", # "IN": "ADP", # "TO": "ADP",
    "RB": "ADV", "RBR": "ADV", "RBS": "ADV",
    "MD": "AUX", # verb uses of be, have, do, and get should be AUX instead of VERB
    "CC": "CCONJ",
    "DT": "DET", "PDT": "DET", "WDT": "DET",
    "UH": "INTJ",
    "NN": "NOUN", "NNS": "NOUN",
    "CD": "NUM",
    "POS": "PART", "TO": "PART",
    "PRP": "PRON", "PRP$": "PRON", "WP": "PRON", "WP$": "PRON", "EX": "PRON",
    "NNP": "PROPN", "NNPS": "PROPN",
    "``": "PUNCT", "\"": "PUNCT", "-LRB-": "PUNCT", "-RRB-": "PUNCT", ",": "PUNCT", ".": "PUNCT", ":": "PUNCT", "HYPH": "PUNCT", "''": "PUNCT", "(": "PUNCT", ")": "PUNCT",
    "IN": "SCONJ",
    "NFP": "SYM", "#": "SYM", "$": "SYM", "SYM": "SYM", "%": "SYM",
    "VB": "VERB", "VBP": "", "VBZ": "", "VBD": "", "VBG": "", "VBN": "",
    "FW": "X", "LS": "X", "XX": "X", "ADD": "X", "AFX": "X", "GW": "X"
}


def _convert_to_universal(postag, lemma, lang="EN"):
    if lang == "EN":
        pos_dict = POS_EN_PTB_TO_UNIVERSAL
    else:
        pos_dict = POS_EN_PTB_TO_UNIVERSAL
    if postag.startswith("V"):
        if lemma.lower() in ["be", "have", "do", "get"]:
            return "AUX"
        else:
            return "VERB"
    else:
        return pos_dict.get(postag, postag)


class ZAPToken():
    def __init__(self, raw_line, word_ix):
        info = raw_line.split()
        self.info = info
        self.id = int(info[0])
        self.position = word_ix  # 0-based position in sentence
        self.word = info[1]
        self.lemma = info[2]
        self.pos_tag = info[4]
        self.head = info[6]
        self.dep_tag = info[7]
        self.is_pred = True if info[10].upper() == "Y" else False
        if len(info) > 11:
            self.pred_sense = info[11]
            self.pred_sense_id = str(self.position) + "##" + self.pred_sense
        else:
            self.pred_sense = None
            self.pred_sense_id = ""
        if len(info) > 12:
            self.labels = info[12:]
        else:
            self.labels = []

    def get_info(self):
        return [str(self.id), self.word, self.lemma, self.pos_tag, str(self.head), self.dep_tag,
                self.is_pred, self.pred_sense] + self.labels

    def get_conllU_line(self, separator="\t"):
        info = self.get_info()
        return separator.join(info)


class CoNLLUP_Token():
    def __init__(self, raw_line, word_ix):
        info = raw_line.split()
        # # ['1', 'Kleszczele', 'Kleszczele', 'PROPN', 'NE', 'Case=Nom|Number=Sing', '12', 'nsubj', '_', '_', 'A1']
        self.info = info
        self.id = info[0] #int(info[0]) # 1-based ID as in the CoNLL file
        self.position = word_ix # 0-based position in sentence
        self.word = info[1]
        self.lemma = info[2]
        self.pos_universal = info[3]
        self.pos_tag = info[4]
        self.detail_tag = info[5]
        self.head = info[6]
        self.dep_tag = info[7]
        self.is_pred = True if info[8] == "Y" else False
        if self.is_pred:
            self.pred_sense = info[9]
            self.pred_sense_id = str(self.position) + "##" + self.pred_sense
        else:
            self.pred_sense = None
            self.pred_sense_id = ""
        if len(info) > 10:
            self.labels = info[10:]
        else:
            self.labels = []

    def get_info(self):
        is_pred_str = "Y" if self.is_pred else "_"
        pred_sense_str = self.pred_sense if self.pred_sense else "_"
        return [str(self.id), self.word, self.lemma, self.pos_universal, self.pos_tag, self.detail_tag,
                str(self.head), self.dep_tag, is_pred_str, pred_sense_str] + self.labels

    def get_conllU_line(self, separator="\t"):
        info = self.get_info()
        return separator.join(info)

    def get_conll09_line(self, delim="\t"):
        is_pred_str = "Y" if self.is_pred else "_"
        sense_str = self.pred_sense if self.is_pred else "_"
        info = [self.id, self.word, self.lemma, self.lemma, self.pos_tag, self.pos_tag, "_", self.detail_tag,
                self.head, self.head, self.dep_tag, self.dep_tag, is_pred_str, sense_str] + self.labels
        return delim.join(info)


class CoNLL09_Token():
    def __init__(self, raw_line, word_ix):
        info = raw_line.split()
        # print(info)
        # # ['1', 'Frau', 'Frau', 'Frau', 'NN', 'NN', '_', 'nom|sg|fem', '5', '5', 'CJ', 'CJ', '_', '_', 'AM-DIS', '_']
        self.info = info
        self.id = int(info[0]) # 1-based ID as in the CoNLL file
        self.position = word_ix # 0-based position in sentence
        self.word = info[1]
        self.lemma = info[2]
        self.pos_tag = info[4]
        self.pos_universal = _convert_to_universal(self.pos_tag, self.lemma)
        self.head = info[8]
        self.dep_tag = info[10]
        self.detail_tag = "_"
        self.is_pred = True if info[12] == "Y" else False
        if self.is_pred:
            self.pred_sense = info[13].strip("[]")
            self.pred_sense_id = str(self.position) + "##" + self.pred_sense
        else:
            self.pred_sense = None
            self.pred_sense_id = ""
        if len(info) > 14:
            self.labels = info[14:]
        else:
            self.labels = []

    def get_conllU_line(self):
        # We want: 25 panic panic NN NN 22 22 OBJ OBJ _ Y panic.01 _ _ _ _ _ _ A1 _ _
        universal_POS = [self.pos_universal]
        conllUinfo = self.info[:3] + universal_POS + self.info[5:6] + self.info[8:12] + ["_"] + self.info[12:]
        return " ".join(conllUinfo)

    def get_conll09_line(self, delim="\t"):
        # We want:
        # 1 Frau Frau Frau NN NN _ nom|sg|fem 5 5 CJ CJ _ _ AM-DIS _
        # 10	fall	fall	fall	VB	VB	_	_	8	8	VC	VC	Y	fall.01	_	_	_	_	_
        is_pred_str = "Y" if self.is_pred else "_"
        sense_str = self.pred_sense if self.is_pred else "_"
        info = [self.id, self.word, self.lemma, self.lemma, self.pos_tag, self.pos_tag, "_", self.detail_tag,
                self.head, self.head, self.dep_tag, self.dep_tag, is_pred_str, sense_str] + self.labels
        return delim.join(info)


class ArgumentHead(): # Used in CoNLL-09 Annotations
    def __init__(self, position, tag, word, predicate_id, parent_pred):
        self.position = position
        self.tag = tag
        self.head_word = word
        self.belongs_to_pred = predicate_id # Id in the sentence
        self.parent_pred = parent_pred # Token

    def get(self):
        return self.position, self.parent_pred, self.tag, self.head_word

    def show(self):
        return "[{} : {}]".format(self.tag, self.head_word)


class AnnotatedSentence():
    def __init__(self):
        self.tokens = []
        self.only_senses = []
        self.predicates = []
        self.argument_structure = {}
        self.BIO_sequences = {}
        self.predicates_global_seq = []
        self.predicates_sequences = {}
        self.predicate_indices = []
        self.nominal_predicates = []

    def _make_head_spans(self, pred_arg, include_nominals):
        global_pred_seq = ["O"] * len(self.tokens)
        pred2ix = []
        for tok in self.tokens:
            pred_seq = ["O"] * len(self.tokens)
            if tok.is_pred:
                pred_seq[tok.position] = "V"
                global_pred_seq[tok.position] = "V"
                self.predicate_indices.append(tok.position)
                self.predicates_sequences[tok.pred_sense_id] = pred_seq
                pred2ix.append((tok.pred_sense_id, tok.pred_sense, tok.position))

        # Create Argument-Head BIO sequences
        for (tok_pos, word, pred_sense, tok_tag), args in pred_arg.items():
            BIO_seq = ["O"] * len(self.tokens)
            if "V" in tok_tag:
                BIO_seq[tok_pos] = "B-V"
                for arg_head in args:
                    BIO_seq[arg_head.position] = "B-" + arg_head.tag
                self.BIO_sequences[(tok_pos, pred_sense)] = BIO_seq
            elif include_nominals:
                BIO_seq[tok_pos] = "B-N-V"
                for arg_head in args:
                    if arg_head.position != tok_pos:
                        BIO_seq[arg_head.position] = "B-" + arg_head.tag
                    else:
                        BIO_seq[arg_head.position] = "B-" + arg_head.tag + "-N-V"
                self.BIO_sequences[(tok_pos, pred_sense)] = BIO_seq
        self.predicates_global_seq = global_pred_seq # Just one seq indicating for all predicates inside the sentence
        self.argument_structure = pred_arg

    def annotate_pred_arg_struct(self, include_nominals):
        self.predicates_global_seq = ["O"]*len(self.tokens)
        my_preds = self.predicates
        pred_arg = {my_preds[pred_ix]: [] for pred_ix in range(len(my_preds))}
        if len(my_preds) == 0: return None
        if isinstance(self.tokens[0], CoNLL09_Token) or isinstance(self.tokens[0], CoNLLUP_Token) or isinstance(self.tokens[0], ZAPToken):
            for tok in self.tokens:
                for pred_ix, lbl in enumerate(tok.labels):
                    if lbl != "_": pred_arg[my_preds[pred_ix]].append(ArgumentHead(tok.position, lbl, tok.word, pred_ix, self.predicates[pred_ix]))
            self._make_head_spans(pred_arg, include_nominals)
        else:
            raise Exception("The Pred-Arg Structure Annotation is not Defined for this Kind of Token!")

    def get_tokens(self):
        return [tok.word for tok in self.tokens]

    def get_sentence(self):
        return " ".join([tok.word for tok in self.tokens])

    def get_pred_args(self):
        s = []
        for predicate, arguments in self.argument_structure.items():
            s.append("{} --> {}".format(predicate, [arg.show() for arg in arguments]))
        return s


def get_annotation(raw_lines, token_class, include_nominals):
    ann = AnnotatedSentence()
    # Annotate the predicates and senses
    real_index = 0
    for i, line in enumerate(raw_lines):
        if len(line.split()) > 8:
            tok = token_class(line, real_index)
            if tok:
                ann.tokens.append(tok)
                if tok.is_pred:
                    ann.predicates.append((tok.position, tok.word, tok.pred_sense, tok.pos_tag))
                    ann.only_senses.append(tok.pred_sense)
            real_index += 1
    # Annotate the arguments of the corresponding predicates
    ann.annotate_pred_arg_struct(include_nominals=include_nominals)
    return ann


def read_conll(filename, conll_token, include_nominals):
    f = open(filename)
    n_sents = 0
    annotated_sentences, buffer_lst = [], []
    for i, line in enumerate(f.readlines()):
        if line[0] == "#": continue
        if len(line) > 5 and len(line.split()) > 0:
            buffer_lst.append(line)
        else:
            ann = get_annotation(buffer_lst, conll_token, include_nominals)
            n_sents += 1
            buffer_lst = []
            annotated_sentences.append(ann)

    if len(buffer_lst) > 0:
        annotated_sentences.append(get_annotation(buffer_lst, conll_token, include_nominals))
    print("Read {} Sentences!".format(n_sents))
    return annotated_sentences