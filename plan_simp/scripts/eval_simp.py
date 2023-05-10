from glob import glob
import re
from unittest.mock import DEFAULT

import fire
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

from easse.bleu import sentence_bleu
from easse.fkgl import corpus_fkgl
from easse.sari import get_corpus_sari_operation_scores
from transformers import BartTokenizer

from plan_simp.eval.dsari import dsari_doc
from plan_simp.eval.bartscore import BARTScorer
from plan_simp.eval.bertscore import calculate_bertscore
from plan_simp.eval.questeval import calculate_questeval
from plan_simp.eval.easse_sari import get_corpus_sari_operation_scores
from plan_simp.eval.smart_eval import matching_functions, scorer

DEFAULT_METRICS = ["bart", "dsari", "sari", "fkgl", "smart", "bleu"]


def read_file(filename):
    return [d.strip() for d in open(filename).readlines()]

def clean_sequences(input_seqs, output_seqs, ref_seqs=[]):
    """
    `input_seqs`: List[str]
    `output_seqs`: List[str]
    `ref_seqs`: List[List[str]]
    """
    # filter out special content
    output_seqs = [re.sub(r"\[PLAN\].*\[SIMPLIFICATION\]", "", t) for t in output_seqs] # prefix plan
    # plan separators NOTE: only works for 4 class plans
    output_seqs = [re.sub(r"(\<COPY\>|\<REPHRASE\>|\<SPLIT\>|\<DELETE\>)", "", t) for t in output_seqs]
    # sep tokens
    input_seqs = [re.sub(r"\<\\?/?s\>", "", t) for t in input_seqs]
    input_seqs = [re.sub(r"\<\SEP\>", "", t) for t in input_seqs]
    ref_seqs = [[re.sub(r"\<\\?/?s\>", "", t[0])] for t in ref_seqs]
    ref_seqs = [[re.sub(r"\<SEP\>", "", t[0])] for t in ref_seqs]
    output_seqs = [re.sub(r"\<\\?/?s\>", "", t) for t in output_seqs]
    # padding
    output_seqs = [re.sub(r"\<pad\>", "", t) for t in output_seqs]
    # excess spaces
    input_seqs = [re.sub(r" +", " ", t) for t in input_seqs]
    output_seqs = [re.sub(r" +", " ", t) for t in output_seqs]
    ref_seqs = [[re.sub(r" +", " ", t[0])] for t in ref_seqs]

    return input_seqs, output_seqs, ref_seqs

def calculate_saris(in_doc, out_doc, ref_docs, easse=True):
    if easse:
        # EASSE base-implementation
        dsari_a, dsari_k, dsari_d = get_corpus_sari_operation_scores([in_doc], [out_doc], [ref_docs], doc_level=True)
        dsari = (dsari_a + dsari_k + dsari_d) / 3
        dsaris = [dsari, dsari_a, dsari_k, dsari_d] # [mean, add, keep, del]

        sari = get_corpus_sari_operation_scores([in_doc], [out_doc], [[x] for x in ref_docs])
        saris = [np.mean(sari), *sari] # [mean, add, keep, del]

        gdsari = get_corpus_sari_operation_scores([in_doc], [out_doc], [ref_docs], doc_level=True, global_penalties=True)
        gdsaris = sum(gdsari) / 3
    else:
        dsari, dsari_k, dsari_d, dsari_a = dsari_doc(in_doc, out_doc, ref_docs, doc_level=True)
        dsaris = [dsari, dsari_a, dsari_k, dsari_d] # [mean, add, keep, del]

        sari, sari_k, sari_d, sari_a = dsari_doc(in_doc, out_doc, ref_docs, doc_level=False)
        saris = [sari, sari_a, sari_k, sari_d] # [mean, add, keep, del]

        gdsari = dsari_doc(in_doc, out_doc, ref_docs, doc_level=True, global_penalties=True)
        gdsaris = gdsari[0]
    
    return dsaris, saris, gdsaris

def output_results(input_data, metrics=DEFAULT_METRICS):
    if "bart" in metrics:
        print("BARTScore:")
        print(input_data["bart_faith"].mean())
        print(input_data["bart_p"].mean())
        print(input_data["bart_r"].mean())
        print(input_data["bart_f1"].mean())
    if "fkgl" in metrics:
        print("FKGL:")
        print(input_data["fkgl"].mean())
    if "smart" in metrics:
        print("SMART:")
        print(input_data["smart_p"].mean())
        print(input_data["smart_r"].mean())
        print(input_data["smart_f1"].mean())
    if "dsari" in metrics:
        print("D-SARI:")
        print(input_data["dsari"].mean())
        for o in "akd":
            print(f'\t{o}: {input_data["dsari" + "_" + o].mean()}')
    if "gdsari" in metrics:
        print("D-SARI (Global):")
        print(input_data["gdsari"].mean())
    if "sari" in metrics:
        print("SARI:")
        print(input_data["sari"].mean())
        for o in "akd":
            print(f'\t{o}: {input_data["sari" + "_" + o].mean()}')
    if "bleu" in metrics:
        print("BLEU:")
        print(input_data["bleu"].mean())
    if "bert" in metrics:
        print("P_BERT:")
        print(input_data["bertscore"].mean())
    if "questeval" in metrics:
        print(f"QuestEval: {input_data['questeval'].mean()}")
        print(f"QuestEval (refless): {input_data['questeval_noref'].mean()}")

    print("Avg. Len:")
    print(f'\tTokens: {input_data["pred_len"].mean()}\n\tSentences: {input_data["pred_num_sents"].mean()}')
            

def evaluate(input_data, output_data=None, ref_data=None, prepro=False, x_col=None, y_col=None, r_col=None, 
                doc_id_col=None, sent_level=False, questeval_batch_size=512, questeval_log_dir="logs", questeval_ref=True,
                bartscore_path=None, metrics=DEFAULT_METRICS):
    # read data into correct types
    if isinstance(input_data, str):
        if input_data.endswith(".csv"):
            input_data = pd.read_csv(input_data)
            input_seqs = list(input_data[x_col])
            ref_seqs = [[d] for d in input_data[r_col]]
        else:
            input_seqs = read_file(input_data)
            if isinstance(output_data, str):
                output_seqs = read_file(output_data)
                # assume single ref if type is str
                ref_seqs = [[d] for d in read_file(ref_data)]
    else:
        # DataFrame
        input_seqs = list(input_data[x_col])
        ref_seqs = [[d] for d in input_data[r_col]]

    # handle DataFrame input
    if output_data is None and y_col is not None:
        input_data[y_col] = input_data[y_col].fillna("")
        output_seqs = list(input_data[y_col])

    # handle sentence-level outputs
    if sent_level and output_data is not None:
        doc_outputs = []
        if isinstance(output_data, str):
            output_data = pd.read_csv(output_data)
        output_data[y_col] = output_data[y_col].fillna("") # clean empty preds (deletes)
        for i, row in input_data.iterrows():
            d_id = row[doc_id_col]
            # NOTE: this should also work for paragraphs where the the id of the first sentence is in the `sent_id` column
            sents = list(output_data[output_data[doc_id_col] == d_id].sort_values(by=["sent_id"])[y_col])

            # clean output before saving
            out_str = re.sub(" +", " ", " ".join(sents)).strip() # join sents and remove extra whitespace
            doc_outputs.append(out_str)

        assert len(input_seqs) == len(doc_outputs)
        output_seqs = doc_outputs

    # clean sequences of special tokens, whitespace, etc.
    input_seqs, output_seqs, ref_seqs = clean_sequences(input_seqs, output_seqs, ref_seqs)

    if prepro:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=False)

    n = len(input_seqs)
    lens = np.zeros(n)
    nsents = np.zeros(n)
    results = {
        "bleu": np.zeros(n),
        "bert": np.zeros(n),
        "sari": np.zeros((n, 4)),
        "dsari": np.zeros((n, 4)),
        "gdsari": np.zeros(n),
        "fkgl": np.zeros(n),
        "smart": np.zeros((n, 3))
    }

    out_docs = []
    ref_docss = []
    for i in tqdm(range(n)):
        in_doc = input_seqs[i]
        out_doc = output_seqs[i]
        ref_docs = ref_seqs[i]

        if prepro:
            in_doc = tokenizer.decode(tokenizer(in_doc)["input_ids"], skip_special_tokens=True)
            out_doc = tokenizer.decode(tokenizer(out_doc)["input_ids"], skip_special_tokens=True)
            for j in range(len(ref_docs)):
                ref_docs[j] = tokenizer.decode(tokenizer(ref_docs[j])["input_ids"], skip_special_tokens=True)

        if any(["sari" in m for m in metrics]):
            sari_results = calculate_saris(in_doc, out_doc, ref_docs)
            if "dsari" in metrics:
                results["dsari"][i] = sari_results[0]
            if "sari" in metrics:
                results["sari"][i] = sari_results[1]
            if "gdsari" in metrics:
                results["gdsari"][i] = sari_results[2]

        if "bleu" in metrics:
            results["bleu"][i] = sentence_bleu(out_doc, ref_docs)

        # sentence tokenized documents for FKGL and SMART
        out_doc_sents = nltk.sent_tokenize(out_doc)
        ref_docs_sents = [nltk.sent_tokenize(ref_doc) for ref_doc in ref_docs]

        if "fkgl" in metrics:
            results["fkgl"][i] = corpus_fkgl(out_doc_sents)

        if "smart" in metrics:
            matcher = matching_functions.chrf_matcher
            smart_scorer = scorer.SmartScorer(matching_fn=matcher)

            # if entire document deleted manually include 1 empty sentence
            if out_doc_sents == []:
                out_doc_sents = [""]

            # need documents split into sentences (NOTE: assumes single reference)
            smarts = smart_scorer.smart_score(ref_docs_sents[0], out_doc_sents)["smartL"]
            results["smart"][i] = np.array([smarts["precision"], smarts["recall"], smarts["fmeasure"]])

        lens[i] = len(tokenizer(out_doc)["input_ids"])
        nsents[i] = len(out_doc_sents)

        out_docs.append(out_doc)
        ref_docss.append(ref_docs)

    if "bart" in metrics and bartscore_path is not None:
        print("Calculating BARTScores...")
        bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
        bart_scorer.load(path=bartscore_path)

        results["bart_faith"] = np.array(bart_scorer.score(input_seqs, out_docs))
        results["bart_p"] = np.array(bart_scorer.score([r[0] for r in ref_docss], out_docs))
        results["bart_r"] = np.array(bart_scorer.score(out_docs, [r[0] for r in ref_docss]))
        results["bart_f1"] = np.array([results["bart_p"], results["bart_r"]]).mean(axis=0)

        input_data["bart_faith"] = results["bart_faith"]
        input_data["bart_p"] = results["bart_p"]
        input_data["bart_r"] = results["bart_r"]
        input_data["bart_f1"] = results["bart_f1"]
    elif "bart" in metrics:
        # need to remove bart from metrics list if we don't have a path for the model
        metrics.remove("bart")

    if "bert" in metrics:
        print("Calculating BERTScores...")
        results["bert"] = np.array(calculate_bertscore(out_docs, [r[0] for r in ref_docss])) # assumes single reference

    # require explicit permission to run as it can be very slow with document-length inputs
    if "questeval" in metrics:
        print("Calculating QuestEvals...")
        questevals = np.array(calculate_questeval(input_seqs, output_seqs, ref_seqs, 
                                batch_size=questeval_batch_size, use_ref=questeval_ref, log_dir=questeval_log_dir))
        
        print("QuestEval:")
        print(questevals.mean())
        out_key = "questeval"
        if not questeval_ref:
            out_key += "_noref"
        input_data[out_key] = questevals

    input_data["bertscore"] = results["bert"]
    input_data["dsari"] = results["dsari"][:,0]
    input_data["dsari_a"] = results["dsari"][:,1]
    input_data["dsari_k"] = results["dsari"][:,2]
    input_data["dsari_d"] = results["dsari"][:,3]
    input_data["sari"] = results["sari"][:,0]
    input_data["sari_a"] = results["sari"][:,1]
    input_data["sari_k"] = results["sari"][:,2]
    input_data["sari_d"] = results["sari"][:,3]
    input_data["bleu"] = results["bleu"]
    input_data["fkgl"] = results["fkgl"]
    input_data["pred_len"] = lens
    input_data["pred_num_sents"] = nsents
    input_data["gdsari"] = results["gdsari"]
    if "smart" in metrics:
        input_data["smart_p"] = results["smart"][:,0]
        input_data["smart_r"] = results["smart"][:,1]
        input_data["smart_f1"] = results["smart"][:,2]

    input_data["pred"] = out_docs # NOTE: comment out if you want to keep unfiltered predictions

    output_results(input_data, metrics=metrics)

    return input_data


if __name__ == '__main__':
    fire.Fire(evaluate)
