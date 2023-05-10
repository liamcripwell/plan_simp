import os
import re
import time
import shutil
from datetime import datetime

import fire
import nltk
import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from plan_simp.data.utils import OP_TOKENS
from plan_simp.models.bart import load_simplifier, run_generator
from plan_simp.models.classifier import load_planner, run_classifier


class Launcher(object):
    def inference(self, model_ckpt, test_file, out_file, x_col="complex", max_samples=None, op_col=None, reading_lvl=None,
                    device="cuda", batch_size=16, num_workers=32, beams=5, length_penalty=1.0, min_length=False,
                    context_dir=None, context_doc_id="pair_id", simple_context_dir=None, simple_context_doc_id="pair_id",):
        start = time.time()
        print(f"Starting time: {datetime.now()}")

        test_set = pd.read_csv(test_file)
        if max_samples is not None:
            test_set = test_set[:max_samples]

        print(f"Loaded test set of {len(test_set)} examples.")

        if min_length == "simple":
            min_length = list(test_set["s_len"])
        preds = run_generator(model_ckpt, test_set, x_col, op_col, reading_lvl, tokenizer=None, max_samples=max_samples, 
                                device=device, batch_size=batch_size, num_workers=num_workers, beams=beams, 
                                length_penalty=length_penalty, min_length=min_length,
                                context_dir=context_dir, context_doc_id=context_doc_id, simple_context_dir=simple_context_dir, 
                                simple_context_doc_id=simple_context_doc_id,)
        test_set["pred"] = preds
        
        test_set.to_csv(out_file, index=False)
        print(f"Predictions written to {out_file}.")

        end = time.time()
        elapsed = end - start
        print(f"Done! (Took {elapsed}s in total)")
        print(f"End time: {datetime.now()}")


    def dynamic(self, model_ckpt, test_file, out_file, clf_model_ckpt=None, doc_id_col="pair_id", context_dir=None, context_doc_id=None, temp_dir="temp_embeds", para_lvl=False,
                reading_lvl=None, op_col=None, beams=5, max_samples=None, device="cuda", result_cache=None, save_rate=10, simple_context_dir=None, simple_context_doc_id=None,):
        start = time.time()
        print(f"Starting time: {datetime.now()}")

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        elif result_cache is None:
            raise FileExistsError(f"Specified temp directory '{temp_dir}' already exists! If you want to continue from existing cache, use the `results_cache` arg.")

        # load data
        if result_cache is not None:
            test_set = pd.read_csv(result_cache, keep_default_na=False)
        else:
            test_set = pd.read_csv(test_file)

        # only enforce `max_samples` at the document level
        doc_ids = test_set[doc_id_col].unique()
        if max_samples is not None:
            doc_ids = doc_ids[:max_samples]
            test_set = test_set[test_set[doc_id_col].isin(doc_ids)]

        # load planning model
        if clf_model_ckpt is not None:
            clf_model, clf_tokenizer, clf_hparams = load_planner(clf_model_ckpt, add_context=True, device=device)

        # load simplification model
        model, tokenizer, hparams = load_simplifier(model_ckpt, device=device)

        # determine context window radius (NOTE: for now assumes each will not use varying window radii)
        if clf_model_ckpt is not None:
            z_radius = clf_hparams["context_window"]
        else:
            z_radius = hparams["context_window"]

        # SBERT model
        sent_encoder = SentenceTransformer('all-mpnet-base-v2')
        # https://github.com/huggingface/transformers/issues/5486 happens when sending sbert vectors to cpu, will break if using many workers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # determine whether to use pre-defined simple left-context
        l_z_dir = temp_dir
        if simple_context_dir is not None:
            l_z_dir = simple_context_dir
        if simple_context_doc_id is None:
            # NOTE: currently cannot specify simple document id column
            simple_context_doc_id = doc_id_col

        preds = ["#"]*len(test_set)
        pred_ls = [-1]*len(test_set)

        # preload existing results from cache
        if result_cache is not None:
            for i, row in test_set.iterrows():
                preds[i] = row.pred
                pred_ls[i] = row.pred_l

        remaining_docs = test_set[doc_id_col].unique()
        text_id = "sent_id" if not para_lvl else "para_id"
        max_text_id = test_set[text_id].max()
        pbar = tqdm(total=len(test_set))
        for i in range(max_text_id+1):
            # compile ith sents of each document (if not already simplified)
            i_texts = test_set[test_set[text_id] == i].copy()
            done = [j for j, _ in i_texts.iterrows() if preds[j] != "#"]
            i_texts = i_texts.drop(done)

            # skip if no valid texts found
            if len(i_texts) == 0:
                continue

            if clf_model_ckpt is not None:
                if para_lvl:
                    # split paragraph into sentences for classification if need be
                    i_sents = pd.DataFrame(columns=list(i_texts.columns)+["rel_sent_id"]) # have to define all cols before assigning rows
                    for _, para in i_texts.iterrows():
                        sents = para.complex.split(" <s> ")
                        simp_ids = eval(para.simp_sent_id)
                        doc_ps = eval(para.doc_pos)
                        doc_qs = eval(para.doc_quint)
                        doc_lens = eval(para.doc_len)
                        for k in range(len(sents)):
                            # flatten/update necessary columns
                            sent_row = para.copy()
                            sent_row["sent_id"] += k
                            sent_row["rel_sent_id"] = k # relative sent id within paragraph
                            sent_row["complex"] = sents[k]
                            sent_row["simp_sent_id"] = simp_ids[k]
                            sent_row["doc_pos"] = doc_ps[k]
                            sent_row["doc_quint"] = doc_qs[k]
                            sent_row["doc_len"] = doc_lens[k]
                            i_sents.loc[len(i_sents)] = sent_row
                else:
                    i_sents = i_texts

                if not para_lvl:
                    # run classifier with context_dir=normal and simple_context_dir=temp_dir (NOTE: doesn't support context-free planners)
                    clf_logits = run_classifier(clf_model, i_sents.reset_index(drop=True), "complex", tokenizer=clf_tokenizer, device=device, add_context=True,
                                                    context_dir=context_dir, context_doc_id=context_doc_id, simple_context_dir=l_z_dir, 
                                                    simple_context_doc_id=simple_context_doc_id, reading_lvl=reading_lvl, simple_sent_id=None,
                                                    return_logits=True, silent=True)
                    i_sents["pred_l"] = [int(y_.argmax()) for y_ in clf_logits]
                    i_texts = i_sents
                else:
                    # iteratively classify 1 sentences per paragraph
                    i_texts["pred_l"] = [[] for _ in range(len(i_texts))]
                    max_rel_sent_id = i_sents["rel_sent_id"].max()
                    for k in range(max_rel_sent_id+1):
                        i_sents_k = i_sents[i_sents["rel_sent_id"] == k]

                        # run planning
                        clf_logits = run_classifier(clf_model, i_sents_k.reset_index(drop=True), "complex", tokenizer=clf_tokenizer, device=device, add_context=True,
                                                    context_dir=context_dir, context_doc_id=context_doc_id, simple_context_dir=l_z_dir, 
                                                    simple_context_doc_id=simple_context_doc_id, reading_lvl=reading_lvl, simple_sent_id=None,
                                                    return_logits=True, silent=True)
                        i_sents_k["pred_l"] = [int(y_.argmax()) for y_ in clf_logits]

                        # update the `pred_l` list for each paragraph
                        for _, sent in i_sents_k.iterrows():
                            i_texts[i_texts[doc_id_col] == sent[doc_id_col]].iloc[0].pred_l.append(sent.pred_l)

                        # update context embeddings with complex sentence (a compromise for paragraphs)
                        update_context_embeddings(i_sents_k, temp_dir, doc_id_col, z_radius, sent_encoder, text_col="complex")
                
                # generate sentence with predicted operation ctrl token
                if hparams["add_context"]:
                    i_texts["pred"] = run_generator(model, i_texts.reset_index(drop=True), x_col="complex", op_col="pred_l", reading_lvl=reading_lvl,
                                                    tokenizer=tokenizer, min_length=False, device=device, beams=beams, silent=True,
                                                    context_dir=context_dir, context_doc_id=context_doc_id, simple_context_dir=l_z_dir, 
                                                    simple_context_doc_id=simple_context_doc_id, simple_sent_id=None)
                else:
                    i_texts["pred"] = run_generator(model, i_texts.reset_index(drop=True), x_col="complex", op_col="pred_l", reading_lvl=reading_lvl,
                                                    tokenizer=tokenizer, min_length=False, device=device, beams=beams, silent=True)
            else:
                i_texts["pred"] = run_generator(model, i_texts.reset_index(drop=True), x_col="complex", op_col=op_col, reading_lvl=reading_lvl,
                                                    tokenizer=tokenizer, min_length=False, device=device, beams=beams, silent=True,
                                                    context_dir=context_dir, context_doc_id=context_doc_id, simple_context_dir=l_z_dir, 
                                                    simple_context_doc_id=simple_context_doc_id, simple_sent_id=None)
                i_sents = i_texts # set `i_sents` to be same as texts (assumes operating at the sentence-level)

            # update predictions list
            i_texts["pred"] = i_texts.pred.fillna("") # need to do the na -> "" replacement
            i_texts, preds, pred_ls = cache_predictions(i_texts, preds, pred_ls)

            # after generating paragraphs, split them into sentences to be used for context embedding
            if para_lvl:
                i_sents = split_paras(i_texts)

            if simple_context_dir is None: # don't do if using pre-defined left-context embeddings
                if z_radius is not None:
                    # remove intra-paragraph complex sentence embeddings from context cache
                    if para_lvl:
                        clip_context_embeddings(i_texts, temp_dir, doc_id_col, clip_size_col="pred_l")

                    # update cached context embeds
                    update_context_embeddings(i_sents, temp_dir, doc_id_col, z_radius, sent_encoder)

            # update progress bar
            pbar.update(len(i_texts))

            # remove cached context embeds when no longer required
            # current_docs = i_sents[doc_id_col].unique()
            # for did in remaining_docs:
            #     if did not in current_docs and os.path.exists(f"{temp_dir}/{did}_z.pt"):
            #         os.remove(f"{temp_dir}/{did}_z.pt")
            # remaining_docs = current_docs

            # intermittent writing
            if i % save_rate == 0:
                test_set["pred"] = preds
                test_set["pred_l"] = pred_ls
                test_set.to_csv(out_file, index=False)
                print(f"Currently processed {i+1} sentences per document. Writing current predictions to {out_file}.")

                end = time.time()
                elapsed = end - start
                print(f"So far {elapsed}s have elapsed.")

        # delete all context embeddings in `temp_dir`
        shutil.rmtree(temp_dir)

        test_set["pred"] = preds
        test_set["pred_l"] = pred_ls
        test_set.to_csv(out_file, index=False)
        print(f"Predictions written to {out_file}.")

        end = time.time()
        elapsed = end - start
        print(f"Done! (Took {elapsed}s in total)")
        print(f"End time: {datetime.now()}")


def split_paras(i_texts):
    """
    Split paragraph into sentences for context encoding.
    """
    i_sents = pd.DataFrame(columns=i_texts.columns)
    for _, para in i_texts.iterrows():
        sents = para.pred.split("  ") # NOTE: assumes special tokens skipped during generation
        for k in range(len(sents)):
            # there can be issues if different number of sents than stipulated by plan
            if k >= len(para.pred_l):
                break
            # all columns except `sent_id`, `pred`, and `pred_id` will be the same as the paragraph
            sent_row = para.copy()
            sent_row["sent_id"] += k
            sent_row["pred"] = sents[k]
            sent_row["pred_l"] = para.pred_l[k]
            i_sents.loc[len(i_sents)] = sent_row
    i_sents["pred"] = i_sents.pred.fillna("")

    return i_sents


def cache_predictions(i_texts, preds, pred_ls):
    """
    Extract current step of predictions and updating existing caches.
    """
    _pred_ls = []
    for j, text in i_texts.iterrows():
        preds[j] = text.pred # update predicted sequence cache
        if "pred_l" not in i_texts.columns:
            # extract `pred_l` from sequence in case of multi-task system
            res = re.findall(r"(\<COPY\>|\<REPHRASE\>|\<SPLIT\>|\<DELETE\>)", text.pred)
            res = OP_TOKENS.index(res[0]) if len(res) else 0
            pred_ls[j] = res # update predicted labels cache
            _pred_ls.append(res)
        else:
            pred_ls[j] = text.pred_l
            _pred_ls.append(text.pred_l)

    i_texts["pred_l"] = _pred_ls # update df

    return i_texts, preds, pred_ls


def update_context_embeddings(i_sents, temp_dir, doc_id_col, z_radius, sent_encoder, text_col="pred"):
    """
    Add embeddings for new dynamic context sentences to cache. 
    This will also left-trim at point of radius (i.e. documents must be processed auto-regressively).
    """
    for _, sent in i_sents.iterrows():
        # load any existing embeds for previous sentences in document
        left_z = torch.tensor([])
        if os.path.exists(f"{temp_dir}/{sent[doc_id_col]}_z.pt"):
            left_z = torch.load(f"{temp_dir}/{sent[doc_id_col]}_z.pt") # load all of left_z (assumes ordered processing)
            if len(left_z) > z_radius: # TODO: could add arg to this func for window radius
                left_z = left_z[-z_radius:]
        
        # embed result and save to temp_dir
        if sent.pred_l != 3:
            pred_sents = nltk.sent_tokenize(sent[text_col]) # handle multi-sentence preds
            if pred_sents == []:
                continue
            z_i = sent_encoder.encode(pred_sents, convert_to_tensor=True).to("cpu")
            torch.save(torch.concat([left_z, z_i]), f"{temp_dir}/{sent[doc_id_col]}_z.pt")


def clip_context_embeddings(i_texts, temp_dir, doc_id_col, clip_size_col="pred_l"):
    """Remove right-end of dynamic left-context. The main purpose of this is to account for intra-paragraph static context."""
    for _, para in i_texts.iterrows():
        clip_size = len(para[clip_size_col])

        if os.path.exists(f"{temp_dir}/{para[doc_id_col]}_z.pt"):
            left_z = torch.load(f"{temp_dir}/{para[doc_id_col]}_z.pt") # load all of left_z (assumes ordered processing)
            left_z = left_z[:-clip_size]
            torch.save(left_z, f"{temp_dir}/{para[doc_id_col]}_z.pt")


if __name__ == '__main__':
    fire.Fire(Launcher)
