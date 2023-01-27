import re
import time
from datetime import datetime

import fire
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from plan_simp.models.bart import run_generator
from plan_simp.models.classifier import run_classifier
from plan_simp.data.utils import CLASS_LABELS, M_CLASS_LABELS, OP_TOKENS, M_OP_TOKENS, convert_labels

BIN_CLASS = ["delete", "keep"]


def evaluate(model_loc, test_file, x_col="complex", y_col="label", out_file=None, add_context=False, context_dir=None,
                context_doc_id="pair_id", simple_context_dir=None, simple_context_doc_id="pair_id", num_labels=4,
                reading_lvl=None, src_lvl=None, by_level=False, multi_split=False, max_samples=None, mtl=False,
                device="cuda", num_workers=8, pred_col=None, doc_level=False):
    start = time.time()
    print(f"Starting time: {datetime.now()}")

    print("Loading data...")
    test_set = pd.read_csv(test_file)
    if max_samples is not None:
        test_set = test_set[:max_samples]

    class_labels = M_CLASS_LABELS if multi_split else CLASS_LABELS
    op_tokens = M_OP_TOKENS if multi_split else OP_TOKENS # only needed for MTL models

    print("Running planner predictions...")
    if not mtl:
        test_set["pred_l"] = run_classifier(model_loc, test_set, x_col=x_col, max_samples=max_samples, device=device, 
                                            num_workers=num_workers, add_context=add_context, context_dir=context_dir,
                                            context_doc_id=context_doc_id, simple_context_dir=simple_context_dir,
                                            simple_context_doc_id=simple_context_doc_id, reading_lvl=reading_lvl, 
                                            src_lvl=src_lvl, return_logits=False)
    else:
        if pred_col is None:
            # run generative model
            ys = run_generator(model_loc, test_set, x_col=x_col, max_samples=max_samples, device=device, num_workers=num_workers,
                                context_dir=context_dir, context_doc_id=context_doc_id, simple_context_dir=simple_context_dir, 
                                simple_context_doc_id=simple_context_doc_id, reading_lvl=reading_lvl,)

            test_set["pred"] = ys
            test_set.to_csv(out_file, index=False)
            print(f"Intermediate generations written to {out_file}.")
        else:
            # load pre-generated simplifications
            ys = test_set[pred_col]

        # convert extracted tags to actual label predictions
        pred_ls = [re.findall('(\<COPY\>|\<REPHRASE\>|\<SPLIT\>|\<DELETE\>)', y) for y in ys]
        pred_ls = [[op_tokens.index(y_) for y_ in y] for y in pred_ls]
        if doc_level:
            # remove extra operation predictions
            if "num_c_sents" in test_set.columns:
                c_lens = [row.num_c_sents for _, row in test_set.iterrows()]
            else:
                c_lens = [len(eval(row.complex)) for _, row in test_set.iterrows()]
            pred_ls = [pred_ls[i][:c_lens[i]] for i in range(len(pred_ls))]

            # append deletion placeholders if predicted shorter sequence
            for i in range(len(pred_ls)):
                if len(pred_ls[i]) < c_lens[i]:
                    pred_ls[i] += [num_labels-1]*(c_lens[i]-len(pred_ls[i]))
        else:
            pred_ls = [y[0] if len(y) > 0 else 0 for y in pred_ls] # reduce to single operation prediction
        test_set["pred_l"] = pred_ls

    if y_col in test_set.columns:
        # get ground truths
        if doc_level:
            # handle case of document-level inputs (multiple operation predictions per input)
            gts = [convert_labels(eval(ls_i), class_labels) for ls_i in test_set[y_col]] # NOTE: assumes correct length every time
            gts = [l for x in gts for l in x] # flatten labels
            preds = [l for x in test_set.pred_l for l in x] # flatten preds
        else:
            gts = convert_labels(list(test_set[y_col]), class_labels)
            preds = test_set.pred_l

        # binary task
        if num_labels == 2:
            gts = [int(y < 4) for y in gts]

        # calculate accuracy scores
        class_scores = precision_recall_fscore_support(gts, preds, average=None)
        micro_scores = precision_recall_fscore_support(gts, preds, average="micro")
        macro_scores = precision_recall_fscore_support(gts, preds, average="macro")

        for cl, i in class_labels.items():
            if i > num_labels - 1: break
            cl_name = cl if num_labels > 2 else BIN_CLASS[i]
            print(f"{cl_name}:\n\tP: {class_scores[0][i]}\n\tR: {class_scores[1][i]}\n\tF1: {class_scores[2][i]}")
        print(f"Micro:\n\tP: {micro_scores[0]}\n\tR: {micro_scores[1]}\n\tF1: {micro_scores[2]}")
        print(f"Macro:\n\tP: {macro_scores[0]}\n\tR: {macro_scores[1]}\n\tF1: {macro_scores[2]}")

    if by_level:
        print("Evaluation performance on each reading level pair...")
        test_set["level_pair"] = [f"{row.c_level}->{row.s_level}" for _, row in test_set.iterrows()]
        print(test_set["level_pair"].value_counts())

        pairs = sorted(test_set.level_pair.unique())
        test_set["gt"] = gts

        for pair in pairs:
            subset = test_set[test_set.level_pair == pair]
            micro_scores = precision_recall_fscore_support(subset["gt"], subset.pred_l, average="micro", 
                                                            labels=subset["gt"].unique())
            macro_scores = precision_recall_fscore_support(subset["gt"], subset.pred_l, average="macro", 
                                                            labels=subset["gt"].unique())

            print(f"{pair}:")
            print(f"\tMicro:\n\tP: {micro_scores[0]}\n\tR: {micro_scores[1]}\n\tF1: {micro_scores[2]}")
            print(f"\tMacro:\n\tP: {macro_scores[0]}\n\tR: {macro_scores[1]}\n\tF1: {macro_scores[2]}")

    # write inpute `DataFrame` with new predictions column to file
    if out_file is not None:
        test_set.to_csv(out_file, index=False)
        print(f"Predictions written to {out_file}.")

    end = time.time()
    elapsed = end - start
    print(f"Done! (Took {elapsed}s in total)")
    print(f"End time: {datetime.now()}")


if __name__ == '__main__':
    fire.Fire(evaluate)
