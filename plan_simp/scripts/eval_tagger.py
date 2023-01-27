import time
from datetime import datetime

import fire
import torch
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from plan_simp.data.utils import CLASS_LABELS, M_CLASS_LABELS, convert_labels
from plan_simp.models.tagger import run_tagger


def evaluate(model_loc, test_file, out_file=None, x_col="pair_id", embed_dir=None,
                reading_lvl=None, multi_split=False, max_samples=None, bart=False, 
                device="cuda", num_workers=8):
    start = time.time()
    print(f"Starting time: {datetime.now()}")

    print("Loading data...")
    test_set = pd.read_csv(test_file)
    if max_samples is not None:
        test_set = test_set[:max_samples]

    print("Running predictions...")
    preds = run_tagger(model_loc, test_set, x_col, max_samples=max_samples, device=device, 
                                        reading_lvl=reading_lvl, num_workers=num_workers, 
                                        embed_dir=embed_dir, bart=bart, return_logits=False)

    class_labels = M_CLASS_LABELS if multi_split else CLASS_LABELS

    # remove padding token predictions
    if "num_c_sents" in test_set.columns:
        c_lens = [row.num_c_sents for _, row in test_set.iterrows()]
    else:
        c_lens = [len(eval(row.complex)) for _, row in test_set.iterrows()]
    if isinstance(preds[0], torch.Tensor):
        preds = [list(preds[i].cpu().numpy()) for i in range(len(preds))]
    preds_trim = [preds[i][:c_lens[i]] for i in range(len(preds))]
    test_set["pred_l"] = preds_trim

    if "labels" in test_set.columns:
        print("Evaluating...")
        gts = [convert_labels(eval(ls_i), class_labels) for ls_i in test_set.labels]

        labels_flat = [l for x in gts for l in x]
        preds_flat = [l for x in preds_trim for l in x]
        assert len(labels_flat) == len(preds_flat)

        class_scores = precision_recall_fscore_support(labels_flat, preds_flat, average=None)
        micro_scores = precision_recall_fscore_support(labels_flat, preds_flat, average="micro")
        macro_scores = precision_recall_fscore_support(labels_flat, preds_flat, average="macro")

        for cl, i in class_labels.items():
            print(f"{cl}:\n\tP: {class_scores[0][i]}\n\tR: {class_scores[1][i]}\n\tF1: {class_scores[2][i]}")
        print(f"Micro:\n\tP: {micro_scores[0]}\n\tR: {micro_scores[1]}\n\tF1: {micro_scores[2]}")
        print(f"Macro:\n\tP: {macro_scores[0]}\n\tR: {macro_scores[1]}\n\tF1: {macro_scores[2]}")

    # TODO: handle by reading level

    if out_file is not None:
        test_set.to_csv(out_file, index=False)
        print(f"Predictions written to {out_file}.")

    end = time.time()
    elapsed = end - start
    print(f"Done! (Took {elapsed}s in total)")
    print(f"End time: {datetime.now()}")


if __name__ == '__main__':
    fire.Fire(evaluate)
