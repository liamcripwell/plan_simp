import os
import time
from datetime import datetime

import fire
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer


def encode(data, save_dir, x_col="complex", id_col="pair_id", max_samples=None, device="cpu"):
    data = pd.read_csv(data)

    if max_samples is not None:
        data = data[:max_samples]

    start = time.time()
    print(f"Starting time: {datetime.now()}")

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if isinstance(id_col, list):
        # derive an id from multiple columns
        doc_ids = ["#?#".join([str(row[c]) for c in id_col]) for _, row in data.iterrows()]
    else:
        doc_ids = [z for z in data[id_col]]

    # figure out whether documents are lists of sentences or combined strings
    if data[x_col].iloc[0][0] == "[":
        data = [eval(z) for z in data[x_col]]
    else:
        data = [z.split( " <s> ") for z in data[x_col]]

    # flatten simple sentence groupings if needed
    if isinstance(data[0][0], list):
        for i, doc in enumerate(data):
            data[i] = [s for g in doc for s in g]

    sent_encoder = SentenceTransformer('all-mpnet-base-v2')

    for i, doc in enumerate(data):
        # NOTE: we need tensors on CPU before saving them if we want to load them while using ddp.
        # If we have access to a GPU, then we can use it to encode the tensor before transferring it
        # to the CPU, which makes this significantly faster than encoding with the CPU as well.
        z = sent_encoder.encode(doc, convert_to_tensor=True).to(device)
        torch.save(z, f"{save_dir}/{doc_ids[i]}_z.pt")

    end = time.time()
    elapsed = end - start
    print(f"Done! (Took {elapsed}s in total)")
    print(f"End time: {datetime.now()}")


if __name__ == '__main__':
    fire.Fire(encode)