import os
import re
import math
from typing import Dict

import torch
import pandas as pd

CLASS_LABELS = {"ignore": 0, "rephrase": 1, "split": 2, "delete": 3}
M_CLASS_LABELS = {"ignore": 0, "rephrase": 1, "ssplit": 2, "dsplit": 3, "delete": 4}

READING_LVLS = ["<RL_0>", "<RL_1>", "<RL_2>", "<RL_3>", "<RL_4>"]
OP_TOKENS = ["<COPY>", "<REPHRASE>", "<SPLIT>", "<DELETE>"]
M_OP_TOKENS = ["<COPY>", "<REPHRASE>", "<SSPLIT>", "<DSPLIT>", "<DELETE>"]
PLAN_TOKENS = ["[PLAN]", "[SIMPLIFICATION]"]


def convert_labels(labels, label_dict: Dict):
    """Convert string class labels to class ids."""
    # handle case of single sentence-level label
    if isinstance(labels, str):
        labels = [labels]

    if isinstance(labels[0], str):
        labels_ = []
        for l in labels:
            # account for splitting sub-types
            if l not in label_dict and l in ["ssplit", "dsplit"]:
                l = "split"
            labels_.append(label_dict[l])
        return labels_
    else:
        return labels

def prepend_tokens(df, x_col, class_labels=None, op_tokens=None, op_col=None, lvl_col=None):
    """Append specified control tokens to text and return results as List."""
    seqs = [row[x_col] for _, row in df.iterrows()]

    # reading level
    if lvl_col is not None:
        for i, row in df.iterrows():
            # can either use column value or manually specified level constant
            tok = row[lvl_col] if not isinstance(lvl_col, int) else lvl_col
            seqs[i] = f"{READING_LVLS[tok]} {seqs[i]}"

    # operation control-tokens
    if op_col is not None:
        op_labels = list(df[op_col])
        if isinstance(op_labels[0], list) or (isinstance(op_labels[0], str) and op_labels[0][0] == '['):
            # if many operations per input, prepare as a list
            op_labels = [convert_labels(eval(ys) if isinstance(ys, str) else ys, class_labels) for ys in op_labels]
        else:
            # regularize singular operation as 1-value list
            op_labels = [[y] for y in convert_labels(op_labels, class_labels)]

        # prepend operation tokens
        for i, row in df.iterrows():
            plan = " ".join([op_tokens[y] for y in op_labels[i]])
            seqs[i] = f"{plan} {seqs[i]}"

    return seqs

def prepare_plan_seqs(df, plan_col, class_labels, op_tokens):
    """Turn document-plan column into operation token sequence."""
    plan_labels = [convert_labels(eval(p) if p[0] == "[" else p, class_labels) for p in df[plan_col]]
    plan_seqs = [" ".join([op_tokens[o] for o in p]) for p in plan_labels]
    return plan_seqs

def prepare_plan_seps(df, ref_col, plan_col, class_labels, op_tokens):
    """Prepare reference documents with plan operation separators."""
    new_refs = []
    for _, row in df.iterrows():
        plan_labels = [op_tokens[o] for o in convert_labels(eval(row[plan_col]), class_labels)]
        simps = [" ".join(s) for s in eval(row[ref_col])]
        new_ref = " ".join([val for pair in zip(plan_labels, simps) for val in pair])
        new_refs.append(new_ref)
    
    return new_refs

def prepare_plan_prefixes(df, ref_col, plan_col, class_labels, op_tokens, sent_level=False):
    """Prepare reference documents with plan prefixes."""
    plans = prepare_plan_seqs(df, plan_col, class_labels, op_tokens)
    new_refs = []
    for i, row in df.iterrows():
        prefix = f"{PLAN_TOKENS[0]} {plans[i]} {PLAN_TOKENS[1]}" if not sent_level else plans[i]
        new_refs.append(f"{prefix} {row[ref_col] if not sent_level else ' '.join(eval(row[ref_col]))}")
    
    return new_refs

def tagger_results_to_sent(df_results, df_sent, pred_col="pred_l"):
    preds = []
    for _, row in df_results.iterrows():
        ls = eval(row[pred_col])
        preds += ls
    df_sent[pred_col] = preds
    
    return df_sent

def load_sent_context_tensors(context_ids, context_dir, sep="#$#", context_window=13, simple_ids=None, doc_pos=None, 
                            simple_context_dir=None, left_z_only=False):
    """Loads and prepares tensors for contexts."""
    full_contexts = []
    sent_pos_ids = []
    doc_pos_ids = []
    for i, member in enumerate(context_ids):
        doc_id, sent_id = [x for x in member.split(sep)]
        sent_id = int(sent_id)

        # load pre-encoded vectors from directory
        z = torch.load(f"{context_dir}/{doc_id}_z.pt")
        left_z = z[:sent_id]
        right_z = z[sent_id+1:]
        self_z = z[sent_id:sent_id+1] # make sure it has same dims as context

        # use dynamic left-context
        if simple_context_dir is not None and simple_ids is not None:
            s_doc_id, s_sent_id = [x for x in simple_ids[i].split(sep)]
            s_sent_id = int(s_sent_id)
            if not os.path.exists(f"{simple_context_dir}/{s_doc_id}_z.pt"):
                # warn if there is currently no cached version of the dynamic context
                print(f"No dynamic context available for {simple_context_dir}/{s_doc_id}. (This is expected early on)")
            else:
                # load simple document's sentence vectors
                z_s = torch.load(f"{simple_context_dir}/{s_doc_id}_z.pt")
                # re-assign left_z with simple sentence vectors
                if s_sent_id > -1:
                    # use entire simple context for left_z (will be the case during inference with dynamic context)
                    left_z = z_s[:s_sent_id]
                else:
                    left_z = z_s
        
        # trim length of context seqs
        if len(left_z) > context_window:
            left_z = left_z[-context_window:]
        if len(right_z) > context_window:
            right_z = right_z[:context_window]

        # enfore left-only context
        if left_z_only:
            right_z = right_z[:0]
        
        # z = torch.concat([left_z, torch.zeros(1, 768), right_z]) # NOTE: use for self sent masking
        z = torch.concat([left_z, self_z, right_z]) # NOTE: use for self context

        full_contexts.append(z)

        # prepare position ids for each sentence
        first_pos_id = context_window - len(left_z) + 1
        last_pos_id = context_window + len(right_z) + 2
        rel_pos = torch.arange(first_pos_id, last_pos_id) # NOTE: use for self context
        # rel_pos[rel_pos==context_window+1] = 0 # NOTE: use for self sent masking
        sent_pos_ids.append(rel_pos)

        # prepare document position ids
        if doc_pos is not None:
            sent_doc_pos, doc_len = doc_pos[i]
            sent_doc_pos = round(sent_doc_pos*doc_len)
            doc_pos_raw = torch.arange(sent_doc_pos-len(left_z), sent_doc_pos+len(right_z)+1)
            doc_pos_raw[doc_pos_raw<=0] = 1
            # gets quintile for each document position
            doc_posx = torch.ceil(((doc_pos_raw) / doc_len) / 0.2)

            doc_pos_ids.append(doc_posx)

    if doc_pos is not None:
        return full_contexts, sent_pos_ids, doc_pos_ids

    return full_contexts, sent_pos_ids

def prepare_batch_context(self, batch, labels, bart=False):
    data = {}

    # load context data for batch
    contexts = [x[1] for x in batch]
    sent_pos_ids = None
    doc_pos_ids = None

    simple_contexts = None
    if self.has_param("simple_context_dir"):
        # load simple document (dynamic) context
        simple_contexts = [x[2] for x in batch]

    # load doc position information
    doc_position = None
    if self.has_param("doc_pos_embeds") or bart:
        batch_loc = -1 if labels is None else -2
        doc_position = [x[batch_loc] for x in batch]

    contexts, sent_pos_ids, doc_pos_ids = load_sent_context_tensors(contexts, self.hparams.context_dir, sep=self.sep,
                                                                    context_window=self.hparams.context_window, 
                                                                    simple_ids=simple_contexts, doc_pos=doc_position,
                                                                    simple_context_dir=self.get_param("simple_context_dir"),
                                                                    left_z_only=self.has_param("left_z_only"))
    
    max_context_len = max([len(z) for z in contexts])
    sent_embeds = []
    for i, z in enumerate(contexts):
        context_size = len(z)

        # context sentence padding
        z = torch.cat([z, torch.zeros(max_context_len-len(z), 768)])
        sent_embeds.append(z)

        # add padding to context position id seqs
        if sent_pos_ids is not None:
            pos_pad = torch.zeros(max_context_len-context_size) - 1 # for roberta pos padding
            sent_pos_ids[i] = torch.cat([sent_pos_ids[i], pos_pad]).long()

        # add padding to document position id seqs
        if doc_pos_ids is not None:
            doc_pos_ids[i] = torch.cat([doc_pos_ids[i], pos_pad]).long()

    data["context_hidden_states"] = torch.stack(sent_embeds)
    if sent_pos_ids is not None:
        data["context_position_ids"] = torch.stack(sent_pos_ids)
    if doc_pos_ids is not None:
        data["document_position_ids"] = torch.stack(doc_pos_ids)

    return data

def merge_preds_to_paras(df, df_para, sent_col="simple", new_col="pred"):
    # clean nan predictions
    df.pred = [row.pred if not row.isna().pred else "" for i, row in df.iterrows()]

    preds = []
    for _, para in df_para.iterrows():
        sents = list(df[(df.pair_id == para.pair_id) & (df.para_id == para.para_id)].sort_values(by=["sent_id"])[sent_col])

        # handle list columns
        if len(sents[0]) > 0 and sents[0][0] == "[":
            sents = [" ".join(eval(ss)) for ss in sents]

        preds.append(" ".join(sents))
    df_para_new = df_para.copy()
    df_para_new[new_col] = preds

    return df_para_new

def docs_to_sents(df, keep_align=True):
    items = []
    for i, row in df.iterrows():
        c_sents = eval(row.complex)
        s_sents = eval(row.simple)
        for j in range(len(c_sents)):
            # get positional info
            pos = (j + 1) / row.num_c_sents
            quint = math.ceil(pos / 0.2)

            if keep_align:
                simple = " ".join(s_sents[j])

                # flatten simple doc to get aligned S sent index for C sent
                simp_sent_id = len([s for c in eval(row.simple)[:j] for s in c])
            else:
                simp_sent_id = min(j, len(s_sents)-1)
                simple = s_sents[simp_sent_id]

            items.append( (row.title, row.pair_id, j, c_sents[j], simple, simp_sent_id,
                            eval(row.para_id)[j], pos, quint, row.num_c_sents) )

    df_sent = pd.DataFrame(items, columns=['title', 'pair_id', 'sent_id', 'complex', 'simple', "simp_sent_id",
                                           "para_id", "doc_pos", "doc_quint", "doc_len"])
    return df_sent

def sents_to_paras(df):
    items = []
    for title in df.title.unique():
        rows = df[df.title == title]
        for para in rows.para_id.unique():
            sents = rows[rows.para_id == para]
            items.append( (title, sents.iloc[0].pair_id, sents.iloc[0].para_id, list(sents.sent_id), 
                            " <s> ".join(sents.complex).strip(), " <s> ".join(sents.simple).strip(), 
                            list(sents.simp_sent_id), list(sents.doc_pos), list(sents.doc_quint), list(sents.doc_len)) )
    
    df_para = pd.DataFrame(items, columns=['title', 'pair_id', "para_id", 'sent_id', 'complex', 'simple', 
                                            "simp_sent_id", "doc_pos", "doc_quint", "doc_len"])
    return df_para

def get_pred_doc_sequences(df, text_col="pred", doc_id_col="pair_id", text_id_col="sent_id", order=None):
    """Return list of document sequences."""
    df[text_col] = df[text_col].fillna("") # clean empty preds (deletes)

    if order is None:
        order = df[doc_id_col].unique()

    doc_seqs = []
    for doc_id in order:
        texts = df[df[doc_id_col] == doc_id].sort_values(by=[text_id_col])[text_col]
        out_str = re.sub(" +", " ", " ".join(texts)).strip() # join sents and remove extra whitespace
        doc_seqs.append(out_str)
    
    return doc_seqs