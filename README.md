# plan_simp

This repo will contain code and resources for the EACL 2023 paper, _Document-Level Planning for Text Simplification_.

We will progressively update with code, instructions, pretrained models, data, etc. as soon as we can make them available.

## Installation

```bash
git clone https://github.com/liamcripwell/plan_simp.git
cd plan_simp
pip install -e .
```

## Pretrained models
We provide pretrained models for the components of our contextual planning system "PG_Dyn" on [HuggingFace](https://huggingface.co/liamcripwell).

These can be loaded within Python as follows:
```python
from plan_simp.models.classifier import load_planner
from plan_simp.models.bart import load_simplifier

# contextual simplification planner
planner, p_tokenizer, p_hparams = load_planner("liamcripwell/pgdyn-plan")

# simplification model
simplifier, tokenizer, hparams = load_simplifier("liamcripwell/pgdyn-simp")
```

To perform end-to-end inference with the full pipeline, see the following section.

## Plan-Guided Simplification
A planner can be used to dynamically generate simplified documents. The planner will iteratively predict an operation for the current sentence (given the document context) and pass this to an encoder-decoder to conditionally generate a simplification. These simplifications are then used within the context of subsequent sentences. There is no need to pass a `--simple_context_dir` argument because dynamic context will be managed on-the-fly.

```bash
python plan_simp/scripts/generate.py dynamic 
  --clf_model_ckpt=<planner_model> # e.g. liamcripwell/pgdyn-plan
  --model_ckpt=<simplification_model> # e.g. liamcripwell/pgdyn-simp
  --test_file=<test_sentences>
  --doc_id_col=pair_id # document identifier for each sentence
  --context_doc_id=c_id
  --context_dir=<context_dir>
  --reading_lvl=s_level 
  --out_file=<output_csv> 
```

Alternatively, plan-guided simplification can be formed with pre-determined operation labels.

```bash
python plan_simp/scripts/generate.py inference 
  --model_ckpt=<simplification_model> # e.g. liamcripwell/pgdyn-simp
  --test_file=<test_sentences> 
  --op_col=label
  --reading_lvl=s_level 
  --out_file=<output_csv> 
```

## Training a planner

The following commands show example use cases for training your own planner models.

For Newsela, make sure to use the `--reading_lvl` flag to specify a column in the data which indicates the target reading level of the simplification.

```bash
# baseline
python plan_simp/scripts/train_clf.py 
  --train_file=<train_file> 
  --val_file=<val_file> 
  --x_col=complex 
  --y_col=label 
  --batch_size=32
  --learning_rate=1e-5
  --ckpt_metric=val_macro_f1

# contextual (PG variants)
python plan_simp/scripts/train_clf.py 
  --train_file=<train_file> 
  --val_file=<val_file> 
  --x_col=complex 
  --y_col=label 
  --batch_size=32 
  --learning_rate=1e-5
  --ckpt_metric=val_macro_f1  
  --add_context 
  --context_doc_id=pair_id 
  --context_dir=<context_dir> 
  --context_window=13
  # used for dynamic context loading
  --simple_context_doc_id=pair_id
  --simple_context_dir=<context_dir>

# sequence tagging
python plan_simp/scripts/train_tagger.py 
  --train_file=<train_file> 
  --val_file=<val_file> 
  --x_col=<doc_id> 
  --y_col=<labels> 
  --batch_size=32 
  --learning_rate=1e-5
  --embed_dir=<sent_rep_dir>
```

## Evaluating a planner

```bash
# baseline
python plan_simp/scripts/eval_clf.py <planner_model> <test_set>

# contextual
python plan_simp/scripts/eval_clf.py <planner_model> <test_set> 
  --add_context=True 
  --context_dir=<context_dir> 
  --context_doc_id=pair_id
  # used for dynamic context loading
  --simple_context_dir=<context_dir>
  --simple_context_doc_id=pair_id
  --reading_lvl=s_level # for Newsela

# sequence tagging; for Newsela add --x_col=c_id --reading_lvl=s_level
python plan_simp/scripts/eval_tagger.py <planner_model> <test_set> --embed_dir=<sent_rep_dir>
```