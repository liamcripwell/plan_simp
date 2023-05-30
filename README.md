# plan_simp

This repo contains code and resources for the following papers:
* EACL 2023 paper, [_Document-Level Planning for Text Simplification_](https://aclanthology.org/2023.eacl-main.70/).
* ACL 2023 Findings paper, [_Context-Aware Document Simplification_](https://arxiv.org/abs/2305.06274).

We will progressively update with code, instructions, pretrained models, data, etc. as soon as we can make them available.

## Installation

```bash
git clone https://github.com/liamcripwell/plan_simp.git
cd plan_simp
pip install -e .
```

## Pretrained models
We provide pretrained models for the components of our contextual planning system `PG_Dyn` as well as several of the models proposed in [_Context-Aware Document Simplification_](https://arxiv.org/abs/2305.06274) on [HuggingFace](https://huggingface.co/liamcripwell).

Systems can be loaded within Python as follows:
```python
from plan_simp.models.classifier import load_planner
from plan_simp.models.bart import load_simplifier

# contextual simplification planner
planner, p_tokenizer, p_hparams = load_planner("liamcripwell/pgdyn-plan")

# simplification model
simplifier, tokenizer, hparams = load_simplifier("liamcripwell/pgdyn-simp")
```

An example use-case of inference on out-of-domain test data is illustrated in [this script](examples/wikiauto_inference.sh).

## Data
The Wiki-Auto data used to train the relevant planners and simplification models can be downloaded [here](https://drive.google.com/file/d/1lU8htUIVBuuU24HrPErpV01hlA6tc-d1/view?usp=sharing). Please contact the authors for more information regarding the Newsela data once you have obtain a licence.

## Preparing context representations
We provide a script to generate sentence-level context encodings which are used within `pgdyn-plan` and `conbart`.

```bash
# encode sentence-level context embeddings to be used by the planner
python plan_simp/scripts/encode_contexts.py \
	--data=examples/wikiauto_docs_valid.csv \
	--x_col=complex \
	--id_col=pair_id \
	--save_dir=fake_context_dir/
```

## Plan-guided simplification
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

Alternatively, plan-guided simplification can be formed with pre-determined operation labels, or with no guidance at all.

```bash
# plan-guidance with predefined operations in `"label"` column
python plan_simp/scripts/generate.py inference 
  --model_ckpt=<simplification_model> # e.g. liamcripwell/pgdyn-simp
  --test_file=<test_sentences> 
  --op_col=label
  --reading_lvl=s_level 
  --out_file=<output_csv> 
  
# generation with end-to-end model (no plan guidance)
python plan_simp/scripts/generate.py inference 
    --model_ckpt=liamcripwell/ledpara 
    --test_file=<test_data>
    --reading_lvl=s_level 
    --out_file=<output_csv> 
```

Generation can also be performed within python (see the source code for more parameter details).

```python
# basic generation with no planning
from plan_simp.models.bart import run_generator
preds = run_generator(model_ckpt="liamcripwell/ledpara", **params)

# dynamic plan-guided generation
from plan_simp.scripts.generate import Launcher
launcher = Launcher()
launcher.dynamic(model_ckpt="liamcripwell/o-conbart", clf_model_ckpt="liamcripwell/pgdyn-plan", **params)
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

## Evaluating simplification
Below is an example of how to evaluate simplification outputs from the systems. BARTScore is disabled by default but can be enabled by pointing to a model via the `--bartscore_path` flag.

```bash
# evaluate simplification performance
python plan_simp/scripts/eval_simp.py \
  --input_data=examples/wikiauto_docs_valid.csv \ # document-level input
  --output_data=test_out.csv \ # sentence-level predictions (will be automatically merged)
  --x_col=complex \
  --r_col=simple \
  --y_col=pred \
  --doc_id_col=pair_id \
  --prepro=True \
  --sent_level=True
```

## Citation

If you find this repository useful, please cite our publications: 

* [Document-Level Planning for Text Simplification](https://aclanthology.org/2023.eacl-main.70/)
```bibtex
@inproceedings{cripwell-etal-2023-document,
    title = "Document-Level Planning for Text Simplification",
    author = {Cripwell, Liam  and
      Legrand, Jo{\"e}l  and
      Gardent, Claire},
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.70",
    pages = "993--1006",
}
```

* [Context-Aware Document Simplification](https://arxiv.org/abs/2305.06274)
```bibtex
@misc{cripwell2023contextaware,
      title={Context-Aware Document Simplification}, 
      author={Liam Cripwell and Joël Legrand and Claire Gardent},
      year={2023},
      eprint={2305.06274},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
