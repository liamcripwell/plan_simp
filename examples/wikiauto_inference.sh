: '
This script will illustrate an example of how the pretrained PG_Dyn system can be used to perform out-of-domain inference.
In this example we use the validation set of Wiki-auto.
'

# encode sentence-level context embeddings to be used by the planner
python plan_simp/scripts/encode_contexts.py 
  examples/wikiauto_docs_valid.csv 
  --save_dir=fake_context_dir/

# run inference with pretrained models
python plan_simp/scripts/generate.py dynamic 
  --clf_model_ckpt=liamcripwell/pgdyn-plan
  --model_ckpt=liamcripwell/pgdyn-simp
  --test_file=examples/wikiauto_sents_valid.csv
  --doc_id_col=pair_id
  --context_doc_id=pair_id
  --context_dir=fake_context_dir
  --reading_lvl=s_level 
  --out_file=test_out.csv