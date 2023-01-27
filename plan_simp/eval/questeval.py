from questeval.questeval_metric import QuestEval

def calculate_questeval(xx, yy_, yy=None, batch_size=512, use_ref=True, log_dir="logs"):
    """
    Compute BERTScore-based QuestEval for given source/prediction pairs.
    """
    questeval = QuestEval(
        list_scores=('answerability', 'bertscore',),
        use_ref=use_ref,
        limit_sent=None,
        log_dir=log_dir,
    )

    score = questeval.corpus_questeval(
        hypothesis=yy_,
        sources=xx,
        list_references=yy if use_ref else None,
        batch_size=batch_size,
    )

    return score["ex_level_scores"]