from easse.bertscore import get_bertscore_sentence_scores

def calculate_bertscore(yy_, yy):
    """
    Compute BERTScore for given prediction/ground-truth pairs.
    """
    if not isinstance(yy[0], list): yy = [yy]
    p, r, f = get_bertscore_sentence_scores(yy_, yy)

    # return precision sub-metric
    return p.tolist()

