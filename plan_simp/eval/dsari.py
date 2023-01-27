from __future__ import division

import math
from collections import Counter

import nltk


def dsari_ngram(igrams, ograms, rgramslist, numref):
    rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
    rgramcounter = Counter(rgramsall)

    igramcounter = Counter(igrams)
    igramcounter_rep = Counter()
    for igram, scount in igramcounter.items():
        igramcounter_rep[igram] = scount * numref

    ogramcounter = Counter(ograms)
    ogramcounter_rep = Counter()
    for ogram, ccount in ogramcounter.items():
        ogramcounter_rep[ogram] = ccount * numref

    # KEEP

    keepgramcounter_rep = igramcounter_rep & ogramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = igramcounter_rep & rgramcounter

    keeptmpscore1 = 0
    keeptmpscore2 = 0

    for keepgram in keepgramcountergood_rep:
        keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
        keeptmpscore2 += keepgramcountergood_rep[keepgram] / keepgramcounterall_rep[keepgram]
        # print "KEEP", keepgram, keepscore, ogramcounter[keepgram], igramcounter[keepgram], rgramcounter[keepgram]

    keepscore_precision = 0
    if len(keepgramcounter_rep) > 0:
        keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)

    keepscore_recall = 0
    if len(keepgramcounterall_rep) > 0:
        keepscore_recall = keeptmpscore2 / len(keepgramcounterall_rep)

    keepscore = 0
    if keepscore_precision > 0 or keepscore_recall > 0:
        keepscore = 2 * keepscore_precision * keepscore_recall / (keepscore_precision + keepscore_recall)

    # DELETION

    delgramcounter_rep = igramcounter_rep - ogramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    delgramcounterall_rep = igramcounter_rep - rgramcounter

    deltmpscore1 = 0
    deltmpscore2 = 0

    for delgram in delgramcountergood_rep:
        deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
        deltmpscore2 += delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]

    delscore_precision = 0
    if len(delgramcounter_rep) > 0:
        delscore_precision = deltmpscore1 / len(delgramcounter_rep)

    delscore_recall = 0
    if len(delgramcounterall_rep) > 0:
        delscore_recall = deltmpscore1 / len(delgramcounterall_rep)

    delscore = 0
    if delscore_precision > 0 or delscore_recall > 0:
        delscore = 2 * delscore_precision * delscore_recall / (delscore_precision + delscore_recall)

    # ADDITION

    addgramcounter = set(ogramcounter) - set(igramcounter)
    addgramcountergood = set(addgramcounter) & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(igramcounter)

    addtmpscore = 0
    for addgram in addgramcountergood:
        addtmpscore += 1

    addscore_precision = 0
    if len(addgramcounter) > 0:
        addscore_precision = addtmpscore / len(addgramcounter)

    addscore_recall = 0
    if len(addgramcounterall) > 0:
        addscore_recall = addtmpscore / len(addgramcounterall)

    addscore = 0
    if addscore_precision > 0 or addscore_recall > 0:
        addscore = 2 * addscore_precision * addscore_recall / (addscore_precision + addscore_recall)

    return (keepscore, delscore_precision, addscore)

def count_length(idoc, odoc, rdocss):
    input_length = len(idoc.split(" "))
    output_length = len(odoc.split(" "))

    reference_length = 0
    for rdocs in rdocss:
        reference_length += len(rdocs.split(" "))
    reference_length = int(reference_length / len(rdocss))

    return input_length, reference_length, output_length

def sentence_number(odoc, rdocss):
    output_sentence_number = len(nltk.sent_tokenize(odoc))

    reference_sentence_number = 0
    for rdocs in rdocss:
        reference_sentence_number += len(nltk.sent_tokenize(rdocs))
    reference_sentence_number = int(reference_sentence_number / len(rdocss))

    return reference_sentence_number, output_sentence_number

def dsari_doc(idoc, odoc, rdocss, doc_level=True, global_penalties=False):
    """Compute D-SARI given input doc, output doc, and reference doc(s)."""
    numref = len(rdocss)
    i1grams = idoc.lower().split(" ")
    o1grams = odoc.lower().split(" ")
    i2grams = []
    o2grams = []
    i3grams = []
    o3grams = []
    i4grams = []
    o4grams = []
    r1gramslist = []
    r2gramslist = []
    r3gramslist = []
    r4gramslist = []

    # compile sets of n-grams for reference doc
    for rdocs in rdocss:
        r1grams = rdocs.lower().split(" ")
        r2grams = []
        r3grams = []
        r4grams = []
        r1gramslist.append(r1grams)

        for i in range(0, len(r1grams) - 1):
            if i < len(r1grams) - 1:
                r2gram = r1grams[i] + " " + r1grams[i + 1]
                r2grams.append(r2gram)

            if i < len(r1grams) - 2:
                r3gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2]
                r3grams.append(r3gram)

            if i < len(r1grams) - 3:
                r4gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2] + " " + r1grams[i + 3]
                r4grams.append(r4gram)

        r2gramslist.append(r2grams)
        r3gramslist.append(r3grams)
        r4gramslist.append(r4grams)

    # infer input n-grams from unigrams
    for i in range(0, len(i1grams) - 1):
        if i < len(i1grams) - 1:
            i2gram = i1grams[i] + " " + i1grams[i + 1]
            i2grams.append(i2gram)

        if i < len(i1grams) - 2:
            i3gram = i1grams[i] + " " + i1grams[i + 1] + " " + i1grams[i + 2]
            i3grams.append(i3gram)

        if i < len(i1grams) - 3:
            i4gram = i1grams[i] + " " + i1grams[i + 1] + " " + i1grams[i + 2] + " " + i1grams[i + 3]
            i4grams.append(i4gram)

    # infer output n-grams from unigrams
    for i in range(0, len(o1grams) - 1):
        if i < len(o1grams) - 1:
            o2gram = o1grams[i] + " " + o1grams[i + 1]
            o2grams.append(o2gram)

        if i < len(o1grams) - 2:
            o3gram = o1grams[i] + " " + o1grams[i + 1] + " " + o1grams[i + 2]
            o3grams.append(o3gram)

        if i < len(o1grams) - 3:
            o4gram = o1grams[i] + " " + o1grams[i + 1] + " " + o1grams[i + 2] + " " + o1grams[i + 3]
            o4grams.append(o4gram)

    (keep1score, del1score, add1score) = dsari_ngram(i1grams, o1grams, r1gramslist, numref)
    (keep2score, del2score, add2score) = dsari_ngram(i2grams, o2grams, r2gramslist, numref)
    (keep3score, del3score, add3score) = dsari_ngram(i3grams, o3grams, r3gramslist, numref)
    (keep4score, del4score, add4score) = dsari_ngram(i4grams, o4grams, r4gramslist, numref)

    avgkeepscore = sum([keep1score, keep2score, keep3score, keep4score]) / 4
    avgdelscore = sum([del1score, del2score, del3score, del4score]) / 4
    avgaddscore = sum([add1score, add2score, add3score, add4score]) / 4

    # Document-level penalties

    input_length, reference_length, output_length = count_length(idoc, odoc, rdocss)
    reference_sentence_number, output_sentence_number = sentence_number(odoc, rdocss)
    
    if doc_level:
        if output_length >= reference_length:
            LP_1 = 1
        else:
            LP_1 = math.exp((output_length - reference_length) / output_length)

        if output_length > reference_length:
            LP_2 = math.exp((reference_length - output_length) / max(input_length - reference_length, 1))
        else:
            LP_2 = 1

        SLP = math.exp(-abs(reference_sentence_number - output_sentence_number) / max(reference_sentence_number,
                                                                                    output_sentence_number))

        if not global_penalties:
            avgkeepscore = avgkeepscore * LP_2 * SLP
            avgaddscore = avgaddscore * LP_1
            avgdelscore = avgdelscore * LP_2
        else:
            avgkeepscore = avgkeepscore * LP_1 * LP_2 * SLP
            avgaddscore = avgaddscore * LP_1 * LP_2 * SLP
            avgdelscore = avgdelscore * LP_1 * LP_2 * SLP

    finalscore = (avgkeepscore + avgdelscore + avgaddscore) / 3

    return finalscore, avgkeepscore, avgdelscore, avgaddscore

def example_dsari():

    idoc = "marengo is a town in and the county seat of iowa county , iowa , united states . it has served as the county seat since august 1845 , even though it was not incorporated until july 1859 . the population was 2,528 in the 2010 census , a decline from 2,535 in 2000 ."

    odoc1 = "in the US . 2,528 in 2010 ."
    odoc2 = "marengo is a city in iowa , the US . it has served as the county seat since august 1845 , even though it was not incorporated . the population was 2,528 in the 2010 census , a decline from 2,535 in 2010 ."
    odoc3 = "marengo is a town in iowa . marengo is a town in the US . in the US . the population was 2,528 . the population in the 2010 census ."
    odoc4 = "marengo is a town in iowa , united states . in 2010 , the population was 2,528 ."
    rdocss = ["marengo is a city in iowa in the US . the population was 2,528 in 2010 ."]
    
    print(dsari_doc(idoc, odoc1, rdocss))
    print(dsari_doc(idoc, odoc2, rdocss))
    print(dsari_doc(idoc, odoc3, rdocss))
    print(dsari_doc(idoc, odoc4, rdocss))
    
# if __name__ == '__main__':
#     example_dsari()