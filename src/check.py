import argparse

import spacy

from scoring import bleu_scoring, cossim_scoring
from translate import trans2trans


def check(origin_text):
    trans_ja, trans_en = trans2trans(origin_text)
    print(f"ja -> en\n{trans_en}")
    print(f"en -> ja\n{trans_ja}")

    nlp = spacy.load("ja_ginza")
    origin_doc = nlp(origin_text)
    trans_ja_doc = nlp(trans_ja)

    meaning_score = cossim_scoring(origin_doc, trans_ja_doc)
    bleu_score = bleu_scoring(origin_doc, trans_ja_doc)
    print(f"meaning score : {meaning_score:.3f}")
    print(f"BLEU score : {bleu_score:.3f}")


if __name__ == "__main__":
    origin_text = input()
    check(origin_text)