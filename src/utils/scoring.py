import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity


def bleu_scoring(origin_doc, ja_doc):
    bleu_score = sentence_bleu([[token.text for token in ja_doc]], [token.text for token in origin_doc], weights=(0.5, 0.5, 0, 0))
    return bleu_score


def cossim_scoring(origin_doc, ja_doc):
    origin_vector = np.mean([token.vector for token in origin_doc if not token.is_oov], axis=0).reshape(1, -1)
    ja_vector = np.mean([token.vector for token in ja_doc if not token.is_oov], axis=0).reshape(1, -1)
    return cosine_similarity(origin_vector, ja_vector).squeeze()
