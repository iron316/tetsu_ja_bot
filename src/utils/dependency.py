from nltk import Tree


def token_format(tk):
    return "_".join([tk.orth_, tk.dep_, tk.pos_])


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(token_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return token_format(node)


def check_tree(doc):
    for sent in doc.sents:
        to_nltk_tree(sent.root).pretty_print()
