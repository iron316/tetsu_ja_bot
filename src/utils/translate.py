from googletrans import Translator


def trans2trans(origin_text):
    translater = Translator()
    en_text = translater.translate(origin_text, dest="en").text
    ja_text = translater.translate(en_text, dest="ja").text
    return ja_text, en_text
