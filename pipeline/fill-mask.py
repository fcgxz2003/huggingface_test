from transformers import pipeline

if __name__ == '__main__':
    # 完形填空
    unmasker = pipeline("fill-mask")

    from pprint import pprint

    sentence = 'HuggingFace is creating a <mask> that the community uses to solve NLP tasks.'

    unmasker(sentence)
