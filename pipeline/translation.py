from transformers import pipeline

if __name__ == '__main__':
    # 翻译
    translator = pipeline("translation_en_to_de")

    sentence = "Hugging Face is a technology company based in New York and Paris"

    translator(sentence, max_length=40)
