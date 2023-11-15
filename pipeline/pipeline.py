from transformers import pipeline

if __name__ == '__main__':

    #文本分类
    classifier = pipeline("sentiment-analysis")

    result = classifier("I hate you")[0]
    print(result)

    result = classifier("I love you")[0]
    print(result)