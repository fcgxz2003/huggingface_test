from transformers import pipeline

if __name__ == '__main__':

    #文本生成
    text_generator = pipeline("text-generation")

    text_generator("As far as I am concerned, I will",
                   max_length=50,
                   do_sample=False)