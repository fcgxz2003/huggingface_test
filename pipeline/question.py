from transformers import pipeline

if __name__ == '__main__':
    # 阅读理解
    question_answerer = pipeline("question-answering")

    context = r"""
    Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a 
    question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune 
    a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.
    """

    result = question_answerer(question="What is extractive question answering?",
                               context=context)
    print(result)

    result = question_answerer(
        question="What is a good example of a question answering dataset?",
        context=context)

    print(result)
