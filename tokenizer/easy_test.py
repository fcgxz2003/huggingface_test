from transformers import BertTokenizer

if __name__ == '__main__':
    proxies = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-chinese',
        cache_dir=None,
        force_download=False,
        proxies=proxies
    )

    sents = [
        '选择珠江花园的原因就是方便。',
        '笔记本的键盘确实爽。',
        '房间太小。其他的都一般。',
        '今天才知道这书还有第6卷,真有点郁闷.',
        '机器背面似乎被撕了张什么标签，残胶还在。',
    ]

    out = tokenizer.encode(
        text=sents[0],
        text_pair=sents[1],

        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        max_length=30,
        return_tensors=None,
    )
    print(out)

    re = tokenizer.decode(out)
    print(re)

