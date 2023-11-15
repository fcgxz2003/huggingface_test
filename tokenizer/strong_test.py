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

    out = tokenizer.encode_plus(
        text=sents[0],
        text_pair=sents[1],

        # 当句子长度大于max_length时,截断
        truncation=True,

        # 一律补零到max_length长度
        padding='max_length',
        max_length=30,
        add_special_tokens=True,

        # 可取值tf,pt,np,默认为返回list
        return_tensors=None,

        # 返回token_type_ids
        return_token_type_ids=True,

        # 返回attention_mask
        return_attention_mask=True,

        # 返回special_tokens_mask 特殊符号标识
        return_special_tokens_mask=True,

        # 返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
        # return_offsets_mapping=True,

        # 返回length 标识长度
        return_length=True,
    )

    # input_ids 就是编码后的词
    # token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
    # special_tokens_mask 特殊符号的位置是1,其他位置是0
    # attention_mask pad的位置是0,其他位置是1
    # length 返回句子长度
    for k, v in out.items():
        print(k, ':', v)

    re = tokenizer.decode(out['input_ids'])
    print(re)
