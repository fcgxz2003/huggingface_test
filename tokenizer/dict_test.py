from transformers import BertTokenizer

if __name__ == '__main__':
    proxies = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-chinese',
        cache_dir=None,
        force_download=False,
        proxies=proxies
    )

    # 获取字典
    zidian = tokenizer.get_vocab()

    type(zidian), len(zidian), '月光' in zidian,

    print(type(zidian), len(zidian), '月光' in zidian,)

    # 添加新词
    tokenizer.add_tokens(new_tokens=['月光', '希望'])

    # 添加新符号
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})

    zidian = tokenizer.get_vocab()

    type(zidian), len(zidian), zidian['月光'], zidian['[EOS]']
    print(type(zidian), len(zidian), zidian['月光'], zidian['[EOS]'])

    # 编码新添加的词
    out = tokenizer.encode(
        text='月光的新希望[EOS]',
        text_pair=None,

        # 当句子长度大于max_length时,截断
        truncation=True,

        # 一律补pad到max_length长度
        padding='max_length',
        add_special_tokens=True,
        max_length=8,
        return_tensors=None,
    )

    print(out)

    tokenizer.decode(out)
