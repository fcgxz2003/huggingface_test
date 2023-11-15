from datasets import load_from_disk

if __name__ == '__main__':
    # 从磁盘加载数据
    dataset = load_from_disk('./data/ChnSentiCorp')
    print(dataset)

    # 取出训练集
    dataset = dataset['train']
    print(dataset)

    # 查看一个数据
    print(dataset[0])

    # sort
    # 未排序的label是乱序的
    print(dataset['label'][:10])

    # 排序之后label有序了
    sorted_dataset = dataset.sort('label')
    print(sorted_dataset['label'][:10])
    print(sorted_dataset['label'][-10:])

    # shuffle
    # 打乱顺序
    shuffled_dataset = sorted_dataset.shuffle(seed=42)
    print(shuffled_dataset['label'][:10])

    # select
    dataset.select([0, 10, 20, 30, 40, 50])

    # filter
    def f(data):
        return data['text'].startswith('选择')

    start_with_ar = dataset.filter(f)
    len(start_with_ar), start_with_ar['text']

    #train_test_split, 切分训练集和测试集
    dataset.train_test_split(test_size=0.1)

    # shard
    # 把数据切分到4个桶中,均匀分配
    dataset.shard(num_shards=4, index=0)

    # rename_column
    dataset.rename_column('text', 'textA')

    # remove_columns
    dataset.remove_columns(['text'])


    # map
    def f(data):
        data['text'] = 'My sentence: ' + data['text']
        return data

    datatset_map = dataset.map(f)
    datatset_map['text'][:5]

    # set_format
    dataset.set_format(type='torch', columns=['label'])
    dataset[0]
