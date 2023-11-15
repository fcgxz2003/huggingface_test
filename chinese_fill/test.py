import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AdamW

# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(768, token.vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(token.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        out = self.decoder(out.last_hidden_state[:, 15])

        return out


#测试
def test(dataset):
    model.eval()
    correct = 0
    total = 0

    loader_test = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=32,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader_test):

        if i == 15:
            break

        print(i)

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

        print(token.decode(input_ids[0]))
        print(token.decode(labels[0]), token.decode(labels[0]))

    print(correct / total)


if __name__ == '__main__':
    dataset = load_from_disk('../dataset/data/ChnSentiCorp')

    dataset = dataset['train']


    def f(data):
        return len(data['text']) > 30


    train_dataset = dataset.filter(f)
    print(len(train_dataset), train_dataset[0])

    # 加载字典和分词工具
    token = BertTokenizer.from_pretrained('bert-base-chinese')


    def collate_fn(data):
        sents = [i['text'] for i in data]

        # 编码
        data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=30,
                                       return_tensors='pt',
                                       return_length=True)

        # input_ids:编码之后的数字
        # attention_mask:是补零的位置是0,其他位置是1
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        token_type_ids = data['token_type_ids']

        # 把第15个词固定替换为mask
        labels = input_ids[:, 15].reshape(-1).clone()
        input_ids[:, 15] = token.get_vocab()[token.mask_token]

        # print(data['length'], data['length'].max())

        return input_ids, attention_mask, token_type_ids, labels


    # 数据加载器
    loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=16,
                                         collate_fn=collate_fn,
                                         shuffle=True,
                                         drop_last=True)
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader):
        break

    print(len(loader))
    print(token.decode(input_ids[0]))
    print(token.decode(labels[0]))
    print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)

    # 训练极其慢
    # 加载预训练模型
    pretrained = BertModel.from_pretrained('bert-base-chinese')

    # 不训练,不需要计算梯度
    for param in pretrained.parameters():
        param.requires_grad_(False)

    # 模型试算
    out = pretrained(input_ids=input_ids,
                     attention_mask=attention_mask,
                     token_type_ids=token_type_ids)

    print(out.last_hidden_state.shape)

    model = Model()
    re = model(input_ids=input_ids,
               attention_mask=attention_mask,
               token_type_ids=token_type_ids).shape
    print(re)

    # 训练
    optimizer = AdamW(model.parameters(), lr=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(5):
        for i, (input_ids, attention_mask, token_type_ids,
                labels) in enumerate(loader):
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 50 == 0:
                out = out.argmax(dim=1)
                accuracy = (out == labels).sum().item() / len(labels)

                print(epoch, i, loss.item(), accuracy)

    # 测试
    test_dataset = dataset['test']
    test(test_dataset)