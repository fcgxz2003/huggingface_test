import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AdamW
import random


# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        dataset = load_from_disk('../dataset/data/ChnSentiCorp')
        dataset = dataset[split]

        def f(data):
            return len(data['text']) > 40

        self.dataset = dataset.filter(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']

        # 切分一句话为前半句和后半句
        sentence1 = text[:20]
        sentence2 = text[20:40]
        label = 0

        # 有一半的概率把后半句替换为一句无关的话
        if random.randint(0, 1) == 0:
            j = random.randint(0, len(self.dataset) - 1)
            sentence2 = self.dataset[j]['text'][20:40]
            label = 1

        return sentence1, sentence2, label


# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out

def test():
    model.eval()
    correct = 0
    total = 0

    loader_test = torch.utils.data.DataLoader(dataset=Dataset('test'),
                                              batch_size=32,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader_test):

        if i == 5:
            break

        print(i)

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        pred = out.argmax(dim=1)

        correct += (pred == labels).sum().item()
        total += len(labels)

    print(correct / total)

if __name__ == '__main__':
    dataset = Dataset('train')
    sentence1, sentence2, label = dataset[0]
    print(len(dataset), sentence1, sentence2, label)

    # 加载字典和分词工具
    token = BertTokenizer.from_pretrained('bert-base-chinese')
    print(token)


    def collate_fn(data):
        sents = [i[:2] for i in data]
        labels = [i[2] for i in data]

        # 编码
        data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=45,
                                       return_tensors='pt',
                                       return_length=True,
                                       add_special_tokens=True)

        # input_ids:编码之后的数字
        # attention_mask:是补零的位置是0,其他位置是1
        # token_type_ids:第一个句子和特殊符号的位置是0,第二个句子的位置是1
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        token_type_ids = data['token_type_ids']
        labels = torch.LongTensor(labels)

        # print(data['length'], data['length'].max())

        return input_ids, attention_mask, token_type_ids, labels


    # 数据加载器
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=8,
                                         collate_fn=collate_fn,
                                         shuffle=True,
                                         drop_last=True)

    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader):
        break

    print(len(loader))
    print(token.decode(input_ids[0]))
    print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)

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
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader):
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)

        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 5 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            print(i, loss.item(), accuracy)

        if i == 300:
            break


    test()