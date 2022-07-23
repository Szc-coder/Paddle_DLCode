import paddle.nn as nn
import paddle


# 定义模型
class My_modle(nn.Layer):
    def __init__(self, drop):
        super(My_modle, self).__init__()
        self.emb_rep = EmbLayer(num=2, dim=16, out_dim=16)
        self.emb_chip = EmbLayer(num=2, dim=16, out_dim=16)
        self.emb_pin = EmbLayer(num=2, dim=16, out_dim=16)
        self.emb_ol = EmbLayer(num=2, dim=16, out_dim=16)

        self.dense_fc = FCWithAct(3, 16, drop)

        self.fc1 = FCWithAct(16, 224, drop)
        self.fc2 = FCWithAct(224, 448, drop)
        self.fc3 = FCWithAct(448, 448, drop)

        self.class_head = nn.Linear(448, 2)

    def forward(self, inputs):
        x1 = self.dense_fc(inputs['no_emb'])
        x1 = paddle.unsqueeze(x1, 1)

        x2 = self.emb_rep(inputs['rep'])
        x3 = self.emb_chip(inputs['chip'])
        x4 = self.emb_pin(inputs['pin'])
        x5 = self.emb_ol(inputs['ol'])

        x = paddle.concat([x1, x2, x3, x4, x5], 1)

        y = self.fc1(x)
        y = self.fc2(y)
        y = paddle.mean(y, 1)
        y = paddle.squeeze(y, 1)
        y = self.fc3(y)

        out = self.class_head(y)
        return out


class EmbLayer(nn.Layer):
    def __init__(self, num, dim, out_dim):
        super(EmbLayer, self).__init__()
        self.emb = nn.Embedding(num, dim)
        self.fc = nn.Linear(dim, out_dim)

    def forward(self, inputs):
        y = self.emb(inputs)
        y = self.fc(y)

        return y


class FCWithAct(nn.Layer):
    def __init__(self, in_dim, out_dim, drop):
        super(FCWithAct, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, inputs):
        y = self.linear(inputs)
        y = self.act(y)
        y = self.drop(y)
        return y
