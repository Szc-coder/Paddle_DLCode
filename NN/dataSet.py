import paddle
import pandas as pd
from paddle.io import Dataset
from sklearn import preprocessing


# 自定义数据集
class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """

    def __init__(self, data_url):
        """
        步骤二：实现构造函数，定义数据集大小
        """
        super(MyDataset, self).__init__()
        self.data_url = data_url
        ann = (pd.read_csv(data_url, encoding="utf-8").to_numpy())
        min_max_scaler = preprocessing.MinMaxScaler()
        self.ann = min_max_scaler.fit_transform(ann)

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        no_emb = self.ann[index, 0:3]
        rep = self.ann[index, 3]
        chip = self.ann[index, 4]
        pin = self.ann[index, 5]
        ol = self.ann[index, 6]

        label = self.ann[index, -1]

        no_emb_tensor = paddle.to_tensor(no_emb, dtype='float32')
        rep = paddle.to_tensor(rep, dtype='int32')
        chip = paddle.to_tensor(chip, dtype='int32')
        pin = paddle.to_tensor(pin, dtype='int32')
        ol = paddle.to_tensor(ol, dtype='int32')
        label_tensor = paddle.zeros([2])
        if label == 0.:
            label_tensor[0] = 1
        else:
            label_tensor[1] = 1

        data = {
            'no_emb': no_emb_tensor,
            'rep': rep,
            'chip': chip,
            'pin': pin,
            'ol': ol,
            'label': label_tensor
        }

        return data

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.ann)
