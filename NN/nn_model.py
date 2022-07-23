import paddle

from dataSet import MyDataset
from paddle.io import random_split
from paddle.io import DataLoader
from model import My_modle
from paddle.optimizer import AdamW
from train import train

# 定义超参数，固定随机种子
paddle.seed(2022)
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 256
DROP = 0.2
LR = 7e-4
DECAY_STEP = 7000
GAMMA = 0.5

# 创建数据集、loader

all_dataset = MyDataset('card_transdata.csv')

data_size = len(all_dataset)
train_size = int(data_size * 0.9)
val_size = int(data_size * 0.1)

train_dataset, val_dataset = random_split(all_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE)

# 实现模型，优化器, loss
model = My_modle(DROP)

lr_scheduler = paddle.optimizer.lr.StepDecay(
    LR,
    step_size=DECAY_STEP,
    gamma=GAMMA
)

optimizer = AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters()
)

loss_fun = paddle.nn.BCEWithLogitsLoss()

train(model, train_loader, val_loader, loss_fun, optimizer, 20)
