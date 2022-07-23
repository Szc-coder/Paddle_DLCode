import paddle

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score


def getF1(predictions, labels):
    # prediction and labels are all level-2 class ids

    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')
    mean_f1 = (f1_micro + f1_macro) / 2.0

    eval_results = {'f1_micro': f1_micro,
                    'f1_macro': f1_macro,
                    'mean_f1': mean_f1}

    return eval_results


# 定义训练函数
def train(model, train_loader, val_loader, loss_fun, opt, epoch):
    for i in range(epoch):

        mean_loss = paddle.zeros([1])

        train_loader = tqdm(train_loader)

        # train
        model.train()
        for step, data in enumerate(train_loader):
            pred = model(data)
            loss = loss_fun(pred, data['label'])

            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
            # 64, 1
            pred_acc = paddle.argmax(pred, 1)
            label = paddle.topk(data['label'], 1)[1]

            acc = accuracy_score(label.cpu().numpy(), pred_acc.cpu().numpy())

            loss.backward()
            opt.step()
            opt.clear_grad()

            train_loader.desc = "[epoch {}] loss{} mean_loss{} acc{} lr{}".format(i,
                                                                                  round(loss.item(), 2),
                                                                                  round(mean_loss.item(), 2),
                                                                                  round(acc, 2), opt.get_lr())
        # eval
        model.eval()

        predictions = []
        labels = []
        eval_loss = paddle.zeros([1])
        mean_acc = paddle.zeros([1])
        val_loader = tqdm(val_loader)
        for step, data in enumerate(val_loader):
            pred = model(data)
            loss = loss_fun(pred, data['label'])
            eval_loss = (eval_loss * step + loss.detach()) / (step + 1)

            pred_acc = paddle.argmax(pred, 1)
            label = paddle.topk(data['label'], 1)[1]

            acc = accuracy_score(label.cpu().numpy(), pred_acc.cpu().numpy())
            mean_acc = (mean_acc * step + acc) / (step + 1)

            predictions.extend(pred_acc.cpu().numpy())
            labels.extend(label.cpu().numpy())

        F1 = getF1(predictions, labels)

        print(f'mena_loss:{eval_loss.item()}, mean_acc:{mean_acc.item()}, {F1}')

        state_dict = model.state_dict()
        paddle.save(state_dict, f"save/Epoch{i}_f1{F1['mean_f1']}.pdparams")
