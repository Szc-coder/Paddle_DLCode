import paddle
import paddle.nn as nn

from paddle import Tensor
from itertools import count
from typing import Callable
from loss import kl_loss, sym_kl_loss


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


def inf_norm(x):
    return paddle.norm(x, p=float('inf'), axis=-1, keepdim=True)


class SMARTLoss(nn.Layer):
    def __init__(
            self,
            eval_fn: Callable,
            loss_fn: Callable,
            loss_last_fn: Callable = None,
            norm_fn: Callable = inf_norm,
            num_steps: int = 1,
            step_size: float = 1e-3,
            epsilon: float = 1e-6,
            noise_var: float = 1e-5
    ) -> None:
        super(SMARTLoss, self).__init__()
        self.eval_fn = eval_fn
        self.loss_fn = loss_fn
        self.loss_last_fn = loss_last_fn
        self.norm_fn = norm_fn
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon
        self.noise_var = noise_var

    def forward(self, embedding: Tensor, state: Tensor) -> Tensor:
        emb_size = embedding.shape
        # 使用正态分布随机初始化扰动，结合 x 得到 x'
        noise = paddle.randn(emb_size) * self.noise_var
        noise.stop_gradient = False
        for i in count():
            embed_perturbed = embedding + noise
            state_perturbed = self.eval_fn(embed_perturbed)
            if i == self.num_steps:
                return self.loss_last_fn(state_perturbed, state)
            loss = self.loss_fn(state_perturbed, state.detach())

            noise_gradient, = paddle.autograd.grad(loss, noise)
            step = noise + self.step_size * noise_gradient
            step_norm = self.norm_fn(step)
            noise = step / (step_norm + self.epsilon)
            noise = noise.detach()
            noise.stop_gradient = False


class model(nn.Layer):
    def __init__(self):
        super(model, self).__init__()
        self.fc = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, inputs):
        def eval_fn(embedding):
            y = self.fc(embedding)
            y = self.fc2(y)
            return y

        state = eval_fn(inputs)
        smart = SMARTLoss(eval_fn=eval_fn, loss_fn=kl_loss, loss_last_fn=sym_kl_loss)

        smart_loss = smart(inputs, state)

        return smart_loss


m = model()
emb = paddle.randn([1, 256])
outs = m(emb)
outs.backward()
print(outs.grad)