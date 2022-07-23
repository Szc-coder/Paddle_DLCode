import paddle.nn.functional as F


# F.kl_div: (reduction) the candicates are ``'none'`` | ``'batchmean'`` | ``'mean'`` | ``'sum'``.
def kl_loss(inputs, target, reduction='batchmean'):
    return F.kl_div(
        F.log_softmax(inputs, axis=-1),
        F.softmax(inputs, axis=-1),
        reduction=reduction,
    )


def sym_kl_loss(inputs, target, reduction='sum', alpha=1.0):
    return alpha * F.kl_div(
        F.log_softmax(inputs, axis=-1),
        F.softmax(target.detach(), axis=-1),
        reduction=reduction,
    ) + F.kl_div(
        F.log_softmax(target, axis=-1),
        F.softmax(inputs.detach(), axis=-1),
        reduction=reduction,
    )


def js_loss(inputs, target, reduction='sum', alpha=1.0):
    mean_proba = 0.5 * (F.softmax(inputs.detach(), axis=-1) + F.softmax(target.detach(), axis=-1))
    return alpha * (F.kl_div(
        F.log_softmax(inputs, axis=-1),
        mean_proba,
        reduction=reduction
    ) + F.kl_div(
        F.log_softmax(target, axis=-1),
        mean_proba,
        reduction=reduction
    ))
