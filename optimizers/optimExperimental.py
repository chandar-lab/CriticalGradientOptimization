import math
import torch
from torch.optim import Optimizer
from .priorityDict import priority_dict
from copy import deepcopy

"""
Collection of Experimental optimizers developed during our research. Included for completeness.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def aggregate(d_p, crit_buf, func, kappa=1.0):
    if "sum" in func:
        crit_buf_ = crit_buf.gradmean()
        crit_buf_.mul_(kappa)
        return torch.add(d_p, crit_buf_)
    elif "mid" in func:
        crit_buf_ = crit_buf.gradmean()
        crit_buf_.mul_(kappa)
        return torch.mul(torch.add(d_p, crit_buf_), 0.5)
    elif "mean" in func:
        crit_buf_ = crit_buf.gradsum()
        crit_buf_.mul_(kappa)
        return torch.div(torch.add(d_p, crit_buf_), crit_buf.size() + 1)
    else:
        raise ValueError("Invalid aggregation function")


class SGD_FIFO(Optimizer):
    """
    Implementation of SGD (and optionally SGD with momentum) with critical gradients.
    Replaces current-iteration gradient in conventional PyTorch implementation with
    an aggregation of current gradient and critical gradients.

    Conventional SGD or SGD with momentum can be recovered by setting kappa=0.

    The critical-gradient-specific keyword parameters are tuned for good
    off-the-shelf performance, though additional tuning may be required for best results
    """

    def __init__(self, params, lr=0.001, kappa=1.0, dampening=0.,
                 weight_decay=0, momentum=0.,
                 decay=0.7, topC=10, aggr='sum', sampling=None, critical_test=True):

        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= decay and not 1.0 > decay:
            raise ValueError("Invalid alpha value: {}".format(decay))
        if not 0.0 <= topC:
            raise ValueError("Invalid alpha value: {}".format(topC))

        self._count = 0.0

        defaults = dict(lr=lr, kappa=kappa, dampening=dampening,
                        weight_decay=weight_decay, momentum=momentum,
                        aggr=aggr, decay=decay, gradHist={}, topC=topC, sampling=sampling, critical_test=critical_test)

        super(SGD_FIFO, self).__init__(params, defaults)
        self.resetOfflineStats()
        self.resetAnalysis()

    def getOfflineStats(self):
        return self.offline_grad

    def getAnalysis(self):
        return self.g_analysis

    def resetAnalysis(self):
        self.g_analysis = {'gt': 0., 'gc': 0., 'count': 0, 'gc_aggr': 0}

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def __setstate__(self, state):
        super(SGD_FIFO, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._count += 1
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            kappa = group['kappa']
            dampening = group['dampening']
            decay = group['decay']
            momentum = group['momentum']
            topc = group['topC']
            aggr = group['aggr']
            sampling = group['sampling']
            critical_test = group['critical_test']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p_norm = self._count
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                if kappa != 0:
                    param_state = self.state[p]
                    if 'critical gradients' not in param_state:
                        crit_buf = param_state['critical gradients'] = priority_dict()
                        crit_buf.sethyper(decay_rate=decay, K=topc, sampling=sampling)
                        crit_buf[d_p_norm] = deepcopy(d_p)
                    else:
                        crit_buf = param_state['critical gradients']
                        aggr_grad = aggregate(d_p, crit_buf, aggr, kappa)
                        if crit_buf.isFull():
                            if critical_test:
                                if d_p_norm > crit_buf.pokesmallest():
                                    self.offline_grad['yes'] += 1
                                    crit_buf[d_p_norm] = deepcopy(d_p)
                                else:
                                    self.offline_grad['no'] += 1
                            else:
                                self.offline_grad['yes'] += 1
                                crit_buf[d_p_norm] = deepcopy(d_p)
                        else:
                            crit_buf[d_p_norm] = deepcopy(d_p)
                        d_p = aggr_grad

                    self.g_analysis['gc'] += crit_buf.averageTopC()
                    self.g_analysis['count'] += 1
                    self.g_analysis['gt'] += p.grad.data.norm()
                    if 'mid' in aggr:
                        self.g_analysis['gc_aggr'] += crit_buf.getmin().norm()
                    elif 'median' in aggr:
                        self.g_analysis['gc_aggr'] += crit_buf.getmedian().norm()
                    elif 'max' in aggr:
                        self.g_analysis['gc_aggr'] += crit_buf.getmax().norm()
                    else:
                        self.g_analysis['gc_aggr'] += crit_buf.averageTopC()

                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        # if nesterov:
                        #     d_p = d_p.add(momentum, buf)
                        # else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

        return loss


class Adam_FIFO(Optimizer):
    """
    Implementation of Adam with critical gradients.
    Replaces current-iteration gradient in conventional PyTorch implementation with
    an aggregation of current gradient and critical gradients.

    Conventional Adam can be recovered by setting kappa=0.

    The critical-gradient-specific keyword parameters are tuned for good
    off-the-shelf performance, though additional tuning may be required for best results
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 decay=0.7, kappa=1.0, topC=10,
                 weight_decay=0, amsgrad=False, aggr='mean', sampling=None, critical_test=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= decay and not 1.0 > decay:
            raise ValueError("Invalid alpha value: {}".format(decay))
        if not 0.0 <= topC:
            raise ValueError("Invalid alpha value: {}".format(topC))

        self._count = 0.0

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, aggr=aggr, amsgrad=amsgrad,
                        kappa=kappa, topC=topC, decay=decay, sampling=sampling, critical_test=critical_test)

        super(Adam_FIFO, self).__init__(params, defaults)
        self.resetOfflineStats()
        self.resetAnalysis()

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def __setstate__(self, state):
        super(Adam_FIFO, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def getAnalysis(self):
        return self.g_analysis

    def resetAnalysis(self):
        self.g_analysis = {'gt': 0., 'gc': 0., 'count': 0, 'gc_aggr': 0}

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._count += 1

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                grad_norm = self._count
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                kappa = group['kappa']
                decay = group['decay']
                topc = group['topC']
                aggr = group['aggr']
                sampling = group['sampling']
                critical_test = group['critical_test']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)  # memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # memory_format=torch.preserve_format)
                    if kappa > 0.:
                        state['critical gradients'] = priority_dict()
                        state['critical gradients'].sethyper(decay_rate=decay, K=topc, sampling=sampling)
                        state['critical gradients'][grad_norm] = deepcopy(grad)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)  # memory_format=torch.preserve_format)
                else:
                    if kappa > 0.:
                        aggr_grad = aggregate(grad, state['critical gradients'], aggr)
                        if state['critical gradients'].isFull():
                            if critical_test:
                                if grad_norm > state['critical gradients'].pokesmallest():
                                    self.offline_grad['yes'] += 1
                                    state['critical gradients'][grad_norm] = deepcopy(grad)
                                else:
                                    self.offline_grad['no'] += 1
                            else:
                                self.offline_grad['yes'] += 1
                                state['critical gradients'][grad_norm] = deepcopy(grad)
                        else:
                            state['critical gradients'][grad_norm] = deepcopy(grad)
                    grad = aggr_grad

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                self.g_analysis['gc'] += state['critical gradients'].averageTopC()
                self.g_analysis['count'] += 1
                self.g_analysis['gt'] += p.grad.data.norm()
                if 'mid' in aggr:
                    self.g_analysis['gc_aggr'] += state['critical gradients'].getmin().norm()
                elif 'median' in aggr:
                    self.g_analysis['gc_aggr'] += state['critical gradients'].getmedian().norm()
                elif 'max' in aggr:
                    self.g_analysis['gc_aggr'] += state['critical gradients'].getmax().norm()
                else:
                    self.g_analysis['gc_aggr'] += state['critical gradients'].averageTopC()

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class RMSprop_FIFO(Optimizer):
    """
    Implementation of RMSprop with critical gradients.
    Replaces current-iteration gradient in conventional PyTorch implementation with
    an aggregation of current gradient and critical gradients.

    Conventional RMSprop can be recovered by setting kappa=0.

    The critical-gradient-specific keyword parameters are tuned for good
    off-the-shelf performance, though additional tuning may be required for best results
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
                 momentum=0, centered=False, decay=0.7, kappa=1.0,
                 topC=10, aggr='mean', sampling=None, critical_test=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= decay and not 1.0 > decay:
            raise ValueError("Invalid alpha value: {}".format(decay))
        if not 0.0 <= topC:
            raise ValueError("Invalid alpha value: {}".format(topC))

        self._count = 0.0

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps,
                        centered=centered, weight_decay=weight_decay,
                        aggr=aggr, kappa=kappa, topC=topC, decay=decay)
        super(RMSprop_FIFO, self).__init__(params, defaults)
        self.resetOfflineStats()

    def __setstate__(self, state):
        super(RMSprop_FIFO, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._count += 1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                grad_norm = self._count
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                kappa = group['kappa']
                decay = group['decay']
                topc = group['topC']
                aggr = group['aggr']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if kappa > 0.:
                        state['critical gradients'] = priority_dict()
                        state['critical gradients'].sethyper(decay_rate=decay, K=topc)
                        state['critical gradients'][grad_norm] = deepcopy(grad)
                else:
                    aggr_grad = aggregate(grad, state['critical gradients'], aggr)
                    if kappa > 0.:
                        if state['critical gradients'].isFull():
                            if grad_norm > state['critical gradients'].pokesmallest():
                                self.offline_grad['yes'] += 1
                                state['critical gradients'][grad_norm] = deepcopy(grad)
                            else:
                                self.offline_grad['no'] += 1
                        else:
                            state['critical gradients'][grad_norm] = deepcopy(grad)
                    grad = aggr_grad

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.add_(buf, alpha=-group['lr'])
                else:
                    p.addcdiv_(grad, avg, value=-group['lr'])

        return loss

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}


class SGD_C_bottom(Optimizer):
    """
    Implementation of SGD (and optionally SGD with momentum) with critical gradients.
    Replaces current-iteration gradient in conventional PyTorch implementation with
    an aggregation of current gradient and critical gradients.

    Conventional SGD or SGD with momentum can be recovered by setting kappa=0.

    The critical-gradient-specific keyword parameters are tuned for good
    off-the-shelf performance, though additional tuning may be required for best results
    """

    def __init__(self, params, lr=0.001, kappa=1.0, dampening=0.,
                 weight_decay=0, momentum=0.,
                 decay=0.7, topC=10, aggr='sum', sampling=None, critical_test=True):

        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= decay and not 1.0 > decay:
            raise ValueError("Invalid alpha value: {}".format(decay))
        if not 0.0 <= topC:
            raise ValueError("Invalid alpha value: {}".format(topC))

        defaults = dict(lr=lr, kappa=kappa, dampening=dampening,
                        weight_decay=weight_decay, momentum=momentum,
                        aggr=aggr, decay=decay, gradHist={}, topC=topC, sampling=sampling, critical_test=critical_test)

        super(SGD_C_bottom, self).__init__(params, defaults)
        self.resetOfflineStats()
        self.resetAnalysis()

    def getOfflineStats(self):
        return self.offline_grad

    def getAnalysis(self):
        return self.g_analysis

    def resetAnalysis(self):
        self.g_analysis = {'gt': 0., 'gc': 0., 'count': 0, 'gc_aggr': 0}

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def __setstate__(self, state):
        super(SGD_C_bottom, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        count = 0.0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            kappa = group['kappa']
            dampening = group['dampening']
            decay = group['decay']
            momentum = group['momentum']
            topc = group['topC']
            aggr = group['aggr']
            sampling = group['sampling']
            critical_test = group['critical_test']
            count += 0.

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p_norm = 1 / d_p.norm()
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                if kappa != 0:
                    param_state = self.state[p]
                    if 'critical gradients' not in param_state:
                        crit_buf = param_state['critical gradients'] = priority_dict()
                        crit_buf.sethyper(decay_rate=decay, K=topc, sampling=sampling)
                        crit_buf[d_p_norm] = deepcopy(d_p)
                    else:
                        crit_buf = param_state['critical gradients']
                        aggr_grad = aggregate(d_p, crit_buf, aggr, kappa)
                        if crit_buf.isFull():
                            if critical_test:
                                if d_p_norm > crit_buf.pokesmallest():
                                    self.offline_grad['yes'] += 1
                                    crit_buf[d_p_norm] = deepcopy(d_p)
                                else:
                                    self.offline_grad['no'] += 1
                            else:
                                self.offline_grad['yes'] += 1
                                crit_buf[d_p_norm] = deepcopy(d_p)
                        else:
                            crit_buf[d_p_norm] = deepcopy(d_p)
                        d_p = aggr_grad

                    self.g_analysis['gc'] += crit_buf.averageTopC()
                    self.g_analysis['count'] += 1
                    self.g_analysis['gt'] += p.grad.data.norm()
                    if 'mid' in aggr:
                        self.g_analysis['gc_aggr'] += crit_buf.getmin().norm()
                    elif 'median' in aggr:
                        self.g_analysis['gc_aggr'] += crit_buf.getmedian().norm()
                    elif 'max' in aggr:
                        self.g_analysis['gc_aggr'] += crit_buf.getmax().norm()
                    else:
                        self.g_analysis['gc_aggr'] += crit_buf.averageTopC()

                    crit_buf.decay()

                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        # if nesterov:
                        #     d_p = d_p.add(momentum, buf)
                        # else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

        return loss


class Adam_C_bottom(Optimizer):
    """
    Implementation of Adam with critical gradients.
    Replaces current-iteration gradient in conventional PyTorch implementation with
    an aggregation of current gradient and critical gradients.

    Conventional Adam can be recovered by setting kappa=0.

    The critical-gradient-specific keyword parameters are tuned for good
    off-the-shelf performance, though additional tuning may be required for best results
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 decay=0.7, kappa=1.0, topC=10,
                 weight_decay=0, amsgrad=False, aggr='mean'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= decay and not 1.0 > decay:
            raise ValueError("Invalid alpha value: {}".format(decay))
        if not 0.0 <= topC:
            raise ValueError("Invalid alpha value: {}".format(topC))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, aggr=aggr, amsgrad=amsgrad,
                        kappa=kappa, topC=topC, decay=decay)

        super(Adam_C_bottom, self).__init__(params, defaults)
        self.resetOfflineStats()
        self.resetAnalysis()

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def __setstate__(self, state):
        super(Adam_C_bottom, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def getAnalysis(self):
        return self.g_analysis

    def resetAnalysis(self):
        self.g_analysis = {'gt': 0., 'gc': 0., 'count': 0, 'gc_aggr': 0}

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                grad_norm = 1 / grad.norm()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                kappa = group['kappa']
                decay = group['decay']
                topc = group['topC']
                aggr = group['aggr']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)  # memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # memory_format=torch.preserve_format)
                    if kappa > 0.:
                        state['critical gradients'] = priority_dict()
                        state['critical gradients'].sethyper(decay_rate=decay, K=topc)
                        state['critical gradients'][grad_norm] = deepcopy(grad)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)  # memory_format=torch.preserve_format)
                else:
                    if kappa > 0.:
                        aggr_grad = aggregate(grad, state['critical gradients'], aggr)
                        if state['critical gradients'].isFull():
                            if grad_norm > state['critical gradients'].pokesmallest():
                                self.offline_grad['yes'] += 1
                                state['critical gradients'][grad_norm] = deepcopy(grad)
                            else:
                                self.offline_grad['no'] += 1
                        else:
                            state['critical gradients'][grad_norm] = deepcopy(grad)
                        grad = aggr_grad

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                self.g_analysis['gc'] += state['critical gradients'].averageTopC()
                self.g_analysis['count'] += 1
                self.g_analysis['gt'] += p.grad.data.norm()
                if 'mid' in aggr:
                    self.g_analysis['gc_aggr'] += state['critical gradients'].getmin().norm()
                elif 'median' in aggr:
                    self.g_analysis['gc_aggr'] += state['critical gradients'].getmedian().norm()
                elif 'max' in aggr:
                    self.g_analysis['gc_aggr'] += state['critical gradients'].getmax().norm()
                else:
                    self.g_analysis['gc_aggr'] += state['critical gradients'].averageTopC()

                state['critical gradients'].decay()

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class SAGA(Optimizer):
    """Implement the SAGA optimization algorithm"""

    def __init__(self, params, n_samples, lr=0.001):

        if n_samples <= 0:
            raise ValueError("Number of samples must be >0: {}".format(n_samples))

        self.n_samples = n_samples

        defaults = dict(lr=lr)

        super(SAGA, self).__init__(params, defaults)
        self.resetOfflineStats()

    def __setstate__(self, state):
        super(SAGA, self).__setstate__(state)

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def step(self, index, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        if index < 0.0:
            raise ValueError("Invalid index value: {}".format(index))
        loss = None
        if closure is not None:
            loss = closure()

        n = self.n_samples

        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:
                    continue
                d_p = p.grad.data

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                param_state = self.state[p]
                if 'gradient_buffer' not in param_state:
                    buf = param_state['gradient_buffer'] = torch.zeros(n, *list(d_p.shape))
                else:
                    buf = param_state['gradient_buffer']

                saga_term = torch.mean(buf, dim=0).to(device)  # hold mean and last gradient in saga_term

                g_i = torch.clone(buf[index]).detach().to(device)

                saga_term.sub_(g_i)

                buf[index] = torch.clone(d_p).detach()

                d_p.sub_(saga_term)

                p.data.add_(d_p, alpha=-group['lr'])

        return loss


class SGD_new_momentum(Optimizer):

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_new_momentum, self).__init__(params, defaults)
        self.resetOfflineStats()

    def __setstate__(self, state):
        super(SGD_new_momentum, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        n = param_state['buffer_size'] = 1
                    else:
                        buf = param_state['momentum_buffer']
                        n = param_state['buffer_size']
                        n += 1
                        buf.add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = torch.clone(buf).detach()
                        d_p.div_(n)

                p.data.add_(d_p, alpha=-group['lr'])

        return loss


class SGD_C_double(Optimizer):
    r"""Implements SGD (optionally with momentum) while keeping a record of critical
    gradients (top C gradients by norm). Adds the sum or mean of these gradients
    to the final update step such that for param p

    p(t+1) = p(t) + lr * (g_t + f(g_crit))

    Where f is either a sum or mean of the gradients in g_crit
    """

    def __init__(self, params, lr=0.001, kappa=1.0, dampening=0.,
                 weight_decay=0, momentum=0., decay=0.99, nesterov=False, topC=10, sum='sum'):

        defaults = dict(lr=lr, kappa=kappa, dampening=dampening,
                        weight_decay=weight_decay, momentum=momentum, sum=sum, decay=decay, nesterov=nesterov,
                        gradHist={}, topC=topC)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_C_double, self).__init__(params, defaults)
        self.resetOfflineStats()

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def __setstate__(self, state):
        super(SGD_C_double, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            kappa = group['kappa']
            dampening = group['dampening']
            decay = group['decay']
            momentum = group['momentum']
            #    nesterov = group['nesterov']
            topc = group['topC']
            sum = group['sum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p_norm = d_p.norm()
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                if kappa != 0:
                    param_state = self.state[p]
                    if 'critical gradients' not in param_state:
                        crit_buf = param_state['critical gradients'] = priority_dict()
                        crit_buf.sethyper(decay_rate=decay, K=topc)
                        crit_buf[d_p_norm] = deepcopy(d_p)
                    else:
                        crit_buf = param_state['critical gradients']
                        if crit_buf.isFull():
                            if d_p_norm > crit_buf.pokesmallest():
                                self.offline_grad['yes'] += 1
                                crit_buf[d_p_norm] = deepcopy(d_p)
                            else:
                                self.offline_grad['no'] += 1
                        else:
                            crit_buf[d_p_norm] = deepcopy(d_p)
                    # Critical Gradients
                    # x_new = x_old - lr * grad
                    # x_new = x_old - lr * (momentum * grad_<t + (1-dampening) * grad_t)
                    # understand how the projection happens col_space or row_space
                    # CG method:
                    # x_new = x_old - lr * (momentum * grad_CG + (1-dampening) * grad_t)
                    # grad_CG <- topk gradients

                    d_p = aggregate(d_p, crit_buf, sum, kappa)
                    crit_buf.decay()

                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        # if nesterov:
                        #     d_p = d_p.add(momentum, buf)
                        # else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

        return loss


class SGD_C_Only(Optimizer):
    r"""Implements SGD (optionally with momentum) while keeping a record of critical
    gradients (top C gradients by norm). Replaces the gradient in conventional
    SGD with either the sum or the mean of critical gradients

    """

    def __init__(self, params, lr=0.001, kappa=1.0, dampening=0.,
                 weight_decay=0, momentum=0., decay=0.99, nesterov=False, topC=10, sum='sum'):

        defaults = dict(lr=lr, kappa=kappa, dampening=dampening,
                        weight_decay=weight_decay, momentum=momentum, sum=sum, decay=decay, nesterov=nesterov,
                        gradHist={}, topC=topC)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_C_Only, self).__init__(params, defaults)
        self.resetOfflineStats()

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def __setstate__(self, state):
        super(SGD_C_Only, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            kappa = group['kappa']
            dampening = group['dampening']
            decay = group['decay']
            momentum = group['momentum']
            #    nesterov = group['nesterov']
            topc = group['topC']
            sum = group['sum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p_norm = d_p.norm()
                crit_buf_ = None
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                if kappa != 0:
                    param_state = self.state[p]
                    if 'critical gradients' not in param_state:
                        crit_buf = param_state['critical gradients'] = priority_dict()
                        crit_buf.sethyper(decay_rate=decay, K=topc)
                        crit_buf[d_p_norm] = deepcopy(d_p)
                    else:
                        crit_buf = param_state['critical gradients']
                        if crit_buf.isFull():
                            if d_p_norm > crit_buf.pokesmallest():
                                self.offline_grad['yes'] += 1
                                crit_buf[d_p_norm] = deepcopy(d_p)
                            else:
                                self.offline_grad['no'] += 1
                        else:
                            crit_buf[d_p_norm] = deepcopy(d_p)
                    # Critical Gradients
                    # x_new = x_old - lr * grad
                    # x_new = x_old - lr * (momentum * grad_<t + (1-dampening) * grad_t)
                    # understand how the projection happens col_space or row_space
                    # CG method:
                    # x_new = x_old - lr * (momentum * grad_CG + (1-dampening) * grad_t)
                    # grad_CG <- topk gradients
                    if 'sum' in sum:
                        crit_buf_ = crit_buf.gradsum()
                    else:
                        crit_buf_ = crit_buf.gradmean()
                    crit_buf_.mul_(kappa)
                    crit_buf.decay()
                    d_p = crit_buf_
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        # if nesterov:
                        #     d_p = d_p.add(momentum, buf)
                        # else:
                        d_p = buf
                # if nesterov:
                #     d_p = d_p.add(momentum, buf)
                # else:
                #     d_p = crit_buf_
                # if kappa != 0:
                #     p.data.add_(-group['kappa'],crit_buf_)

                p.data.add_(d_p, alpha=-group['lr'])

        return loss


class Adam_C_double(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, decay=0.95, kappa=1.0, topC=10,
                 weight_decay=0, amsgrad=False, sum='sum', param_level=True):  # decay=0.9
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, sum=sum, amsgrad=amsgrad, kappa=kappa, topC=topC, decay=decay)
        super(Adam_C_double, self).__init__(params, defaults)
        self.resetOfflineStats()

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def __setstate__(self, state):
        super(Adam_C_double, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                grad_norm = grad.norm()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                kappa = group['kappa']
                decay = group['decay']
                topc = group['topC']
                sum = group['sum']
                param_level = group['param_level']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)  # , memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # , memory_format=torch.preserve_format)
                    if kappa > 0.:
                        state['critical gradients'] = priority_dict()
                        state['critical gradients'].sethyper(decay_rate=decay, K=topc)
                        state['critical gradients'][grad_norm] = deepcopy(grad)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)  # , memory_format=torch.preserve_format)
                else:
                    if kappa > 0.:
                        if state['critical gradients'].isFull():
                            if grad_norm > state['critical gradients'].pokesmallest():
                                self.offline_grad['yes'] += 1
                                state['critical gradients'][grad_norm] = deepcopy(grad)
                            else:
                                self.offline_grad['no'] += 1
                        else:
                            state['critical gradients'][grad_norm] = deepcopy(grad)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                if kappa > 0. and not param_level:
                    grad = aggregate(grad, state['critical gradients'], sum)
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                state['critical gradients'].decay()

                if param_level:
                    exp_avg = aggregate(exp_avg, state['critical gradients'], sum)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class RMSprop_C_double(Optimizer):
    r"""Implements RMSprop algorithm.
    Proposed by G. Hinton in his
    `course <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.
    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.
    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False, decay=0.95,
                 kappa=1.0, topC=10, sum='sum'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay,
                        sum=sum, kappa=kappa, topC=topC, decay=decay)
        super(RMSprop_C_double, self).__init__(params, defaults)
        self.resetOfflineStats()

    def __setstate__(self, state):
        super(RMSprop_C_double, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                grad_norm = grad.norm()
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                kappa = group['kappa']
                decay = group['decay']
                topc = group['topC']
                sum = group['sum']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if kappa > 0.:
                        state['critical gradients'] = priority_dict()
                        state['critical gradients'].sethyper(decay_rate=decay, K=topc)
                        state['critical gradients'][grad_norm] = deepcopy(grad)
                else:
                    if kappa > 0.:
                        if state['critical gradients'].isFull():
                            if grad_norm > state['critical gradients'].pokesmallest():
                                self.offline_grad['yes'] += 1
                                state['critical gradients'][grad_norm] = deepcopy(grad)
                            else:
                                self.offline_grad['no'] += 1
                        else:
                            state['critical gradients'][grad_norm] = deepcopy(grad)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if kappa > 0.:
                    grad = aggregate(grad, state['critical gradients'], sum)

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                state['critical gradients'].decay()

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.add_(buf, alpha=-group['lr'])
                else:
                    p.addcdiv_(grad, avg, value=-group['lr'])

        return loss

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}


class AggMo(Optimizer):

    def __init__(self, params, lr=0.001, momenta=[], dampening=0,
                 weight_decay=0):
        if any(momentum < 0.0 for momentum in momenta):
            raise ValueError("Invalid momentum value: at least one value is negative")
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momenta=torch.tensor(momenta).to(device), dampening=dampening,
                        weight_decay=weight_decay)
        super(AggMo, self).__init__(params, defaults)
        self.resetOfflineStats()

    def __setstate__(self, state):
        super(AggMo, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momenta = group['momenta']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                if len(momenta) != 0 and all(momentum != 0.0 for momentum in momenta):
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.stack([torch.clone(d_p).detach()]*len(momenta))
                        vec = param_state['momentum'] = torch.clone(momenta)
                        while vec.dim() < buf.dim(): vec.unsqueeze_(1)
                    else:
                        buf = param_state['momentum_buffer']
                        vec = param_state['momentum']
                        buf.mul_(vec)
                        buf.add_(d_p, alpha=1 - dampening)

                    d_p = torch.mean(buf, dim = 0)

                p.data.add_(d_p, alpha=-group['lr'])

        return loss

class SGD_C_new(Optimizer):
    """
    Implementation of SGD (and optionally SGD with momentum) with critical gradients.
    Replaces current-iteration gradient in conventional PyTorch implementation with
    an aggregation of current gradient and critical gradients.
    Conventional SGD or SGD with momentum can be recovered by setting kappa=0.
    The critical-gradient-specific keyword parameters are tuned for good
    off-the-shelf performance, though additional tuning may be required for best results
    """

    def __init__(self, params, lr=0.001, kappa=1.0, dampening=0.,
                 weight_decay=0, momentum=0.,
                 decay=0.7, topC=10, aggr='sum'):

        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= decay and not 1.0 > decay:
            raise ValueError("Invalid alpha value: {}".format(decay))
        if not 0.0 <= topC:
            raise ValueError("Invalid alpha value: {}".format(topC))

        defaults = dict(lr=lr, kappa=kappa, dampening=dampening,
                        weight_decay=weight_decay, momentum=momentum,
                        aggr=aggr, decay=decay, gradHist={}, topC=topC)

        super(SGD_C_new, self).__init__(params, defaults)
        self.resetOfflineStats()
        self.resetAnalysis()

    def getOfflineStats(self):
        return self.offline_grad

    def getAnalysis(self):
        return self.g_analysis

    def resetAnalysis(self):
        self.g_analysis = {'gt': 0., 'gc': 0., 'count': 0}

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def __setstate__(self, state):
        super(SGD_C_new, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            kappa = group['kappa']
            dampening = group['dampening']
            decay = group['decay']
            momentum = group['momentum']
            topc = group['topC']
            aggr = group['aggr']

            total_norm = 0.0

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                total_norm += torch.sqrt(torch.sum(torch.square(d_p)))

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                if kappa != 0:
                    param_state = self.state[p]
                    if 'critical gradients' not in param_state:
                        crit_buf = param_state['critical gradients'] = priority_dict()
                        crit_buf.sethyper(decay_rate=decay, K=topc)
                        crit_buf[total_norm] = deepcopy(d_p)
                    else:
                        crit_buf = param_state['critical gradients']
                        #
                        aggr_grad = aggregate(d_p, crit_buf, aggr, kappa)
                        if crit_buf.isFull():
                            if total_norm > crit_buf.pokesmallest():
                                self.offline_grad['yes'] += 1
                                crit_buf[total_norm] = deepcopy(d_p)
                            else:
                                self.offline_grad['no'] += 1
                        else:
                            crit_buf[total_norm] = deepcopy(d_p)
                        d_p = aggr_grad

                    self.g_analysis['gc'] += crit_buf.averageTopC()
                    self.g_analysis['count'] += 1
                    self.g_analysis['gt'] += p.grad.data.norm()

                    crit_buf.decay()

                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        # if nesterov:
                        #     d_p = d_p.add(momentum, buf)
                        # else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

        return loss

