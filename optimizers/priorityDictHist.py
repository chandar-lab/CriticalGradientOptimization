from heapq import heapify, heappush, heappop
import torch
from copy import deepcopy


class Gradient:
    def __init__(self, g):
        self.g = g
        self.age = 0
        self.epoch_age = 0

    def step(self):
        self.age += 1
        self.epoch_age += 1

    def resetEpoch(self):
        self.epoch_age = 0


class HeapItem:
    def __init__(self, p, t):
        self.p = p
        self.t = t

    def __lt__(self, other):
        return self.p < other.p


class priority_dict(dict):
    """Dictionary that can be used as a priority queue.
    This class can be used in conjunction with SGD_C_HIST to keep a history of gradient ages to generate histograms.
    Reimplemented to avoid conflicts with other optimizers.
    """

    def __init__(self, *args, **kwargs):
        super(priority_dict, self).__init__(*args, **kwargs)
        self._heap = [HeapItem(k, v) for k, v in self.items()]
        self._rebuild_heap()

    def sethyper(self, decay_rate=0.5, K=5):
        self.k = K
        self.decay_rate = decay_rate

    def _reorder(self):
        # super(priority_dict, self).__init__(self._heap[-self.k:])
        self._heap = deepcopy(self._heap[-self.k:])
        in_heap = [it.p for it in self._heap]
        del_ = [k for k in self.keys() if k not in in_heap]
        for k in del_:
            del self[k]

    def _rebuild_heap(self):
        self._heap = [it for it in self._heap if it.p > 0.0]
        heapify(self._heap)
        if not self.isEmpty() and self.isFull():
            self._reorder()

    def isEmpty(self):
        if len(self._heap) == 0:
            return True
        return False

    def decay(self):
        self._heap = [HeapItem(self.decay_rate * it.p, it.t) for it in self._heap]

    def isFull(self):
        if len(self._heap) < self.k:
            return False
        return True

    def pokesmallest(self):
        """Return the lowest priority.
        Raises IndexError if the object is empty.
        """

        it = self._heap[0]
        return it.p

    def pokesmallest_age(self):
        """Return the lowest priority.
        Raises IndexError if the object is empty.
        """

        it = self._heap[0]
        return it.t.age

    def gradmean(self):
        """Return the sum of top k gradients
        """

        mean = torch.clone(self._heap[0].t.g)
        cnt = 1.
        for it in self._heap[1:]:
            mean.add_(it.t.g)
            cnt += 1.
        return mean.div_(cnt)

    def gradsum(self):
        """Return the sum of top k gradients
        """

        sum = torch.clone(self._heap[0].t.g)
        for it in self._heap[1:]:
            sum.add_(it.t.g)
        return sum

    def __getitem__(self, key):
        return dict(self._heap)

    def __len__(self):
        return len(self._heap)

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).

        # super(priority_dict, self).__setitem__(key, val)
        self._heap.append(HeapItem(key, Gradient(val)))
        self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.

        super(priority_dict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.
        Beware: this will destroy elements as they are returned.
        """

        while self:
            yield self.pop_smallest()

    def step(self):
        for item in self._heap: item.t.step()

    def epoch(self):
        ages = []
        for item in self._heap:
            ages.append(item.t.epoch_age)
            item.t.resetEpoch()
        return ages

    def averageTopC(self):
        ave = 0.
        if len(self._heap) > 0:
            ave = sum([it.t.g.norm() for it in self._heap]) / float(len(self._heap))
        return ave
