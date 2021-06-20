import random
from copy import deepcopy
from heapq import heapify

import torch

random.seed(100)


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


class priorityDict(dict):
    """Dictionary that can be used as a priority queue.

    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'

    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.

    The 'sorted_iter' method provides a destructive sorted iterator.

    This class can be used in conjunction with SGD_C_HIST to keep a history of gradient ages to generate histograms.
    Reimplemented to avoid conflicts with other optimizers by setting hist=true.
    """

    def __init__(self, *args, **kwargs):
        super(priorityDict, self).__init__(*args, **kwargs)
        self._heap = [HeapItem(k, v) for k, v in self.items()]
        self._rebuild_heap()

    def getNorms(self):
        return [item.p for item in self._heap]

    def size(self):
        return len(self._heap)

    def setHyper(self, decay_rate=0.5, K=5, sampling='KotH', hist=False):
        self.k = K
        self.decay_rate = decay_rate
        self.sampling = sampling
        self.hist = hist

    def _reorder(self):
        self._heap = deepcopy(self._heap[-self.k:])
        in_heap = [it.p for it in self._heap]
        del_ = [k for k in self.keys() if k not in in_heap]
        for k in del_:
            del self[k]

    def _rebuild_heap(self):
        self._heap = [it for it in self._heap if
                      it.p >= 0.0]  # >= used as fix for errors in some data
        if len(self._heap) > 0:
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

    def averageTopC(self):
        ave = 0.
        if len(self._heap) > 0:
            if not self.hist:
                ave = sum([it.t.norm() for it in self._heap]) / float(len(self._heap))
            else:
                ave = sum([it.t.g.norm() for it in self._heap]) / float(len(self._heap))
        return ave

    def pokeSmallest(self):
        """Return the lowest priority.
        Raises IndexError if the object is empty.
        """

        it = self._heap[0]
        return it.p

    def pokeSmallestAge(self):
        """Return the lowest priority.
        Raises IndexError if the object is empty.
        """

        it = self._heap[0]
        return it.t.age

    def getMin(self):
        """
        Get smallest gradient
        :return: The smallest gradient
        """
        return self._heap[0].t

    def getMax(self):
        "Returns the largest gradient"
        max = self._heap[0]
        for it in self._heap[1:]:
            if max.p < it.p:
                max = it
        return max.t

    def getMedian(self):
        "Returns the median gradient"
        sorted_list = sorted(self._heap, key=lambda x: x.p)
        return sorted_list[len(self._heap) // 2].t

    def gradMean(self):
        """Return the mean of top k gradients
        """
        if not self.hist:
            mean = torch.clone(self._heap[0].t)
            cnt = 1.
            for it in self._heap[1:]:
                mean.add_(it.t)
                cnt += 1.
        else:
            mean = torch.clone(self._heap[0].t.g)
            cnt = 1.
            for it in self._heap[1:]:
                mean.add_(it.t.g)
                cnt += 1.
        return mean.div_(cnt)

    def gradSum(self):
        """Return the sum of top k gradients
        """
        if not self.hist:
            sum = torch.clone(self._heap[0].t)
            for it in self._heap[1:]:
                sum.add_(it.t)
        else:
            sum = torch.clone(self._heap[0].t.g)
            for it in self._heap[1:]:
                sum.add_(it.t.g)
        return sum

    def __getitem__(self, key):
        return dict(self._heap)

    def __len__(self):
        return len(self._heap)

    def __setitem__(self, key, val):
        if self.isFull():
            if self.sampling == "pure_random":
                if random.uniform(0, 1) < 0.5:
                    item = random.choices(self._heap)[0]
                    self._heap.remove(item)
                    self._heap.append(HeapItem(key, val))
            if self.sampling == "random":
                item = random.choices(self._heap)[0]
                self._heap.remove(item)
                self._heap.append(HeapItem(key, val))
            elif self.sampling == "proportional":
                norm_sum = sum([it.t.norm() for it in self._heap])
                weights = [1 - (it.t.norm() / norm_sum) for it in self._heap]
                item = random.choices(self._heap, weights=weights)[0]
                self._heap.remove(item)
                self._heap.append(HeapItem(key, val))
            elif self.sampling == "mean_diversity":
                av = self.gradMean()
                prob = min(1., (torch.sub(av, val).norm()) / av.norm())
                if random.uniform(0, 1) <= prob:
                    self._heap.append(HeapItem(key, val))
            elif self.sampling == "cos_diversity":
                av = torch.flatten(self.gradMean())
                flat = torch.flatten(val)
                prob = (torch.dot(flat, av) / (flat.norm() * av.norm()) - 1.) * -0.5
                if random.uniform(0, 1) <= prob:
                    self._heap.append(HeapItem(key, val))
            elif self.sampling == "cos_diversity_keep_backwards":
                av = self.gradMean()
                prob = min(1., -(torch.dot(val, av) / (val.norm() * av.norm()) - 1.))
                if random.uniform(0, 1) <= prob:
                    self._heap.append(HeapItem(key, val))
            elif self.sampling == "cos_similarity":
                av = self.gradMean()
                av = torch.flatten(av)
                prob = 0.5 * (torch.dot(torch.flatten(val), av) / (
                        val.norm() * av.norm()) + 1)
                if random.uniform(0, 1) <= prob:
                    self._heap.append(HeapItem(key, val))
            else:
                self._heap.append(HeapItem(key, val))
        else:
            if self.sampling == "pure-random" and not self.isEmpty():
                if random.uniform(0, 1) < 0.5:
                    self._heap.append(HeapItem(key, val))
            else:
                self._heap.append(HeapItem(key, val))

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

        super(priorityDict, self).update(*args, **kwargs)
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
