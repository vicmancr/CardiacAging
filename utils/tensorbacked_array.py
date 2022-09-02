'''
This is a custom tensor-based array from
https://github.com/pytorch/pytorch/issues/13246#issuecomment-684831789
to solve the issue with increase shared memory usage summarized in
https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
This "memory leak" is actually the copy-on-acces problem of forked
python processes due to changing refcounts, not a memory leak.
'''
import torch


class TensorBackedImmutableStringArray:
    def __init__(self, strings, encoding = 'utf-8'):
        encoded = [torch.ByteTensor(torch.ByteStorage.from_buffer(s.encode(encoding))) for s in strings]
        self.cumlen = torch.cat((torch.zeros(1, dtype = torch.int64), torch.as_tensor(list(map(len, encoded)), dtype = torch.int64).cumsum(dim = 0)))
        self.data = torch.cat(encoded)
        self.encoding = encoding

    def __getitem__(self, i):
        return bytes(self.data[self.cumlen[i] : self.cumlen[i + 1]]).decode(self.encoding)

    def __len__(self):
        return len(self.cumlen) - 1

    def __list__(self):
        return [self[i] for i in range(len(self))]


class TensorBackedImmutableNestedArray:
    def __init__(self, array):
        self.len, self.n_arrays = array.shape
        self.data = []
        # self.data.append(TensorBackedImmutableStringArray(array[0]))
        for j in range(self.n_arrays):
            self.data.append(torch.FloatTensor(array[:,j].flatten()))

    def __getitem__(self, i):
        return list(map(lambda j: self.data[j][i], range(self.n_arrays)))

    def __len__(self):
        return self.len

    def __list__(self):
        return [self[i] for i in range(len(self))]


if __name__ == '__main__':

    a = TensorBackedImmutableStringArray(['asd', 'def'])
    print('len = ', len(a))
    print('data = ', list(a))
