import torch
import gpytorch
import numpy as np

class PGFKernel(gpytorch.kernels.Kernel):
    def __init__(self, width=[1], **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = len(width)
        for i in range(len(width)):
            for j in range(width[i]):
                self.register_parameter(
                    name='raw_prob'+str(j+1)+'_d'+str(i+1), parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
                )

    def forward(self, x1, x2, **params):
        def softplus(a: int):
            return np.log(1 + np.exp(a))
        probs = []
        sums = []
        for i in range(self.depth):
            probs.append([])
            for j in range(self.width[i]):
                raw_p = getattr(self, 'raw_prob'+str(j+1)+'_d'+str(i+1))
                p = softplus(raw_p.item())
                probs[i].append(p)
            sums.append(sum(probs[i]))
        diff = torch.FloatTensor(x1.size()[0], x2.size()[0])
        for i in range(x1.size()[0]):
            x = x1[i, :]
            x_norm = torch.linalg.norm(x)
            for j in range(x2.size()[0]):
                y = x2[j, :]
                y_norm = torch.linalg.norm(y)
                diff[i, j] = torch.dot(x, y) / (x_norm * y_norm)
        d = 0
        for x in range(self.depth):
            def f(s):
                total = 0
                for i, prob in enumerate(probs[d]):
                    total += prob * pow(s,i)
                return total/sums[d]
            diff.apply_(f)
            d += 1
        return diff