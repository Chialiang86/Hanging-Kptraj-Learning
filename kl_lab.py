import numpy as np
import matplotlib.pyplot as plt

class kl_annealing():
    def __init__(self, kl_anneal_cyclical=True, niter=500, start=0.0, stop=1.0, kl_anneal_cycle=10, kl_anneal_ratio=1):
        super().__init__()
        if kl_anneal_cyclical:
            self.L = self.frange_cycle_linear(niter, start, stop, kl_anneal_cycle, kl_anneal_ratio)
        else:
            self.L = self.frange_cycle_linear(niter, start, stop, 1, kl_anneal_ratio)
        self.index = 0

    def frange_cycle_linear(self, n_iter, start=0.0001, stop=0.01, n_cycle=10, ratio=0.5):
        L = np.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio) # linear schedule
        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L 
    
    def update(self):
        self.index = (self.index + 1) % (len(self.L) - 1)

    def get_beta(self):
        return self.L[self.index]

# kla = kl_annealing(kl_anneal_cyclical=True, niter=200, start=0, stop=0.1, kl_anneal_cycle=5, kl_anneal_ratio=0.5)
kla = kl_annealing(kl_anneal_cyclical=True, 
                niter=200, 
                start=0.0, stop=0.1, kl_anneal_cycle=5, kl_anneal_ratio=0.5)
plt.figure(figsize=(12,4))
plt.plot(kla.L, label='KL loss weight')
plt.title('KL Annealing (weight of KL loss)')
plt.xlabel('epoch')
plt.ylabel('weight')
plt.grid()
plt.savefig('kl_annealing_single.png')