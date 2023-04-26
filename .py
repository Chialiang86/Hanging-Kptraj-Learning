
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA

# import torch
# import torch.nn as nn

from tqdm import tqdm
import time

# # offset = np.zeros(12)
# # offset1 = np.copy(offset)
# # offset2 = np.copy(offset)
# # offset1[0] += 0.5
# # offset2[1] += 0.5
# # X1 = np.random.rand(1000, 12) + offset1
# # X2 = np.random.rand(1000, 12) + offset2

# # X = np.vstack((X1, X2))

# # X_embedded = TSNE(n_components=2, learning_rate='auto',
# #                   init='random', perplexity=3).fit_transform(X)
# # # X_embedded = PCA(n_components=2).fit_transform(X)
# # print(X_embedded.shape)

# # plt.scatter(X_embedded[:1000, 0], X_embedded[:1000, 1], label='1')
# # plt.scatter(X_embedded[1000:, 0], X_embedded[1000:, 1], label='2')
# # plt.legend()
# # plt.show()

# mlp = nn.Sequential(
#             nn.Linear(9, 32),
#             nn.LeakyReLU(),
#             nn.Linear(32, 64),
#             nn.LeakyReLU(),
#             nn.Linear(64, 64)
#         )

# x = torch.randn(64, 40, 9)
# out = mlp(x)
# print(out.shape)


for _ in range(100000):
    for i in tqdm(range(1000)):
        time.sleep(0.001)

