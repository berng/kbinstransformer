## @package kbinstransformer_v1.0
# Made under GNU GPL v3.0
# by Oleg I.Berngardt, 2026
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as 
# published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. 
# If not, see <https://www.gnu.org/licenses/>. 

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

class KBinsTransformer():
 def __init__(self,n_bins=10, strategy="quantile", encode="ordinal",subsample=None):
  self.kbd=KBinsDiscretizer(n_bins=n_bins, strategy=strategy, encode=encode,subsample=subsample)
  self.n_bins=n_bins
  self.strategy=strategy
  self.encode=encode
  self.subsample=subsample

 def fit(self,X):
  self.kbd.fit(X)
  self.bin_edges_=self.kbd.bin_edges_
  self.n_bins_=self.kbd.n_bins_
  self.left=X.min(axis=0)
  self.right=X.max(axis=0)
#  print('limits',self.left,self.right)
#  print('edges',self.bin_edges_)

 def transform(self,X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    out = np.empty(X.shape, dtype=float)
    for j, edges in enumerate(self.bin_edges_):
        left=self.left[j]
        out[:, j] = np.interp(
            X[:, j],
            edges,
            np.arange(edges.size, dtype=float),
            left=self.left[j],
            right=self.right[j],
        )
    return out

 def inverse_transform(self,U):
    U = np.asarray(U, dtype=float)

    if U.ndim == 1:
        U = U.reshape(-1, 1)

    out = np.empty(U.shape, dtype=float)

    for j, edges in enumerate(self.bin_edges_):
        out[:, j] = np.interp(
            U[:, j],
            np.arange(edges.size, dtype=float),
            edges
        )

    return out
 def fit_transform(self,X):
  self.fit(X)
  return self.transform(X)


if __name__=='__main__':
 import matplotlib.pyplot as plt
 BINS=10
 X=np.random.rand(10000,2)*3.14/2
 X[:,0]=np.tan(X[:,0])
 X[:,1]=np.sin(X[:,1]*2-3.14/2)
 kbt = KBinsTransformer(n_bins=BINS, strategy="quantile", encode="ordinal")
 kbt.fit(X)
 print(kbt.bin_edges_)
 X_linear = kbt.transform(X)
 X2 = kbt.inverse_transform(X_linear)

 plt.subplot(3,2,1)
 plt.hist(X,density=True,bins=BINS)
 plt.yscale('log')
 plt.subplot(3,2,2)
 plt.hist(X_linear,density=True,bins=BINS)
 plt.subplot(3,2,3)
 plt.scatter(X[:,0],X_linear[:,0])
 plt.xscale('log')
 plt.subplot(3,2,4)
 plt.scatter(X[:,0],X2[:,0])

 plt.subplot(3,2,5)
 plt.scatter(X[:,1],X_linear[:,1])
# plt.xscale('log')
 plt.subplot(3,2,6)
 plt.scatter(X[:,1],X2[:,1])
 plt.show()
 print(np.concatenate((X,X_linear),axis=-1))

