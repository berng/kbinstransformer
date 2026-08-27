# KBinsTransformer 

monotonic transform to given interval [0,n_bins] using KBinsDiscretizer
and inverse transform of the result

## For simple test run 

python kbinstransformer.py

## To install to python:

pip install kbinstransformer

## Usage example

Init transformer:

from kbinstransformer import KBinsTransformer

kbt = KBinsTransformer(n_bins=BINS, strategy="quantile", encode="ordinal")

Fit parameters:

kbt.fit(X)

print(kbt.bin_edges_)

Monotonic transfrom X to [0..BINS] interval:

X_linear = kbt.transform(X)

Inverse transform:

X2 = kbt.inverse_transform(X_linear)

