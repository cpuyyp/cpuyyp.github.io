---
layout: post
title: Testing python code in Jekyll
comments: true
tags: NLP BoW
---
# Testing if python code shows properly in Jekyll

```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
train_data = np.loadtxt('Gisette/gisette_train.data').T
train_labels = np.loadtxt('Gisette/gisette_train.labels').T
test_data = np.loadtxt('Gisette/gisette_valid.data').T
test_labels = np.loadtxt('Gisette/gisette_valid.labels').T
```


```python
train_labels = train_labels>0
test_labels = test_labels>0

test_data = test_data - np.mean(train_data)
test_data = test_data / np.std(train_data)
train_data = train_data - np.mean(train_data)
train_data = train_data / np.std(train_data)
```


```python
def z(data, w):
    return  w[0] + np.dot(w[1:], data)
```


```python
def log_Loss(data, label, w):
    score = z(data, w)
    return np.dot(label, score) - np.sum( np.log(1 + np.exp(score)) )
```


```python
def classifier(data, w):
    return np.array([0.0 if i<=0 else 1.0 for i in z(data, w)])
```


```python
def one_step_opt(data, label, w, eta = 1, lambd = 0.001):
    M, N = np.shape(data)
    new_w = np.zeros_like(w)
    new_w[0] = w[0] - eta * lambd * w[0] + eta / N * np.sum((label - np.exp(z(data, w))/(1 + np.exp(z(data, w)))) )
    new_w[1:] = w[1:] - eta * lambd * w[1:] + eta / N * np.dot(data, (label - np.exp(z(data, w))/(1 + np.exp(z(data, w)))) )
    return new_w
```


```python
def optimize(train_data, train_labels, w, steps = 300, eta = 1, lambd = 0.001):
    for t in range(steps):
        w = one_step_opt(train_data, train_labels, w, eta, lambd)
    return w
```


```python
ETA = [1, 0.1, 0.01, 0.001, 0.0001]
accuracy = []
for eta in ETA:
    w = np.zeros(np.shape(train_data)[0] + 1)
    w = optimize(train_data, train_labels, w, eta = eta)
    clf_labels = classifier(test_data, w)
    accuracy.append(np.sum(clf_labels == test_labels)/len(test_labels))
```


```python
accuracy
```




    [0.971, 0.974, 0.976, 0.951, 0.911]




```python
plt.plot(np.linspace(1,len(accuracy),len(accuracy)),accuracy)
plt.show()
```
![png](images/output_10_1.png)
