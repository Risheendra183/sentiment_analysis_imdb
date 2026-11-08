import numpy as np

input=[1,2,3]

h=0

Wxh=0.5
Whh=0.5
bh=0

for x in input:
    h=np.tanh(Wxh*x+Whh*h+bh)

print("input:",x)
print("hidden states",h)