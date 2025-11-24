##just to test new stuff i didnt know before



## testing unsqueeze 

import torch
a = torch.tensor([1,2,3])
print(a.shape)  # torch.Size([3])
print(a)
b = a.unsqueeze(0)
print(b.shape)  # torch.Size([1, 3])
print(b)
c = a.unsqueeze(1)
print(c.shape)  # torch.Size([3, 1])
print(c)
d = a.unsqueeze(-1)
print(d.shape)  # torch.Size([3, 1])
print(d)
