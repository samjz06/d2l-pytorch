import torch
import torch.nn.functional
import random

x = torch.randn(2,3,4)
print(x)
ix = tuple([random.randrange(m) for m in x.shape])
print(ix)

x[ix].item()

x.new_empty(3,5)

# 
x = torch.arange(12).reshape(3, 4)

print(x)
x.stride() # original stride (4,1)
x[1, 3].item()

# * view:
x_view = x.view(4, 3)
x_view.view(-1)
x_view.stride() # (1,4) new meta info
x_view[2, 1].item()
# tensor.stride() is a part of meta information

# * transpose:
x_trans = x.transpose(0, 1)
x_trans.stride() # (1,4)
x_trans.view(-1) # errornous

# * permute:
x_perm = x.permute([1,0])
x.permute(0)
x_perm.stride()

# all these three pointed to the same location (same pointer())
x_view.data_ptr()
x_trans.data_ptr()
x_perm.data_ptr()

