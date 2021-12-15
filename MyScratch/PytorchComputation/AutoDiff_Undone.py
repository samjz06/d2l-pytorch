import torch
from torch._C import dtype, float64
from torch.autograd import Variable

# rmq: 没太搞懂
## Head derivative and Chain rule

x = Variable(torch.arange(4,dtype=torch.float64).reshape(4,1), requires_grad=True)
y = x*2
z = y*x

print(x)
# compute derivative with backward method: dy/dx = [2,2,2,2]
# torch.autograd.backward()
y.backward(torch.ones_like(x, dtype=torch.float64))
x.grad # this should be dy/dx
# dz/dx = 4*x
z.backward() # this cannot be executed after y.backward() ???
x.grad

# instead
x = Variable(torch.arange(4,dtype=torch.float64).reshape(4,1), requires_grad=True)
y = x*2
z = y*x

print(x)
# compute derivative with backward method: dy/dx = [2,2,2,2]
# torch.autograd.backward()
y.backward(torch.ones_like(x, dtype=torch.float64),
           retain_graph=True)
x.grad # this should be dy/dx

z.backward(torch.ones_like(y))
x.grad


head_grad = torch.tensor([[10], [1.], [.1], [.01]])
z.backward(head_grad)
x.grad