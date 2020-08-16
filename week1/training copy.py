import os.path as op
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try: scriptpath = op.dirname(__file__)
except NameError: scriptpath = './'
filename = op.join(scriptpath,"weight-height.csv")
data = pd.read_csv(filename)

x, y =  data['Height'], data['Weight']
# V = lambda v: Variable(torch.Tensor(v))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = torch.Tensor([[x] for x in list(x_train)])
y_train = torch.Tensor([[y] for y in list(y_train)])
# plt.plot(x_train,y_train)
# plt.show()
# x_test = Variable(torch.Tensor(x_test))
# y_test = Variable(torch.Tensor(y_test))

# print(x_train, x_data, y_train, y_data,sep='\n\n')
# print(x_train)

# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.linear = torch.nn.Linear(1,1)
    
#     def forward(self, x):
#         y_pred = self.linear(x)
#         return y_pred
    
model = torch.nn.Linear(1,1)
# criterion = torch.nn.MSELoss(size_average=False)
criterion = torch.nn.functional.mse_loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# y_p = model(x_train)
# ls = criterion(y_p,y_train)
# print(y_p,y_train,ls,sep='\n')

# Training
for epoch in range(50):
    y_pred = model(x_train)

    loss = criterion(y_pred, y_train)
    print(epoch, loss.data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test
x_test = Variable(torch.Tensor([[x] for x in list(x_test)]))
y_test = Variable(torch.Tensor([[y] for y in list(y_test)]))

print(scriptpath)