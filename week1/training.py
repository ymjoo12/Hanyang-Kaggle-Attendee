import os.path as op
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try: scriptpath = op.dirname(__file__)
except NameError: scriptpath = './'
filename = op.join(scriptpath,"weight-height.csv")
data = pd.read_csv(filename)

x, y =  data['Height'], data['Weight']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = torch.Tensor([[x] for x in list(x_train)])
y_train = torch.Tensor([[y] for y in list(y_train)])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1,1)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

savePath = op.join(scriptpath,"modelsv2.pth")
model = Model()
model.load_state_dict(torch.load(savePath))
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.000025)

# Training
for epoch in range(1500000):
    # Forward pass and loss
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    
    # Backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%1000==0:
        print(epoch+1, '%.4f'%loss.item())

# Plot
pred = model(x_train).detach().numpy()
plt.plot(x_train,y_train,'ro')
plt.plot(x_train,pred,'b')
plt.show()

# Saving model
print('save & new load model')
savePath = op.join(scriptpath,"modelsv3.pth")
torch.save(model.state_dict(), savePath)
newModel = Model()
newModel.load_state_dict(torch.load(savePath))

# Test
x_test = torch.Tensor([[x] for x in list(x_test)])
y_test = torch.Tensor([[y] for y in list(y_test)])
y_pred = model(x_test)
loss = criterion(y_pred,y_test)
print(loss)