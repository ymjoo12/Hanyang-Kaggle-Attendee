import os.path as op
import torch
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

try: scriptpath = op.dirname(__file__)
except NameError: scriptpath = './'

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1,1)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

class SubmittedApp:
    def __init__(self):
        savePath = op.join(scriptpath,"modelsv3.pth")
        self.model = Model()
        self.model.load_state_dict(torch.load(savePath))

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output_tensor = self.model(input_tensor)
        """Main Run Method for scoring system
        :param input_tensor: (torch.Tensor) [batchsize, height(1)]
        :return: (torch.Tensor) [batchsize, weight(1)]
        """
        return output_tensor

    def metric(self, inferred_tensor: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Mean Square Error
        l(y, y') = (y - y')^2        
        :param inferred_tensor: (torch.Tensor) [batch_size, weight(1)], inferred value
        :param ground_truth:  (torch.FloatTensor) [batch_size, weight(1)], ground truth value
        :return: (torch.Tensor) metric 점수
        """
        return torch.mean((inferred_tensor - ground_truth)**2)

filename = op.join(scriptpath,"weight-height.csv")
data = pd.read_csv(filename)
x, y =  data['Height'], data['Weight']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=75)
x_test = torch.Tensor([[x] for x in list(x_test)])
y_test = torch.Tensor([[y] for y in list(y_test)])

app = SubmittedApp()
y_pred = app.run(x_test)
print(y_pred)
print(app.metric(y_pred,y_test))