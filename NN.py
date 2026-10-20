import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#As I told you in class, you're looking for regression! A basic MLP can do this. Use one output neuron per value you want to predict, with no activation on the output layer. You can train with MSE and evaluate using RMSE.


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = X_train.shape[1]
hidden_Size = 256
num_outputs = 1
model = MLP(input_size, hidden_Size, num_outputs)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


batch_size = 16
train_dataset = TensorDataset(X_train,Y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)





