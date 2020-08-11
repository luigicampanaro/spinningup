import torch
import torch.nn as nn
import numpy as np
import pdb
import random


class linearRegression(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.params = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(2, 1).float(), mean=0.0, std=0.1))

        self.A = torch.tensor([1, 0], requires_grad=False, dtype=torch.float32).view(1, 2)
        self.B = torch.tensor([0, 1], requires_grad=False, dtype=torch.float32).view(1, 2)

    def forward(self, obs):
        y = (self.A @ self.params) * obs + self.B @ self.params
        return y

def miniBatch(x_array, y_array, batch_size_):
    randomlist = random.sample(range(0, x_array.shape[0]), batch_size_)
    x_list = list()
    y_list = list()
    for i in range(batch_size_):
        x_list.append(x_array[randomlist[i], 0])
        y_list.append(y_array[randomlist[i], 0])
    return np.array(x_list).reshape(batch_size_, 1), np.array(y_list).reshape(batch_size_, 1)

# create dummy data for training
data_points = 10000
a, b = 2, 1
mu, sigma = 0, 0.1

# Crucial to scale the inputs between [0, 1] to avoid gradient explosion
x_values = [i / data_points for i in range(data_points)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [a*i + b + np.random.normal(mu, sigma) for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

# Settings and model instance
learningRate = 0.01
epochs = 5000   # Number of epochs way more important than batch_size
linearModel = linearRegression()

# Criterion and Optimizer
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(linearModel.parameters(), lr=learningRate)

# Mini-batches
batch_size = 4

# Main loop
for epoch in range(epochs):
    # Mini batches
    batchX, batchY = miniBatch(x_train, y_train, batch_size)
    batchX_torch, batchY_torch = torch.from_numpy(batchX).float(), torch.from_numpy(batchY).float()

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = linearModel(batchX_torch)

    print(outputs)
    pdb.set_trace()

    # get loss for the predicted output
    loss = criterion(outputs, batchY_torch)

    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()
    print(f'grads: {[par.grad for par in linearModel.parameters()]}')  # To check that the parameters are recognised as part of the model
print('\n\n################\n')
print(f'parameters: {[par.data for par in linearModel.parameters()]}\n')  # To check that the parameters are recognised as part of the model
print(f'grads: {[par.grad for par in linearModel.parameters()]}')  # To check that the parameters are recognised as part of the model
