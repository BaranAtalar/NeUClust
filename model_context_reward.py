import numpy as np
import torch
from torch._C import device
from torch.autograd import Variable


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(in_features=inputSize, out_features=outputSize)
        # torch.nn.init.zeros_(self.linear.weight)
        # torch.nn.init.ones_(self.linear.weight)
        # self.relu = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(in_features=5, out_features=outputSize)

    def forward(self, x):
        out = self.linear(x)
        # out = self.relu(out)
        # out = self.linear2(out)
        return out

    
def train(model, inputDim, outputDim, learningRate, epochs, x_train, y_train):
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(x_train).cuda())
#             print(inputs.shape)
            labels = Variable(torch.from_numpy(y_train).cuda())
        else:
            inputs = Variable(torch.from_numpy(x_train))
#             print(inputs.shape)
            labels = Variable(torch.from_numpy(y_train))

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs.float())

        # get loss for the predicted output
        labels = labels.float()
        loss = criterion(outputs, labels)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()
        # if epoch == epochs - 1:
        #     print('epoch {}, loss {}'.format(epoch, loss.item()))

# if __name__ == "__main__":
#     # create dummy data for training
#     m = 5
#     x_train = np.random.random_sample((10, m))
    
#     y_values = []
#     for row in x_train:
#         y_values.append(sum(row))
#     y_train = np.array(y_values, dtype=np.float32).reshape(-1,1)
    
#     # train
#     inputDim = m        # takes variable 'x'
#     outputDim = 1       # takes variable 'y'
#     learningRate = 0.01
#     epochs = 50
#     model = linearRegression(inputDim, outputDim)
#     train(model, inputDim, outputDim, learningRate, epochs, x_train, y_train)
    
#     # evaluate
#     x_eval = np.random.random_sample((1, m))

#     for row in x_eval:
#         inputs = Variable(torch.from_numpy(row).cpu())
#         outputs = model(inputs.float())
#         print(np.sum(inputs.cpu().detach().numpy()))
#         print(outputs.cpu().detach().numpy()[0])