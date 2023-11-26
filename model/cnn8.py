import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 

class CNN(nn.Module):
    def __init__(self, in_channels, out_dim=1, linear_dim=7):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.linear_dim = linear_dim
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=self.in_channels,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,
                bias=False
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 64, 5, 1, 2, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 256, 5, 1, 2, bias=False),     
            nn.ReLU(),
            nn.Conv2d(256, 256, 5, 1, 2, bias=False),     
            nn.ReLU(),
            nn.Conv2d(256, 64, 5, 1, 2, bias=False),     
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, 1, 2, bias=False),     
            nn.ReLU(),
            nn.MaxPool2d(2),                
        )
        self.out = nn.Linear(32 * self.linear_dim * self.linear_dim, 10, bias=False)
        self.out2 = nn.Linear(10, self.out_dim, bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)       
        output = self.out(x)
        output = self.out2(output)
        return output
    
    def get_activation_before_last_layer(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)       
        output = self.out(x)
        return output

def train(model, dataloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        average_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')

model = CNN(in_channels=3, out_dim=1, linear_dim=7)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train(model, train_loader, criterion, optimizer, epochs=5)
