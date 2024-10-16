## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional Block 1 (using 5x5 kernels)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2 (using 3x3 kernels)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
      

        # Convolutional Block 3 (using 3x3 kernels)
        self.conv4 = nn.Conv2d(64,128, kernel_size=3, padding=0)

        # Convolutional Block 4 (using 3x3 kernels)
        self.conv5 = nn.Conv2d(128,256, kernel_size=3, padding=0)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=0)
        

        # Define the fully connected layers
        self.fc1 = nn.Linear(512 * 1 * 1, 512)  # Adjust size according to the final output size after convolutions
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256, 68*2)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #x = self.bn3(x)  # Apply Batch Normalization after activation

        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        # Flatten the tensor while preserving the batch dimension
        x = x.view(x.size(0), -1)

        # Forward pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)

        return x
    


class Net_1(nn.Module):
    def __init__(self):
        super(Net_1, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Layer Normalization layers
        self.ln1 = nn.LayerNorm([32, 224, 224])  # Normalize the output of conv1
        self.ln2 = nn.LayerNorm([64, 112, 112])  # Normalize the output of conv2
        self.ln3 = nn.LayerNorm([128, 56, 56])   # Normalize the output of conv3
        self.ln4 = nn.LayerNorm([256, 28, 28])   # Normalize the output of conv4
        self.ln5 = nn.LayerNorm([512, 14, 14])   # Normalize the output of conv5

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 136)  # Output is 68 keypoints * 2 coordinates (x, y)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        # Apply convolutional layers with Layer Normalization and ReLU
        x = self.pool(F.relu(self.ln1(self.conv1(x))))  # Conv1 -> LN -> ReLU -> MaxPool
        x = self.pool(F.relu(self.ln2(self.conv2(x))))  # Conv2 -> LN -> ReLU -> MaxPool
        x = self.pool(F.relu(self.ln3(self.conv3(x))))  # Conv3 -> LN -> ReLU -> MaxPool
        x = self.pool(F.relu(self.ln4(self.conv4(x))))  # Conv4 -> LN -> ReLU -> MaxPool
        x = self.pool(F.relu(self.ln5(self.conv5(x))))  # Conv5 -> LN -> ReLU -> MaxPool
        
        # Flatten the tensor before passing to fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Output layer

        return x
