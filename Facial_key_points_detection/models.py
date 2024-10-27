## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import torchvision.models as models

class VGG16SingleChannel(nn.Module):
    def __init__(self, num_keypoints=68):
        super(VGG16SingleChannel, self).__init__()
        
        # Load the pretrained VGG16 model
        self.vgg16 = models.vgg16(weights='DEFAULT')  # Load the model
        
        # Modify the first layer to accept a single channel
        self.vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        # Modify the classifier to output keypoints
        self.vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=num_keypoints * 2)  # x, y for each keypoint
    
    def forward(self, x):
        return self.vgg16(x)



class Net_1(nn.Module):
    def __init__(self):
        super(Net_1, self).__init__()
        
        # Conv Block 1
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers (adjusted for regression)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)  # The input size depends on the input image dimensions
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 136)  # 68 keypoints * 2 (x, y) coordinates

        # Dropout layers
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv Block 1
        x = nn.ReLU()(self.conv1_1(x))
        x = nn.ReLU()(self.conv1_2(x))
        x = self.pool1(x)

        # Conv Block 2
        x = nn.ReLU()(self.conv2_1(x))
        x = nn.ReLU()(self.conv2_2(x))
        x = self.pool2(x)

        # Conv Block 3
        x = nn.ReLU()(self.conv3_1(x))
        x = nn.ReLU()(self.conv3_2(x))
        x = nn.ReLU()(self.conv3_3(x))
        x = self.pool3(x)

        # Conv Block 4
        x = nn.ReLU()(self.conv4_1(x))
        x = nn.ReLU()(self.conv4_2(x))
        x = nn.ReLU()(self.conv4_3(x))
        x = self.pool4(x)

        # Conv Block 5
        x = nn.ReLU()(self.conv5_1(x))
        x = nn.ReLU()(self.conv5_2(x))
        x = nn.ReLU()(self.conv5_3(x))
        x = self.pool5(x)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(nn.ReLU()(self.fc1(x)))
        x = self.dropout(nn.ReLU()(self.fc2(x)))
        x = self.fc3(x)

        return x





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Block 1
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 68 * 2)  # Output layer for 136 values (68 x 2)
        
        # Activation and Dropout
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Block 1
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool1(x)
        
        # Block 2
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool2(x)
        
        # Block 3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.relu(self.conv3_4(x))
        x = self.pool3(x)
        
        # Block 4
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.relu(self.conv4_4(x))
        x = self.pool4(x)
        
        # Block 5
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.relu(self.conv5_4(x))
        x = self.pool5(x)
        
        # Flatten the output for the fully connected layers
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    


class Net_2(nn.Module):
    def __init__(self):
        super(Net_2, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # Convolution2d1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # Convolution2d2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)  # Convolution2d3
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)  # Convolution2d4

        # Max Pooling Layers
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout Layers
        self.dropout1 = nn.Dropout(0.1)  # Dropout1
        self.dropout2 = nn.Dropout(0.2)  # Dropout2
        self.dropout3 = nn.Dropout(0.3)  # Dropout3
        self.dropout4 = nn.Dropout(0.4)  # Dropout4
        self.dropout5 = nn.Dropout(0.5)  # Dropout5
        self.dropout6 = nn.Dropout(0.6)  # Dropout6

        # Fully Connected Layers
        self.fc1 = nn.Linear(36864, 1000)  # Dense1
        self.fc2 = nn.Linear(1000, 1000)  # Dense2
        self.fc3 = nn.Linear(1000, 68 * 2)  # Dense3 (output layer)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolutional Block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        #x = self.dropout1(x)

        # Convolutional Block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        #x = self.dropout2(x)

        # Convolutional Block 3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        #x = self.dropout3(x)

        # Convolutional Block 4
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        #x = self.dropout4(x)

        # Flatten
        x = x.view(x.size(0), -1)  # Flatten layer

        # Fully Connected Block 1
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Fully Connected Block 2
        x = self.fc2(x)
        x = self.relu(x)
        #x = self.dropout6(x)

        # Output Layer
        x = self.fc3(x)  # No activation for regression output

        return x


