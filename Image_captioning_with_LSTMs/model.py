import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Using ResNet-50 for a lighter model
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features



import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Initialize the Decoder RNN.
        Args:
        - embed_size: Size of the word embeddings.
        - hidden_size: Number of features in the hidden state of the RNN.
        - vocab_size: Size of the vocabulary.
        - num_layers: Number of layers in the RNN (default is 1).
        """
        super(DecoderRNN, self).__init__()
        
        # Embedding layer to convert words into vector representations
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer to process the sequence of embedded words
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to map LSTM outputs to the vocabulary size
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Store the number of layers and hidden size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, features, captions):
        """
        Forward pass for the decoder.
        Args:
        - features: Image features from the encoder.
        - captions: Ground truth caption sequences for training.
        Returns:
        - outputs: Predicted words at each time step (logits before softmax).
        """
        # Embed the captions (excluding the last <end> token for training)
        embeddings = self.embed(captions[:, :-1])
        
        # Concatenate the image features with the embedded captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # Pass through the LSTM
        hiddens, _ = self.lstm(embeddings)
        
        # Pass through the fully connected layer to get vocabulary predictions
        outputs = self.fc(hiddens)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """
        Generate a sequence of words from image features using the LSTM.
        Args:
        - inputs: The encoded image features.
        - states: Initial states for the LSTM (default is None).
        - max_len: Maximum length of the generated caption (default is 20).
        Returns:
        - caption: List of predicted word indices.
        """
        caption = []
        
        states = (torch.randn(1, 1, self.hidden_size).to(inputs.device), 
                  torch.randn(1, 1, self.hidden_size).to(inputs.device))

        # Generate the caption word by word
        for _ in range(max_len):
            # Pass the input and the hidden states through the LSTM
            lstm_out, states = self.lstm(inputs, states)  # lstm_out: (1, 1, hidden_size)
            
            # Pass LSTM output through the fully connected layer to get the vocabulary distribution
            outputs = self.fc(lstm_out.squeeze(1))        # outputs: (1, vocab_size)
            
            # Get the index of the most probable word
            wordid = outputs.argmax(dim=1)                # wordid: (1)
            
            # Append the predicted word to the caption
            caption.append(wordid.item())
            
            # If the predicted word is the end token, stop early (optional)
            #if wordid.item() == end_token_idx:            # Make sure to pass the correct `end_token_idx`
             #   break

            # Embed the predicted word to use as the next input
            inputs = self.embed(wordid).unsqueeze(1)      # inputs: (1, 1, embed_size)

        return caption
