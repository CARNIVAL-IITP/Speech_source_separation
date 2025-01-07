from .xLSTM_modules import sLSTMCell,mLSTMCell

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomxLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,batch_first=True):
        super(CustomxLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Create a list of LSTMCells for each layer
        # self.lstm_cells = nn.ModuleList([sLSTM(input_size, hidden_size),mLSTM(input_size, hidden_size)])
        self.lstm_cells = nn.ModuleList()
        for _ in range(0, num_layers):
            self.lstm_cells.append(sLSTMCell(input_size, hidden_size))
            self.lstm_cells.append(mLSTMCell(hidden_size, hidden_size))

    def forward(self, input_seq, initial_states=None):
        """
        :param input_seq: Tensor of shape (batch_size, seq_length, input_size)
        :param initial_states: Initial hidden and cell states as a tuple of (h_0, c_0)
                               - h_0: Tensor of shape (num_layers, batch_size, hidden_size)
                               - c_0: Tensor of shape (num_layers, batch_size, hidden_size)
        :return: output: Tensor of shape (batch_size, seq_length, hidden_size)
                 (h_n, c_n): The final hidden and cell states
        """
        if self.batch_first == True:
            batch_size, seq_length, _ = input_seq.size()
        else:
            seq_length, batch_size, _ = input_seq.size()
            input_seq = input_seq.permute(1,0,2)

        # Initialize hidden and cell states if not provided
        if initial_states is None:
            h_t = [torch.zeros(batch_size, self.hidden_size).to(input_seq.device) for _ in range(self.num_layers)]
            c_t = [torch.zeros(batch_size, self.hidden_size).to(input_seq.device) for _ in range(self.num_layers)]
            C_t = [torch.zeros(batch_size, self.hidden_size , self.hidden_size).to(input_seq.device) for _ in range(self.num_layers)]
            n_t = [torch.zeros(batch_size, self.hidden_size).to(input_seq.device) for _ in range(self.num_layers)]
            m_t = [torch.zeros(batch_size, self.hidden_size).to(input_seq.device) for _ in range(self.num_layers)]
        else:
            h_t, c_t = initial_states

        # Store the output for each time step
        outputs = []

        # Iterate over the sequence
        for t in range(seq_length):
            input_t = input_seq[:, t, :]  # Extract the input for the current time step
            for layer in range(self.num_layers):
                h_t[layer], (h_t[layer], c_t[layer], C_t[layer], n_t[layer],m_t[layer]) = self.lstm_cells[layer](input_t,(h_t[layer], c_t[layer], C_t[layer],n_t[layer],m_t[layer]))
                input_t = h_t[layer]  # The output of the current layer is the input to the next layer

            outputs.append(h_t[-1])  # Collect the output from the last layer

        # Stack the outputs to form the output sequence
        output = torch.stack(outputs, dim=1)

        # Return the output sequence and the final hidden and cell states
        if self.batch_first == True:
            return output, (h_t, c_t)
        else:
            return output.permute(1,0,2), (h_t, c_t)


asdf = torch.randn([16,343,128])
layers = CustomxLSTM(128,256,2)

zxcv = layers(asdf)
