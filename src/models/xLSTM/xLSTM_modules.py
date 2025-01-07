import torch
import torch.nn as nn
import torch.nn.functional as F


class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices for gates and cell state
        self.Wz = nn.Linear(input_size, hidden_size)
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Wo = nn.Linear(input_size, hidden_size)

        self.Rz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ri = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Rf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ro = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, hidden):
        h_prev, c_prev, C_prev, n_prev, m_prev = hidden

        # Gates with exponential activation functions
        i = torch.exp(self.Wi(x) + self.Ri(h_prev))
        f = torch.sigmoid(self.Wf(x) + self.Rf(h_prev))
        o = torch.sigmoid(self.Wo(x) + self.Ro(h_prev))

        # Candidate cell state
        z = torch.tanh(self.Wz(x) + self.Rz(h_prev))

        # Update cell state
        c = f * c_prev + i * z

        # Normalizer state
        n = f * n_prev + i

        # Stabilization
        m = torch.max(torch.log(f) + m_prev, torch.log(i))
        # i_stable = torch.exp(torch.log(i) - m)
        # f_stable = torch.exp(torch.log(f) + m_prev - m)

        # Update hidden state
        h = o * (c / n)

        return h, (h, c,C_prev, n, m)


# Example usage
# input_size = 10
# hidden_size = 20
# seq_length = 5
# batch_size = 3
#
# sLSTM = sLSTMCell(input_size, hidden_size)
# x = torch.randn(seq_length, batch_size, input_size)
# h, c, n, m = torch.zeros(batch_size, hidden_size), torch.zeros(batch_size, hidden_size), torch.zeros(batch_size,
#                                                                                                      hidden_size), torch.zeros(
#     batch_size, hidden_size)
#
# for t in range(seq_length):
#     h, (h, c, n, m) = sLSTM(x[t], (h, c, n, m))
#
# print(h)
# print(x.size())
# print(h.size())

class mLSTMCell(nn.Module):
    def __init__(self, input_size, memory_size):
        super(mLSTMCell, self).__init__()
        self.input_size = input_size
        self.memory_size = memory_size

        # Linear layers for query, key, value, and gates
        self.Wq = nn.Linear(input_size, memory_size)  # Query input
        self.Wk = nn.Linear(input_size, memory_size)  # Key input
        self.Wv = nn.Linear(input_size, memory_size)  # Value input

        self.Wi = nn.Linear(input_size, memory_size)  # Input gate
        self.Wf = nn.Linear(input_size, memory_size)  # Forget gate
        self.Wo = nn.Linear(input_size, memory_size)  # Output gate

    def forward(self, x, hidden):
        h_prev, c_prev, C_prev, n_prev, m_prev = hidden

        # Query, Key, and Value calculations
        q_t = self.Wq(x) # Query input
        k_t = (1 / torch.sqrt(torch.tensor(self.memory_size, dtype=torch.float32))) * (
                    self.Wk(x))  # Key input
        v_t = self.Wv(x)   # Value input

        # Unsqueeze for batch-wise outer product
        v_t = v_t.unsqueeze(-1)  # [batch_size, memory_size, 1]
        k_t_T = k_t.unsqueeze(1)  # [batch_size, 1, memory_size]

        # Input gate (exponential)
        i_t = torch.exp(self.Wi(x))  # exp(i_t)

        # Forget gate (can be sigmoid or exponential)
        f_t = torch.sigmoid(self.Wf(x))  # sigmoid(f_t)

        # Update cell state
        C_t = f_t.unsqueeze(-1) * C_prev+ i_t.unsqueeze(-1) * (v_t @ k_t_T)  # Outer product v_t @ k_t_T

        # Update normalizer state
        n_t = f_t * n_prev + i_t * k_t

        # Compute hidden state
        h_tilde = C_t @ q_t.unsqueeze(-1)  # Matrix multiplication C_t @ q_t
        h_t = torch.sigmoid(self.Wo(x)) * (h_tilde.squeeze(-1) / torch.max((n_t.unsqueeze(1) @ q_t.unsqueeze(-1)).squeeze(-1),
                                                                           torch.ones_like(
                                                                               (n_t.unsqueeze(1) @ q_t.unsqueeze(-1)).squeeze(-1))))

        return h_t, (h_t,c_prev, C_t, n_t, m_prev)


# Example usage:
# input_size = 10
# memory_size = 20
# batch_size = 3
#
# mLSTM = mLSTMCell(input_size, memory_size)
# x = torch.randn(batch_size, input_size)  # Input at time t
#
# h_prev = torch.zeros(batch_size, memory_size)
# C_prev = torch.zeros(batch_size, memory_size, memory_size)
# n_prev = torch.zeros(batch_size, memory_size)
#
# h_t, (h_t, C_t, n_t) = mLSTM(x, (h_prev, C_prev, n_prev))
# print(h_t)
# print(x.size())
# print(h_t.size())