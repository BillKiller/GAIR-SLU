from torch import nn
from fastNLP.modules import MLP


class MLPAdapter(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """
    def __init__(self,
                 mlp_type,
                 input_dim,
                 output_dim,
                 drop_out=0.4,
                 alpha=0.2):
        super(MLPAdapter, self).__init__()
        if mlp_type == 'pure':
            self.linear = nn.Linear(input_dim, output_dim)
        elif mlp_type == 'qin':
            self.linear = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LeakyReLU(alpha),
                nn.Linear(input_dim, output_dim),  # 解除耦合
            )
        elif mlp_type == 'fastnlp':
            self.linear = MLP([input_dim, input_dim, output_dim])

    def forward(self, inputs):
        x = inputs
        return  self.linear(x)
