import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F

import deepsqa_preprocess_data.embedding as ebd
# glove_path = 'glove/glove.6B.300d.txt'  # 6B tokens, 400K vocab, 300-dimension embedding 
# create(glove_path)
# word_idx = ebd.load_idx()
# embedding_matrix = ebd.load()
# print(embedding_matrix.shape)


def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin

class ControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super().__init__()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(linear(dim * 2, dim))

        self.control_question = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

        self.dim = dim

    def forward(self, step, context, question, control):
        position_aware = self.position_aware[step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * context).sum(1)

        return next_control


class ReadUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

    def forward(self, memory, know, control):
        mem = self.mem(memory[-1]).unsqueeze(2)
        concat = self.concat(torch.cat([mem * know, know], 1) \
                                .permute(0, 2, 1))
        attn = concat * control[-1].unsqueeze(1)
        attn = self.attn(attn).squeeze(2)
        attn = F.softmax(attn, 1).unsqueeze(1)

        read = (attn * know).sum(2)

        return read


class WriteUnit(nn.Module):
    def __init__(self, dim, self_attention=False, memory_gate=False):
        super().__init__()

        self.concat = linear(dim * 2, dim)

        if self_attention:
            self.attn = linear(dim, 1)
            self.mem = linear(dim, dim)

        if memory_gate:
            self.control = linear(dim, 1)

        self.self_attention = self_attention
        self.memory_gate = memory_gate

    def forward(self, memories, retrieved, controls):
        prev_mem = memories[-1]
        concat = self.concat(torch.cat([retrieved, prev_mem], 1))
        next_mem = concat

        if self.self_attention:
            controls_cat = torch.stack(controls[:-1], 2)
            attn = controls[-1].unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            next_mem = self.mem(attn_mem) + concat

        if self.memory_gate:
            control = self.control(controls[-1])
            gate = F.sigmoid(control)
            next_mem = gate * prev_mem + (1 - gate) * next_mem

        return next_mem


class MACUnit(nn.Module):
    def __init__(self, dim, max_step=12,
                self_attention=False, memory_gate=False,
                dropout=0.15):
        super().__init__()

        self.control = ControlUnit(dim, max_step)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0 = nn.Parameter(torch.zeros(1, dim))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge):
        b_size = question.size(0)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        controls = [control]
        memories = [memory]

        for i in range(self.max_step):
            control = self.control(i, context, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)

            read = self.read(memories, knowledge, controls)
            memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories.append(memory)

        return memory

    
class SensoryNet(nn.Module):

    def __init__(self, source_data):
        super(SensoryNet, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        
        # define the input feature dimension based on source dataset
        if source_data == 'opp':
            self.feature_dim = 768  #77
        else:
            self.feature_dim = 225
            
        self.conv1 = nn.Conv2d(1, 64, (1,3), padding = (0,1))
        self.conv2 = nn.Conv2d(64, 64, (1,3), padding = (0,1))
        
        # lstm takes input [batch, seq, feature] (if batch_first). # original: (seq_len, batch, input_size): 
        self.lstm = nn.LSTM(64* self.feature_dim, 128, 1, batch_first=True)
#         self.lstm = nn.LSTM(64*77, 128, 1, batch_first=True)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(128, 128)
#         self.activation = nn.Tanh()


    def forward(self, x):

        batch_size = x.size()[0]
        x = x.permute(0, 3, 1,2)

        x = F.max_pool2d(F.relu(self.conv1(x)), (1, 2))
        x = F.dropout(x, 0.5)

        x = F.max_pool2d(F.relu(self.conv2(x)), (1,2))
        x = F.dropout(x, 0.5)

        x = x.permute(0, 3, 1, 2)

        x = x.reshape(batch_size, -1, 64* self.feature_dim)
#         x = x.view(batch_size, -1, 64*77)

        x, hidden = self.lstm(x)
        
        x = x[:,-1]   #i.e. x = x.select(0, maxlen-1).contiguous()
      
        x = torch.tanh(self.fc1(x))
#         x = F.relu(self.fc1(x))
#         x = self.fc1(x)
#         x = self.activation(x)
        
        return x  
    

class MACNetwork(nn.Module):
    def __init__(self, n_vocab, dim, embed_hidden=300, vocabulary_embd = True, embd_train = False, 
                max_step=12, self_attention=False, memory_gate=False,
                classes=28, dropout=0.15, source_data = 'opp'):
        super().__init__()
        
        self.sensornet = SensoryNet(source_data = source_data)

        self.conv = nn.Sequential(nn.Conv2d(128, dim, 1, padding=0),  # change input to 128 dime vec
                                nn.ELU(),
                                nn.Conv2d(dim, dim, 1, padding=0),
                                nn.ELU())
#         self.conv = nn.Sequential(nn.Conv2d(128, dim, 3, padding=1),  # change input to 128 dime vec
# #         self.conv = nn.Sequential(nn.Conv2d(1024, dim, 3, padding=1),
#                                 nn.ELU(),
#                                 nn.Conv2d(dim, dim, 3, padding=1),
#                                 nn.ELU())

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        
        if vocabulary_embd:
            # loading weights
            print('Loading glove word embeddings...')
            embedding_matrix = ebd.load()
            print(embedding_matrix.shape)
            self.embed.weight.data = torch.Tensor(embedding_matrix)
    #         self.embed.weight.data = torch.Tensor(get_or_load_embeddings())
            if embd_train:
                self.embed.weight.requires_grad = True
            else:
                self.embed.weight.requires_grad = False
            print('Word embeddings trainable: ', self.embed.weight.requires_grad)
        
        
        self.lstm = nn.LSTM(embed_hidden, dim,
                        batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(dim * 2, dim)

        self.mac = MACUnit(dim, max_step,
                        self_attention, memory_gate, dropout)
        
#         self.dropout = nn.Dropout(p=dropout)

        self.classifier = nn.Sequential(linear(dim * 3, dim),
                                        nn.ELU(),
                                        nn.Dropout(p = dropout),  # adding dropout here!
                                        linear(dim, classes))

        self.max_step = max_step
        self.dim = dim

        self.reset()

    def reset(self):
#         self.embed.weight.data.uniform_(0, 1)

        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()

        kaiming_uniform_(self.classifier[0].weight)

    def forward(self, image, question, question_len, dropout=0.15):
        b_size = question.size(0)

        image = self.sensornet(image)
        image = image.unsqueeze(-1)
        image = image.unsqueeze(-1)
        
        img = self.conv(image)
        img = img.view(b_size, self.dim, -1)

        embed = self.embed(question)
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len,
                                                batch_first=True)
        lstm_out, (h, _) = self.lstm(embed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                    batch_first=True)
        lstm_out = self.lstm_proj(lstm_out)
        h = h.permute(1, 0, 2).contiguous().view(b_size, -1)

        memory = self.mac(lstm_out, h, img)

        out = torch.cat([memory, h], 1)
        
#         # add dropout before classifier:
#         out = self.dropout(out)
#         #############################
        
        out = self.classifier(out)

        return out