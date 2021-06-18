import torch
from torch import nn
import numpy as np
import utils
from memory import DKVMN


class MODEL(nn.Module):
    def __init__(self, n_question, batch_size, q_embed_dim, qa_embed_dim, memory_size, memory_key_state_dim,
                memory_value_state_dim, final_fc_dim, student_num=None):
        self.n_question = n_question
        self.batch_size = batch_size
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        self.final_fc_dim = final_fc_dim
        self.student_num = student_num

        self.input_embed_linear = nn.Linear(self.q_embed_dim, self.final_fc_dim, bias=True)
        self.read_embed_linear = nn.Linear(self.memory_value_state_dim + self.final_fc_dim, self.final_fc_dim, bias=True)
        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)
        self.init_memory_key = nn.Parameter(torch.randn(self.memory_size, self.memory_key_state_dim))
        nn.init.kaiming_normal(self.init_memory_key)
        self.init_memory_value = nn.Parameter(torch.randn(self.memory_size, self.memory_value_state_dim))
        nn.init.kaiming_normal(self.init_memory_value)

        self.mem = DKVMN(self.memory_size, self.memory_key_state_dim, self.memory_value_state_dim, self.init_memory_key)
        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.set_memory_value(memory_value)
        
        self.q_embed = nn.Embedding(self.n_question + 1, self.q_embed_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(self.n_question * 2 + 1, self.qa_embed_dim, padding_idx=0)

    def init_params(self):
        nn.init.kaiming_normal(self.predict_linear.weight)
        nn.init.kaiming_normal(self.read_embed_linear.weight)
        nn.init.constant(self.read_embed_linear.bias, 0)
        nn.init.constant(self.predict_linear.bias, 0)

    def init_embeddings(self):
        nn.init.kaiming_normal(self.q_embed.weight)
        nn.init.kaiming_normal(self.qa_embed.weight)

    def forward(self, q_data, qa_data, target, student_id=None):
        batch_size = q_data.shpae[0]
        seqlen = q_data.shape[1]
        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(qa_data)

        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.set_memory_value(memory_value)
        
        slice_q_data = torch.chunk(q_data, seqlen, 1)
        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)
        slice_qa_embed_data = torch.chunk(qa_embed_data, seqlen, 1)
        
        value_read_content_1 = []
        input_embed_1 = []
        predict_logs = []
        for i in range(seqlen):
            # attention
            q = slice_q_embed_data[i].squeeze(1)
            correlation_weight = self.mem.attention(q)
            if_memory_write = slice_q_data[i].squeeze(1).ge(1)
            if_memory_write = utils.variable(torch.FloatTensor(if_memory_write.data.tolist()), 1)

            # read
            read_content = self.mem.read(correlation_weight)
            value_read_content_1.append(read_content)
            input_embed_1.append(q)

            # write
            qa = slice_qa_embed_data[i].squeeze(1)
            new_memory_value = self.mem.write(correlation_weight, qa)

        all_read_value_content = torch.cat([value_read_content_1[i].squeeze(1) for i in range(seqlen)], 1)
        input_embed_content = torch.cat([input_embed_1[i].squeeze(1) for i in range(seqlen)], 1)

        predict_input = torch.cat([all_read_value_content, input_embed_content], 2)
        read_content_embed = torch.tanh(self.read_embed_linear(predict_input.reshape(batch_size * seqlen, -1)))
        
        pred = self.predict_linear(read_content_embed)

        target_1d = target
        mask = target_1d.ge(0)

        pred_1d = pred.reshape(-1, 1)
        
        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        loss = nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        return loss, torch.sigmoid(filtered_pred), filtered_target
