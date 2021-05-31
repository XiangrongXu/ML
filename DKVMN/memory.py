import torch
import utils
import numpy as np

class DKVMNHeadGroup(torch.nn.Module):
    def __init__(self, memory_size, memory_state_dim, is_write):
        """
        params:
            memory_size: scaler
            memory_state_dim: scaler
            is_write: boolean
        """
        super(DKVMNHeadGroup, self).__init__()
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write
        if is_write:
            self.erase = torch.nn.Linear(self.memory_state_dim, self.memory_state_dim, bias=True)
            self.add = torch.nn.Linear(self.memory_state_dim, self.memory_state_dim, bias=True)
            torch.nn.init.kaiming_normal(self.erase.weight)
            torch.nn.init.constant(self.erase.bias, 0)
            torch.nn.init.kaiming_normal(self.add.weight)
            torch.nn.init.constant(self.erase.bias, 0)
        
    def get_correlation_weight_from_k(self, input_k, memory_key):
        """
        A funtion used to get the correlation weight w_t from the k
        params:
            input_k: Shape(batch_size, control_state_dim)
            memory_key: Shape(memory_size, memory_state_dim)
        Returns:
            correlation_weight: Shape(batch_size, memory_size)
        """
        similarity_score = torch.matmul(input_k, torch.t(memory_key))
        correlation_weight = torch.nn.functional.softmax(similarity_score, dim=1)
        return correlation_weight

    def read(self, memory_value, read_weight):
        """
        params:
            memory_value: Shape(batch_size, memory_size, memory_value_state_dim)
            read_weight: Shape(batch_size, memory_size)
        returns:
            read_content: Shape(batch_size, memory_value_state_dim)
        """
        read_weight = read_weight.reshape(shape=(-1, 1, self.memory_size))
        read_content = torch.matmul(read_weight, memory_value)
        read_content = read_content.reshape(shape=(-1, self.memory_size, self.memory_state_dim))
        read_content = torch.sum(read_content, dim=1)
        return read_content

    def write(self, input_v, memory_value, write_weight):
        """
        params:
            input_v: Shape(batch_size, input_v_state_dim)
            memory_value: Shape(batch_size, memory_size, memory_state_dim)
            write_weight: Shape(batch_size, memory_size)
        returns:
            new_memory_value: Shape(batch_size, memory_size, memory_state_dim) 
        """
        assert self.is_write
        erase_signal = torch.sigmoid(self.erase(input_v)).reshape((-1, 1, self.memory_state_dim))
        add_signal = torch.tanh(self.add(input_v)).reshape((-1, 1, self.memory_state_dim))
        write_weight_reshape = write_weight.reshape(-1, self.memory_size, 1)
        erase_mult = torch.matmul(write_weight_reshape, erase_signal)
        add_mult = torch.matmul(write_weight_reshape, add_signal)
        new_memory_value = memory_value * (1 - erase_mult) + add_mult
        return new_memory_value

class DKVMN(torch.nn.Module):
    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, init_memory_key):
        """
        params:
            memory_size: scalar
            memory_key_state_dim: scalar
            memory_value_state_dim: scalar
            init_memory_key: Shape(memory_size, memory_key_state_dim)
            init_memory_value: Shape(batch_size, memory_size, memory_value_state_dim)
        """
        super(DKVMN, self).__init__()
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        self.key_head = DKVMNHeadGroup(self.memory_size, self.memory_key_state_dim, False)
        self.value_head = DKVMNHeadGroup(self.memory_size, self.memory_value_state_dim, True)
        self.memory_key = init_memory_key
        # self.memory_value = init_memory_value
        self.memory_value = None

    def set_memory_value(self, memory_value):
        self.memory_value = memory_value

    def attention(self, input_k):
        correlation_weight = self.key_head.get_correlation_weight_from_k(input_k, self.memory_key)
        return correlation_weight

    def read(self, read_weight):
        read_content = self.value_head.read(self.memory_value, read_weight)
        return read_content

    def write(self, write_weight, input_v):
        memory_value = self.value_head.write(input_v, self.memory_value, write_weight)
        self.memory_value = torch.nn.Parameter(memory_value.data)
        return self.memory_value
