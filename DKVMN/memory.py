import torch
import utils
import numpy as np

class DKVMNHeadGroup(nn.Module):
    def __init__(self, memory_size, memory_state_dim, is_write):
        """
        params:
            memory_size: scaler
            memory_state_dim: scaler
            is_write: boolean
        """
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
        # Y = X * W.T + b
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
        read_content = read_content.reshape(shape=(-1, self.memory_state_dim))
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

class DKVMN:
    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, init_memory_key=None, init_memory_value=None,
                name="DKVMN"):
        self.name = name
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        self.init_memory_key = mx.sym.Variable(self.name + ": init_memory_key_weight") if init_memory_key is None\
                                else init_memory_key
        self.init_memory_value = mx.sym.Variable(self.name + ": init_memory_value_weight") if init_memory_value is None\
                                else init_memory_value
        self.key_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                        memory_state_dim=self.memory_key_state_dim,
                                        is_write=False,
                                        name=self.name + "->key_head")
        self.value_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                            memory_state_dim=self.memory_value_state_dim,
                                            is_write=True,
                                            name=self.name + "->value_head")
        self.memory_key = self.init_memory_key
        self.memory_value = self.init_memory_value

    def attention(self, input_k):
        assert isinstance(input_k, mx.symbol.Symbol)
        correlation_weight = self.key_head.get_correlation_weight_from_k(input_k=input_k, memory_key=self.memory_key)
        return correlation_weight

    def read(self, read_weight):
        read_content = self.value_head.read(memory_value=self.memory_value, read_weight=read_weight)
        return read_content

    def write(self, write_weight, input_v):
        assert isinstance(input_v, mx.symbol.Symbol)
        self.memory_value = self.value_head.write(input_v=input_v, memory_value=self.memory_value,
                                                    write_weight=write_weight)
        return self.memory_value
