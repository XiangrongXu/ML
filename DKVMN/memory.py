import mxnet as mx

class DKVMNHeadGroup:
    def __init__(self, memory_size, memory_state_dim, is_write, name="DVKMNHeadGroup"):
        """
        params:
            memory_size: scaler
            memory_state_dim: scaler
            is_write: boolean
        """
        self.name = name
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write
        if is_write:
            self.erase_signal_weight = mx.sym.Variable(name=name + ":erase_signal_weight")
            self.erase_signal_bias = mx.sym.Variable(name=name + ":erase_signal_bias")
            self.add_signal_weight = mx.sym.Variable(name=name + ":add_signal_weight")
            self.add_signal_bias = mx.sym.Variable(name=name + ":add_signal_bias")
        
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
        similarity_score = mx.sym.FullyConnected(data=input_k,
                                                num_hidden=self.memory_size,
                                                weight=memory_key,
                                                no_bias=True,
                                                name="similarity_score")
        correlation_weight = mx.sym.SoftmaxActivation(similarity_score)
        return correlation_weight

    def read(self, memory_value, read_weight):
        """
        params:
            memory_value: Shape(batch_size, memory_size, memory_value_state_dim)
            read_weight: Shape(batch_size, memory_size)
        returns:
            read_content: Shape(batch_size, memory_value_state_dim)
        """
        read_weight = mx.sym.Reshape(read_weight, shape=(-1, 1, self.memory_size))
        read_content = mx.sym.Reshape(data=mx.sym.batch_dot(read_weight, memory_value), shape=(-1, self.memory_state_dim))
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
        erase_signal = mx.sym.FullyConnected(data=input_v, num_hidden=self.memory_state_dim,
                                            weight=self.erase_signal_weight, bias=self.erase_signal_bias)
        erase_signal = mx.sym.Activation(data=erase_signal, act_type="sigmoid", name=self.name + "_erase_signal")
        add_signal = mx.sym.FullyConnected(data=input_v, num_hidden=self.memory_state_dim,
                                            weight=self.add_signal_weight, bias=self.add_signal_bias)
        add_signal = mx.sym.Activation(data=add_signal, act_tytpe="tanh", name=self.name + "_add_signal")
        erase_mut = 1 - mx.sym.batch_dot(mx.sym.Reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                        mx.sym.Reshape(erase_signal, shape=(-1, 1, self.memory_state_dim)))
        aggre_add_signal = mx.sym.batch_dot(mx.sym.Reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                            mx.sym.Reshape(add_signal, shape=(-1, 1, self.memory_state_dim)))
        new_memory_value = memory_value * erase_mut + aggre_add_signal
        return new_memory_value

class DKVMN(object):
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
