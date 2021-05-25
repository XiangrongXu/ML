import mxnet as mx

class DKVMNHeadGroup(object):
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
        
    def addressing(self, control_input, memory):
        """
        params:
            control_input: Shape(batch_size, control_state_dim)
            memory: Shape(memory_size, memory_state_dim)
        Returns:
            correlation_weight: Shape(batch_size, memory_size)
        """

        similarity_score = mx.sym.FullyConnected(data=control_input,
                                                num_hidden=self.memory_size,
                                                weight=memory,
                                                no_bias=True,
                                                name="similarity_score")
        correlation_weight = mx.sym.SoftmaxActivation(similarity_score)
        return correlation_weight

    def read(self, memory, control_input=None, read_weight=None):
        """
        params:
            memory: Shape(batch_size, memory_size, memory_state_dim)
            control_input: Shape(batch_size, control_state_dim)
            read_weight: Shape(batch_size, memory_size)
        returns:
            read_content: Shape(batch_size, memory_state_dim)
        """
        if read_weight is None:
            read_weight = self.addressing(control_input=control_input, memory=memory)
        read_weight = mx.sym.Reshape(read_weight, shape=(-1, 1, self.memory_size))
        read_content = mx.sym.Reshape(data=mx.sym.batch_dot(read_weight, memory), shape=(-1, self.memory_state_dim))
        return read_content

    def write(self, control_input, memory, write_weight=None):
        """
        params:
            control_input: Shape(batch_size, control_state_dim)
            memory: Shape(batch_size, memory_size, memory_state_dim)
            write_weight: Shape(batch_size, memory_size)
        returns:
            new_memory: Shape(batch_size, memory_size, memory_state_dim) 
        """
        assert self.is_write
        if write_weight is None:
            write_weight = self.addressing(control_input=control_input, memory=memory)
        erase_signal = mx.sym.FullyConnected(data=control_input, num_hidden=self.memory_state_dim,
                                            weight=self.erase_signal_weight, bias=self.erase_signal_bias)
        erase_signal = mx.sym.Activation(data=erase_signal, act_type="sigmoid", name=self.name + "_erase_signal")
        add_signal = mx.sym.FullyConnected(data=control_input, num_hidden=self.memory_state_dim,
                                            weight=self.add_signal_weight, bias=self.add_signal_bias)
        add_signal = mx.sym.Activation(data=add_signal, act_tytpe="tanh", name=self.name + "_add_signal")
        erase_mut = 1 - mx.sym.batch_dot(mx.sym.Reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                        mx.sym.Reshape(erase_signal, shape=(-1, 1, self.memory_state_dim)))
        aggre_add_signal = mx.sym.batch_dot(mx.sym.Reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                            mx.sym.Reshape(add_signal, shape=(-1, 1, self.memory_state_dim)))
        new_memory = memory * erase_mut + aggre_add_signal
        return new_memory

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
        self.key_head = DKVMNHeadGroup(memory_size=memory_size)

