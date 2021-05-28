from mxnet.ndarray.gen_op import norm
import numpy as np
import math
import mxnet as mx
import mxnet.ndarray as nd
from sklearn import metrics

def norm_clipping(params_grad, threshold):
    norm_val = 0.0
    for i in range(len(params_grad[0])):
        norm_val += np.sqrt(sum([nd.norm(grads[i]).asnumpy()[0] ** 2 for grads in params_grad]))
    norm_val /= float(len(params_grad[0]))

    if norm_val > threshold:
        ratio = threshold / float(norm_val)
        for grads in params_grad:
            for grad in grads:
                grad[:] *= ratio

    return norm_val

def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, 1.0-pred)) + (1.0 - target) * np.log(1-np.maximum(1e-10, 1.0-pred))

    if mod == "avg":
        return np.average(loss) * (-1.0)
    elif mod == "sum":
        return -loss.sum()
    else:
        assert False

def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)

def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)

def train(net, params, q_data, qa_data, label):
    N = int(math.floor(len(q_data) / params.batch_size))
    q_data = q_data.T
    qa_data = qa_data.T
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]

    pred_list = []
    target_list = []

    if params.show:
        from utils import ProgressBar
        bar = ProgressBar(label, max=N)
    
    for idx in range(N):
        if params.show: bar.next()

        q_one_seq = q_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        input_q = q_one_seq[:, :]
        qa_one_seq = qa_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        input_qa = qa_one_seq[:, :]
        
        target = qa_one_seq[:, :]
        target = (target - 1) / params.n_question
        target = np.floor(target)

        input_q = mx.nd.array(input_q)
        
        

