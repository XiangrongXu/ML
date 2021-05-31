import numpy as np
import math

class Data:
    def __init__(self, n_question, seqlen, separate_char, name="data"):
        """
        In the ASSISTments2009 dataset:
        param: n_question = 110
               seqlen = 200
        """
        self.name = name
        self.separate_char = separate_char
        self.n_question = n_question
        self.seqlen = seqlen

    """
    data format
    15
    1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
    """
    def load_data(self, path):
        f_data = open(path, "r")
        q_data = []
        qa_data = []
        for line_id, line in enumerate(f_data):
            line = line.strip()
            # when read the line of questions, only split the questions and do nothing else
            if line_id % 3 == 1:
                Q = line.split(self.separate_char)
                if len(Q[-1]) == 0:
                    Q = Q[:-1]
            # when read the line of answers, split the answers and start to put data into q_data and qa_data
            elif line_id % 3 == 2:
                A = line.split(self.separate_char)
                if len(A[-1]) == 0:
                    A = A[:-1]

                #start to split the data
                n_split = math.ceil(len(Q) / self.seqlen)
                
                for k in range(n_split):
                    question_seq = []
                    answer_seq = []
                    end_index = (k + 1) * self.seqlen if k < n_split else len(A)
                    for i in range(k * self.seqlen, end_index):
                        if len(Q[i]) > 0:
                            X_index = int(Q[i]) + int(A[i]) * self.n_question
                            question_seq.append(int(Q[i]))
                            answer_seq.append(X_index)
                        else:
                            print(Q[i])
                    
                    q_data.append(question_seq)
                    qa_data.append(answer_seq)
        f_data.close()
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat
        
        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat
        
        return q_dataArray, qa_dataArray
                
    def generate_all_index_data(self, batch_size):
        n_question = self.n_question
        batch = math.ceil(n_question / self.seqlen)

        seq = np.arange(1, self.seqlen * batch + 1)
        zero_index = np.arange(n_question, self.seqlen * batch)
        zero_index = zero_index.tolist()
        seq[zero_index] = 0
        q = seq.reshape((batch, self.seqlen))
        q_dataArray = np.zeros((batch_size, self.seqlen))
        q_dataArray[:batch, :] = q
        return q_dataArray