import torch
import argparse
from model import MODEL
from run import train, test
import numpy as np
from torch import optim
from data_loader import Data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="the gpu will be used")
    parser.add_argument("--max_iter", type=int, default=300, help="number of iterations")
    parser.add_argument("--decay_epoch", type=int, default=20, help="number of iterations")
    parser.add_argument("--test", type=bool, default=False, help="enable tesing")
    parser.add_argument("--train_test", type=bool, default=True, help="enable testing")
    parser.add_argument("-show", type=bool, default=True, help="print progress")
    parser.add_argument("--init_std", type=float, default=0.1, help="weight initial std")
    parser.add_argument("--init_lr", type=float, default=0.01, help="initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.75, help="learning rate decay")
    parser.add_argument("--final_lr", type=float, default=1e-5, help="learning rate will not decrease after hitting the threshold final_lr")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum rate")
    parser.add_argument("--maxgradnorm", type=float, default=50.0, help="maximum gradient norm")
    parser.add_argument("--final_fc_dim", type=float, default=50, help="hidden state dim for final fc layer")

    dataset = "assist2009_updated"

    if dataset == "assist2009_updated":
        parser.add_argument("--q_embed_dim", type=int, default=50, help="question embedding dimensions")
        parser.add_argument("--batch_size", type=int, default=32, help="the batch size")
        parser.add_argument("--qa_embed_dim", type=int, default=200, help="answer and question embedding dimensions")
        parser.add_argument("--memory_size", type=int, default=20, help="memory_size")
        parser.add_argument("--n_question", type=int, default=110, help="the number of unique questions in the database")
        parser.add_argument("--seqlen", type=int, default=200, help="the allowed maximum length of a seqence")
        parser.add_argument("--data_dir", type=str, default="./data/assist2009_updated")
        parser.add_argument("--data_name", type=str, default="assist2009_updated")
        parser.add_argument("--load", type=str, default="assist2009_updated", help="model file to load")
        parser.add_argument("--save", type=str, default="assist2009_updated", help="path to save model")

    params = parser.parse_args()
    params.lr = params.init_lr
    params.memory_key_state_dim = params.q_embed_dim
    params.memory_value_state_dim = params.qa_embed_dim

    print(params)

    dat = Data(params.n_question, params.seqlen, ",")
    train_data_path = params.data_dir + "/" + params.data_name + "_train1.csv"
    valid_data_path = params.data_dir + "/" + params.data_name + "_valid1.csv"
    test_data_path = params.data.dir + "/" + params.data_name + "_test.csv"
    train_q_data, train_qa_data = dat.load_data(train_data_path)
    valid_q_data, valid_qa_data = dat.load_data(valid_data_path)
    test_q_data, test_qa_data = dat.load_data(test_data_path)

    model = MODEL(params.n_question, params.batch_size, params.q_embed_dim, params.qa_embed_dim, params.memory_size, params.memory_key_state_dim, params.memory_value_state_dim, params.final_fc_dim)

    model.init_embedding()
    model.init_params()

    optimizer = optim.Adam(params=model.parameters(), lr=params.lr, betas=(0.9, 0.9))

    if params.gpu >= 0:
        print("device: " + str(params.gpu))
        torch.cuda.set_device(params.gpu)
        model.cuda()

    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0

    for idx in range(params.max_iter):
        train_loss, train_accuracy, train_auc = train(idx, model, params, optimizer, train_q_data, train_qa_data)
        print(f"Epoch {idx + 1}/{params.max_iter}, loss: ")
        valid_loss, valid_accuracy, valid_auc = test(model, params, valid_q_data, valid_qa_data)

        all_train_auc[idx + 1] = train_auc
        all_train_accuracy[idx + 1] = train_accuracy
        all_train_loss[idx + 1] = train_loss
        all_valid_auc[idx + 1] = valid_auc
        all_valid_accuracy[idx + 1] = valid_accuracy
        all_valid_loss[idx + 1] = valid_loss

        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc

    print(f"best_auc: ${best_valid_auc}")

if __name__ == "__main__":
    main()

