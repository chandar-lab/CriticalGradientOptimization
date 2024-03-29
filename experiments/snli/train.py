import sys

from dataset_utils.data_iterator import *

sys.path.append('../..')
from optimizers.optim import SGD_C, SGD, Adam_C, Adam, RMSprop, RMSprop_C
import numpy as np
import argparse
import os
import submitit
from pathlib import Path
from datetime import datetime
import wandb
from filelock import FileLock

import time

import torch
from torch.autograd import Variable
import torch.nn as nn

from infersent_comp.models import NLINet

from itertools import product

# commandline arguments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getEmbeddingWeights(vocab, dataset='snli'):
    emb_dict = {}
    with open(f'Utils/glove_' + dataset + '_embeddings.tsv', 'r') as f:
        for l in f:
            line = l.split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            emb_dict.update({word: vect})
    vectors = []
    for i in range(len(vocab)):
        vectors += [emb_dict[vocab[i]]]
    return torch.from_numpy(np.stack(vectors)).to(device)


def trainepoch(nli_net, train_iter, optimizer, loss_fn, epoch, params):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    epoch_loss = 0
    S = {'yes': 0, 'no': 0}
    # shuffle the data

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0][
                                          'lr'] * params.decay if epoch > 1 and 'sgd' in params.optimizer else \
        optimizer.param_groups[0]['lr']

    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))
    total_samples = 0

    for i, batch in enumerate(train_iter):
        # prepare batch
        s1_batch, s1_len = batch.Sentence1
        s2_batch, s2_len = batch.Sentence2
        s1_batch, s2_batch = Variable(s1_batch.to(device)), Variable(
            s2_batch.to(device))
        tgt_batch = batch.Label.to(device)
        k = s1_batch.size(1)  # actual batch size
        total_samples += k
        # model forward
        output, (s1_out, s2_out) = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

        # loss
        # pdb.set_trace()
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.item())
        words_count += (s1_batch.nelement() + s2_batch.nelement())

        # backward
        optimizer.zero_grad()
        loss.backward()

        wandb.log({"Iteration Training Loss": loss})

        # update the parameters
        optimizer.step()
        stats = optimizer.getOfflineStats()
        if stats:
            for k, v in stats.items():
                S[k] += v
        epoch_loss += loss.item()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in nli_net.parameters():
            if p.requires_grad:
                p.grad.div_(k)  # divide by the actual batch size
                total_norm += p.grad.norm() ** 2
        total_norm = np.sqrt(total_norm.item())

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0][
            'lr']  # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor  # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            logs.append(
                '{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                    (i) * params.batch_size, round(np.mean(all_costs), 2),
                    int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                    int(words_count * 1.0 / (time.time() - last_time)),
                    round(100. * correct / ((i + 1) * params.batch_size), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct / total_samples, 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    return train_acc, S


def evaluate(nli_net, valid_iter, optimizer, epoch, train_config, params,
             eval_type='valid', test_folder=None,
             inv_label=None, itos_vocab=None, final_eval=False):
    nli_net.eval()
    correct = 0.
    test_prediction = []
    s1 = []
    s2 = []
    target = []

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))
    total_samples = 0
    for i, batch in enumerate(valid_iter):
        # prepare batch
        s1_batch, s1_len = batch.Sentence1
        s2_batch, s2_len = batch.Sentence2
        s1_batch, s2_batch = Variable(s1_batch.to(device)), Variable(
            s2_batch.to(device))
        tgt_batch = batch.Label.to(device)
        total_samples += s1_batch.size(1)

        # model forward
        output, (s1_out, s2_out) = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

        if eval_type == 'test':
            for b_index in range(len(batch)):
                test_prediction = inv_label[pred[b_index].item()]
                s1 = ' '.join([itos_vocab[idx.item()] for idx in
                               batch.Sentence1[0][:batch.Sentence1[1][b_index],
                               b_index]]).replace('Ġ', ' ')
                s2 = ' '.join([itos_vocab[idx.item()] for idx in
                               batch.Sentence2[0][:batch.Sentence2[1][b_index],
                               b_index]]).replace('Ġ', ' ')
                target = inv_label[batch.Label[b_index]]
                res_file = os.path.join(test_folder, 'samples.txt')
                lock = FileLock(os.path.join(test_folder, 'samples.txt.new.lock'))
                with lock:
                    with open(res_file, 'a') as f:
                        f.write(
                            'S1: ' + s1 + '\n' + 'S2: ' + s2 + '\n' + 'Target: ' + target + '\n' + 'Predicted: '
                            + test_prediction + '\n\n')
                    lock.release()
    # save model
    eval_acc = round(100 * correct / total_samples, 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
        # ex.log_metric('{}_accuracy'.format(eval_type), eval_acc, step=epoch)
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))
        # ex.log_metric('{}_accuracy'.format(eval_type), eval_acc, step=epoch)

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > train_config['val_acc_best']:
            print('saving model at epoch {0}'.format(epoch))
            torch.save(nli_net.state_dict(), params.outputmodelname)
            train_config['val_acc_best'] = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0][
                                                      'lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    train_config['stop_training'] = True
            if 'sgd' not in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                train_config['stop_training'] = train_config['adam_stop']

    return eval_acc, optimizer, train_config


def hyper_evaluate(config):
    print(config)
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-ip-address=')  # ,192.168.2.19
    parser.add_argument('--node-manager-port=')
    parser.add_argument('--object-store-name=')
    parser.add_argument(
        '--raylet-name=')  # /tmp/ray/session_2020-07-15_12-00-45_292745_38156/sockets/raylet
    parser.add_argument('--redis-address=')  # 192.168.2.19:6379
    parser.add_argument('--config-list=', action='store_true')  #
    parser.add_argument('--temp-dir=')  # /tmp/ray
    parser.add_argument('--redis-password=')  # 5241590000000000
    # /////////NLI-Args//////////////
    parser = argparse.ArgumentParser(description='NLI training')
    # paths
    parser.add_argument("--nlipath", type=str, default=config['dataset'],
                        help="NLI data (SNLI or MultiNLI)")
    parser.add_argument("--outputdir", type=str, default='Results/',
                        help="Output directory")
    parser.add_argument("--outputmodelname", type=str, default='model.pickle')
    parser.add_argument("--word_emb_path", type=str,
                        default="dataset/GloVe/glove.840B.300d.txt",
                        help="word embedding file path")

    # training
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dpout_model", type=float, default=0.2,
                        help="encoder dropout")
    parser.add_argument("--dpout_fc", type=float, default=0.2,
                        help="classifier dropout")
    parser.add_argument("--nonlinear_fc", type=float, default=0,
                        help="use nonlinearity in fc")
    parser.add_argument("--optimizer", type=str, default=config["optim"], help="adam")
    parser.add_argument("--lr", type=float, default=0.001, help="lr")  # sgd 0.1
    parser.add_argument("--lrshrink", type=float, default=5,
                        help="shrink factor for sgd")
    parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
    parser.add_argument("--optdecay", type=float, default=config['decay'],
                        help="_C variant decay")
    parser.add_argument("--topC", type=float, default=config['topC'],
                        help="_C variant decay")
    parser.add_argument("--minlr", type=float, default=1e-10, help="minimum lr")
    parser.add_argument("--max_norm", type=float, default=5.,
                        help="max norm (grad clipping)")

    # model
    parser.add_argument("--encoder_type", type=str, default=config['encoder_type'],
                        help="see list of encoders")
    parser.add_argument("--enc_lstm_dim", type=int, default=200,
                        help="encoder nhid dimension")  # 2048
    parser.add_argument("--n_enc_layers", type=int, default=1,
                        help="encoder num layers")
    parser.add_argument("--fc_dim", type=int, default=200, help="nhid of fc layers")
    parser.add_argument("--n_classes", type=int, default=config['num_classes'],
                        help="entailment/neutral/contradiction")
    parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

    # gpu
    parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID")
    parser.add_argument("--seed", type=int, default=config['seed'], help="seed")

    # data
    parser.add_argument("--word_emb_dim", type=int, default=300,
                        help="word embedding dimension")
    parser.add_argument("--word_emb_type", type=str, default='normal',
                        help="word embedding type, either glove or normal")

    params, _ = parser.parse_known_args()

    run_id = params.optimizer + '_exp_seed_{}'.format(params.seed)

    wandb.init(project="critical-gradients-" + config['dataset'], reinit=True)
    wandb.run.name = run_id

    wandb.config.update(config)

    print('Came here')
    exp_folder = os.path.join(params.outputdir, params.nlipath, params.encoder_type,
                              run_id)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    # set proper name
    save_folder_name = os.path.join(exp_folder, 'model')
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)

    test_sample_folder = os.path.join(exp_folder, 'samples_test')
    if not os.path.exists(test_sample_folder):
        os.makedirs(test_sample_folder)
    params.outputmodelname = os.path.join(save_folder_name,
                                          '{}_model.pkl'.format(params.encoder_type))
    # print parameters passed, and all parameters
    print('\ntogrep : {0}\n'.format(sys.argv[1:]))
    print(params)
    pr = vars(params)

    """
    SETUP
    """
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # torch.cuda.manual_seed(params.seed)

    """
    DATA
    """
    train, valid, test, vocab, label_vocab = DataIteratorGlove(
        batch_size=params.batch_size, dataset=params.nlipath,
        max_length=20, prefix='processed_')
    # Train label class balancing
    weights = [2, 2, 2, 0.3, 7, 2, 6]
    # invert the weights by values
    word_vec = getEmbeddingWeights(vocab.itos, params.nlipath)
    print('Embeddings loaded')
    """
    MODEL
    """
    # model config
    config_nli_model = {
        'n_words': len(vocab),
        'word_emb_dim': params.word_emb_dim,
        'enc_lstm_dim': params.enc_lstm_dim,
        'n_enc_layers': params.n_enc_layers,
        'dpout_model': params.dpout_model,
        'dpout_fc': params.dpout_fc,
        'fc_dim': params.fc_dim,
        'bsize': params.batch_size,
        'n_classes': params.n_classes,
        'pool_type': params.pool_type,
        'nonlinear_fc': params.nonlinear_fc,
        'encoder_type': params.encoder_type,
        'use_cuda': True,

    }

    # model
    encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                     'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                     'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
    assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                                 str(encoder_types)
    nli_net = NLINet(config_nli_model, weights=word_vec, device=device)
    print(nli_net)

    weight = torch.FloatTensor(weights)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.size_average = False

    # optimizer

    if params.optimizer == 'sgd':
        optimizer = SGD(nli_net.parameters(), lr=0.1)  # suggested LR for SGD
    elif params.optimizer == 'sgdm':
        optimizer = SGD(nli_net.parameters(), lr=0.1,
                        momentum=0.9)  # suggested LR for SGD
    elif params.optimizer == 'sgd_c':
        optimizer = SGD_C(nli_net.parameters(), lr=0.1, decay=config['decay'],
                          topC=config['topC'])
    elif params.optimizer == 'sgdm_c':
        optimizer = SGD_C(nli_net.parameters(), lr=0.1, momentum=0.9,
                          decay=config['decay'], topC=config['topC'])
    elif params.optimizer == 'adam_c':
        optimizer = Adam_C(nli_net.parameters(), lr=0.001, decay=config['decay'],
                           topC=config['topC'])
    elif params.optimizer == 'adam':
        optimizer = Adam(nli_net.parameters(), lr=0.001)
    elif params.optimizer == 'rmsprop':
        optimizer = RMSprop(nli_net.parameters(), lr=0.001)
    elif params.optimizer == 'rmsprop_c':
        optimizer = RMSprop_C(nli_net.parameters(), lr=0.001, decay=config['decay'],
                              topC=config['topC'])

    # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    # cuda by default
    nli_net.to(device)
    loss_fn.to(device)

    """
    TRAIN
    """
    train_config = {
        'val_acc_best': -1e10,
        'adam_stop': False,
        'stop_training': False,
        'lr': 0.1 if 'sgd' in params.optimizer else None}

    """
    Train model on Natural Language Inference task
    """
    epoch = 0

    while not train_config['stop_training'] and epoch <= params.n_epochs:
        epoch += 1
        train_acc, offline_stats = trainepoch(nli_net, train, optimizer, loss_fn, epoch,
                                              params)
        eval_acc, optimizer, train_config = evaluate(nli_net, valid, optimizer, epoch,
                                                     train_config, params,
                                                     eval_type='valid')
        # scheduler.step()
        off = offline_stats['no'] * 100 / (
                sum([v for v in offline_stats.values()]) + 1e-7)
        on = offline_stats['yes'] * 100 / (
                sum([v for v in offline_stats.values()]) + 1e-7)

        optimizer.resetOfflineStats()
        lock = FileLock(os.path.join(save_folder_name, 'logs.txt' + '.new.lock'))
        with lock:
            with open(os.path.join(save_folder_name, 'logs.txt'), 'a') as f:
                f.write(
                    f'| Epoch: {epoch:03} | Train Acc: {train_acc:.3f} | Val. Acc: {eval_acc:.3f} \n')
            lock.release()

        wandb.log(
            {"Train Accuracy": train_acc, "Val. Accuracy": eval_acc,
             "offline updates": off, "online udpates": on})

    # Run best model on test set.
    nli_net.load_state_dict(torch.load(params.outputmodelname))

    print('\nTEST : Epoch {0}'.format(epoch))
    valid_acc, _, _ = evaluate(nli_net, valid, optimizer, epoch, train_config, params,
                               eval_type='valid',
                               test_folder=None, inv_label=label_vocab.itos,
                               itos_vocab=vocab.itos, final_eval=True)
    test_acc, _, _ = evaluate(nli_net, test, optimizer, epoch, train_config, params,
                              eval_type='test',
                              test_folder=test_sample_folder,
                              inv_label=label_vocab.itos, itos_vocab=vocab.itos,
                              final_eval=True)
    lock = FileLock(os.path.join(save_folder_name, 'logs.txt' + '.new.lock'))
    with lock:
        with open(os.path.join(save_folder_name, 'logs.txt'), 'a') as f:
            f.write(
                f'| Epoch: {epoch:03} | Test Acc: {test_acc:.3f} | Val. Acc: {valid_acc:.3f} \n')
        lock.release()
    # ex.log_asset(file_name=res_file, file_like_object=open(res_file, 'r'))

    return test_acc


# Test set performance

best_hyperparameters = None

PARAM_GRID = list(product(
    ['InferSent'],  # , 'BLSTMprojEncoder', 'ConvNetEncoder'],             # model
    [100, 101, 102],  # , 103, 104], # seeds
    ['snli'],  # dataset
    ['sgd_c'],  # optimizer
    [0.1, 0.01, 0.001, 0.0001],  # lr
    [0.7, 0.9, 0.99],  # , 0.95],  # decay
    [5, 10, 20],  # topC
    ['sum'],  # aggr
    [1.0]  # kappa
))

# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
this_worker = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))

is_slurm = False

for param_ix in range(this_worker, len(PARAM_GRID), N_WORKERS):

    params = PARAM_GRID[param_ix]

    m, s, d, o, dec, t = params
    config = {}
    config['encoder_type'] = m
    config['seed'] = s
    config['dataset'] = d
    config['optim'] = o
    config['decay'] = dec
    config['topC'] = t
    config['num_classes'] = 5

    if is_slurm:
        # run by submitit
        d = datetime.today()
        exp_dir = (
                Path("/home/pamcrae/")
                / "projects"
                / "crit-grad"
                / "nli-task"
                / f"{d.strftime('%Y-%m-%d')}_rand_eval_{config['dataset']}_{config['encoder_type']}"
        )
        exp_dir.mkdir(parents=True, exist_ok=True)
        submitit_logdir = exp_dir / "submitit_logs"
        executor = submitit.AutoExecutor(folder=submitit_logdir)
        executor.update_parameters(
            timeout_min=720,
            slurm_partition="learnfair",
            gpus_per_node=1,
            tasks_per_node=1,
            cpus_per_task=10,
            slurm_mem="",
        )
        job = executor.submit(hyper_evaluate, config)
        print(f"Submitted job {job.job_id}")
    else:
        hyper_evaluate(config)
