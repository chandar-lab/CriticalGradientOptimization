from model.convnets import CIFAR10CNNModel, CIFAR100CNNModel, CIFAR100CNNModel_noDropOut

import torch.optim as optim

import sys
import torch
import torch.nn as nn
import argparse
import os
import wandb

from itertools import product

import model.cifar as models

sys.path.append('..')
from data_loader import load_data_subset
from optimizers.optim import SGD_C, SGD, Adam_C, Adam, RMSprop, RMSprop_C
from optimizers.optimExperimental import Adam_C_bottom, SGD_C_bottom, AggMo, SGD_C_new, AggMo_C

os.environ["WANDB_API_KEY"] = '90b23c86b7e5108683b793009567e676b1f93888'
os.environ["WANDB_MODE"] = "dryrun"

# commandline arguments

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='../Dataset')
parser.add_argument('--results_path', type=str, default='..')

parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()

data_path = args.data_path
results_path = args.results_path

os.environ["WANDB_DIR"] = results_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, iterator, optimizer, criterion, clip=10):
    ''' Training loop for the model to train.
    Args:
        model: A EncoderDecoder model instance.
        iterator: A DataIterator to read the data.
        optimizer: Optimizer for the model.
        criterion: loss criterion.
        clip: gradient clip value.
    Returns:
        epoch_loss: Average loss of the epoch.
    '''
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    model.train()
    # loss
    epoch_loss = 0
    S = {'yes': 0, 'no': 0}
    for i, batch in enumerate(iterator):
        stats = None

        src = batch[0]
        trg = batch[1]
        src, trg = src.to(device), trg.to(device)
        output = model(src)
        optimizer.zero_grad()
        loss = criterion(output, trg)
        loss.backward()

        # clip the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update the parameters
        optimizer.step()
        stats = optimizer.getOfflineStats()
        if stats:
            for k, v in stats.items():
                S[k] += v
        epoch_loss += loss.item()
    # return the average loss
    return epoch_loss / len(iterator), S


def evaluate(model, iterator, criterion):
    ''' Evaluation loop for the model to evaluate.
    Args:
        model: A Seq2Seq model instance.
        iterator: A DataIterator to read the data.
        criterion: loss criterion.
    Returns:
        epoch_loss: Average loss of the epoch.
    '''
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    model.eval()
    # loss
    epoch_loss = 0
    epoch_correct = 0
    # we don't need to update the model parameters. only forward pass.
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0]
            trg = batch[1]
            src, trg = src.to(device), trg.to(device)
            total += trg.size(0)
            output = model(src)
            loss = criterion(output, trg)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(trg.view_as(pred)).sum()

            epoch_loss += loss.item()
            epoch_correct += correct.item()
    return epoch_loss / len(iterator), 100. * epoch_correct / total


def HyperEvaluate(config):
    """
    Completes training, validation, and testing for one set of hyperparameters
    :param config: dictionary of hyperparameters to train on
    :return: Best validation performance, best test performance/loss
    """
    torch.manual_seed(config['seed'])

    N_EPOCHS = 100  # number of epochs
    BATCH_SIZE = args.batch_size

    if '_C' in config['optim']:
        run_id = "seed_" + str(config['seed']) + '_LR_' + str(config['lr']) + '_topC_' + str(
            config['topC']) + '_decay_' + str(config['decay']) + '_kappa_' + str(config['kappa']) + '_' + config['aggr']
    else:
        run_id = "seed_" + str(config['seed']) + '_LR_' + str(config['lr'])

    #wandb.init(project="Critical-Gradients-" + config['dataset'], reinit=True)
    wandb.init(project="Critical-Gradients-" + config['dataset'] + "_ext", reinit=True)
    #wandb.init(project="Critical-Gradients-EXT", reinit=True)
    wandb.run.name = run_id

    wandb.config.update(config)

    MODEL_SAVE_PATH = os.path.join('../Results', config['dataset'], config['model'] + '_' + config['optim'], 'Model',
                                   run_id)
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    if config['dataset'] == 'cifar10':
        train_iterator, valid_iterator, _, test_iterator, num_classes = load_data_subset(data_aug=0,
                                                                                         batch_size=BATCH_SIZE,
                                                                                         workers=0, dataset='cifar10',
                                                                                         data_target_dir=data_path,
                                                                                         labels_per_class=5000,
                                                                                         valid_labels_per_class=500)
    elif config['dataset'] == 'cifar100':
        train_iterator, valid_iterator, _, test_iterator, num_classes = load_data_subset(data_aug=0,
                                                                                         batch_size=BATCH_SIZE,
                                                                                         workers=0, dataset='cifar100',
                                                                                         data_target_dir=data_path,
                                                                                         labels_per_class=500,
                                                                                         valid_labels_per_class=50)

    # encoder

    if config['model'] == 'convnet':
        if config['dataset'] == 'cifar10':
            model = CIFAR10CNNModel()
        elif config['dataset'] == 'cifar100':
            model = CIFAR100CNNModel()

        optimizer = optim.Adadelta(model.parameters(), lr=config['lr'])
    elif config['model'] == 'convnet_noDropOut':
        model = CIFAR100CNNModel_noDropOut()
    elif config['model'] == 'resnet':
        model = models.__dict__['resnet'](
            num_classes=num_classes,
            depth=110,
            block_name='BasicBlock',
        )

    else:
        print('Error: Model Not There')
        sys.exit(0)

    model = model.to(device)

    if config['optim'] == 'SGD':
        optimizer = SGD(model.parameters(), lr=config['lr'])
    elif config['optim'] == 'SGDM':
        optimizer = SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    elif config['optim'] == 'SGD_C':
        optimizer = SGD_C(model.parameters(), lr=config['lr'], decay=config['decay'], topC=config['topC'],
                          aggr=config['aggr'], critical_test=config['crit_test'], sampling=config['sampling'])
    elif config['optim'] == 'SGD_C_bottom':
        optimizer = SGD_C_bottom(model.parameters(), lr=config['lr'], decay=config['decay'], topC=config['topC'],
                          aggr=config['aggr'], critical_test=config['crit_test'], sampling=config['sampling'])

    elif config['optim'] == 'SGDM_C':
        optimizer = SGD_C(model.parameters(), lr=config['lr'], momentum=config['momentum'], decay=config['decay'], topC=config['topC'],
                          aggr=config['aggr'], critical_test=config['crit_test'],
                          sampling=config['sampling'])
    elif config['optim'] == 'Adam_C':
        optimizer = Adam_C(model.parameters(), lr=config['lr'], decay=config['decay'], kappa=config['kappa'],
                           topC=config['topC'], aggr=config['aggr'], critical_test=config['crit_test'],
                           sampling=config['sampling'], betas = (config['beta1'], config['beta2']))
    elif config['optim'] == 'Adam_C_bottom':
        optimizer = Adam_C_bottom(model.parameters(), lr=config['lr'], decay=config['decay'], kappa=config['kappa'],
                                  topC=config['topC'], aggr=config['aggr'])
    elif config['optim'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=config['lr'], betas = (config['beta1'], config['beta2']))
    elif config['optim'] == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=config['lr'], alpha=config['alpha'])
    elif config['optim'] == 'RMSprop_C':
        optimizer = RMSprop_C(model.parameters(), lr=config['lr'], decay=config['decay'], kappa=config['kappa'],
                              topC=config['topC'], aggr=config['aggr'], critical_test=config['crit_test'],
                              sampling=config['sampling'], alpha=config['alpha'])
    elif config['optim'] == 'SGD_C_new':
            optimizer = SGD_C_new(model.parameters(), lr=config['lr'], decay=config['decay'], topC=config['topC'],
                              aggr=config['aggr'])
    elif config['optim'] == 'AggMo_C':
        optimizer = AggMo_C(model.parameters(), lr=config['lr'], betas=[0, 0.9, 0.99], decay=config['decay'],
                            topC=config['topC'], aggr=config['aggr'])

    criterion = nn.CrossEntropyLoss()

    best_validation_perf = float('-inf')
    best_test_perf = float('-inf')
    best_test_loss = float('inf')

    for epoch in range(N_EPOCHS):

        train_loss, offline_stats = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_perf = evaluate(model, valid_iterator, criterion)
        test_loss, test_perf = evaluate(model, test_iterator, criterion)

        off = offline_stats['no'] * 100 / (sum([v for v in offline_stats.values()]) + 1e-7)
        on = offline_stats['yes'] * 100 / (sum([v for v in offline_stats.values()]) + 1e-7)

        if not config['stats']:
            wandb.log({"Train Loss": train_loss, "Validation Loss": valid_loss, "Validation Accuracy": valid_perf,
                       "Test Loss": test_loss, "Test Accuracy": test_perf, "offline updates": off,
                       "online udpates": on})
        # If triggered, will log stats on the values of the average gc and ct
        else:
            gc_v_gt = optimizer.getAnalysis()
            wandb.log({"Train Loss": train_loss, "Validation Loss": valid_loss, "Validation Accuracy": valid_perf,
                       "Test Loss": test_loss, "Test Accuracy": test_perf, "offline updates": off,
                       "online udpates": on, 'gt': gc_v_gt['gt'] / gc_v_gt['count'],
                       'gc': gc_v_gt['gc'] / gc_v_gt['count'], 'gc_aggr': gc_v_gt['gc_aggr'] / gc_v_gt['count']}
                      )
            optimizer.resetAnalysis()

        optimizer.resetOfflineStats()

        if valid_perf > best_validation_perf:
            best_validation_perf = valid_perf
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'best_model.pt'))

        if test_loss < best_test_loss:
            best_test_loss = test_loss

        if test_perf > best_test_perf:
            best_test_perf = test_perf

    return best_validation_perf, best_test_loss, best_test_perf


PARAM_GRID = list(product(
    ['convnet'],  # model
    [100, 101, 102, 103, 104],  # seeds
    ['cifar10', 'cifar100'],  # dataset
    ['AggMo_C'],  # optimizer
    [0.1, 0.01, 0.001, 0.0001, 0.00001],  # lr
    [0.7, 0.9, 0.99],  # decay
    [5, 10, 20],  # topC
    ['sum'],  # gradsum
    [0], # momentum
    [0], #beta1
    [0.], #beta2
    [0] #alpha
))



# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
this_worker = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))

for param_ix in range(this_worker, len(PARAM_GRID), N_WORKERS):

    params = PARAM_GRID[param_ix]

    m, s, d, o, l, dec, t, ch, p, b1, b2, a = params

    config = {}
    config['model'] = m
    config['seed'] = s
    config['lr'] = l
    config['dataset'] = d
    config['optim'] = o
    config['stats'] = False
    config['crit_test'] = True
    config['sampling'] = None
    config['kappa'] = 1.0
    if "_C" in o:
        config['decay'] = dec
        config['aggr'] = ch
        config['topC'] = t
    else:
        config['decay'] = 0
        config['aggr'] = 'none'
        config['topC'] = 0
    if "SGDM" in o:
        config['momentum'] = p
    else:
        config['momentum'] = 0
    if "Adam" in o:
        config['beta1'] = b1
        config['beta2'] = b2
    else:
        config['beta1'] = 0
        config['beta2'] = 0
    if "RMS" in o:
        config['alpha'] = a
    else:
        config['alpha'] = 0

    val_ppl, test_loss, test_ppl = HyperEvaluate(config)
    wandb.log({'Best Validation Accuracy': val_ppl, 'Best Test Loss': test_loss, 'Best Test Accuracy': test_ppl})
