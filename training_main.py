# Copyright: Wentao Shi, 2021
import yaml
import os
import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from rdkit import Chem
import selfies as sf

from dataloader import dataloader_gen
from dataloader import SELFIEVocab, RegExVocab, CharVocab
from model import RNN

import logging

import sys
sys.path.append("/root/retro_synthesis/template_analysis")
from tools.validation_format import check_format
from tools import get_templates
from tools import result2pdf
# suppress rdkit error
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from rxnmapper import RXNMapper
rxn_mapper = RXNMapper()

def make_vocab(config):
    # load vocab
    which_vocab = config["which_vocab"]
    vocab_path = config["vocab_path"]

    if which_vocab == "selfies":
        return SELFIEVocab(vocab_path)
    elif which_vocab == "regex":
        return RegExVocab(vocab_path)
    elif which_vocab == "char":
        return CharVocab(vocab_path)
    else:
        raise ValueError(
            "Wrong vacab name for configuration which_vocab!"
        )


def sample(model, vocab, batch_size):
    """Sample a batch of SMILES from current model."""
    model.eval()
    # sample
    sampled_ints = model.sample(
        batch_size=batch_size,
        vocab=vocab,
        device=device
    )

    # convert integers back to SMILES
    molecules = []
    sampled_ints = sampled_ints.tolist()
    for ints in sampled_ints:
        molecule = []
        for x in ints:
            if vocab.int2tocken[x] == '<eos>':
                break
            else:
                molecule.append(vocab.int2tocken[x])
        molecules.append("".join(molecule))

    # convert SELFIES back to SMILES
    if vocab.name == 'selfies':
        molecules = [sf.decoder(x) for x in molecules]
    for i in range(10):
        print("sampled raection {}: {}".format(i, molecules[i]))
    return molecules
# sys.path.append("/root/retro_synthesis/reaction_utils")
def compute_valid_template_rate(reaction_smiles_list):
    """compute the percentage of valid SMILES given
    a list SMILES strings"""
    num_valid, num_invalid,num_best = 0, 0, 0
    components = []
    for reaction_smiles in reaction_smiles_list:
        if check_format(reaction_smiles):
            num_valid += 1
            product,templates,reactants = reaction_smiles.split("^")
            components.append((product, templates, reactants))
        else:
            num_invalid += 1
    return num_valid, num_invalid , components

def load_config(config_path):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
def set_logging(logging_level):
    if logging_level == "DEBUG":
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    elif logging_level == "INFO":
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    elif logging_level == "WARNING":
        logging.basicConfig(level=logging.WARNING,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    elif logging_level == "ERROR":
        logging.basicConfig(level=logging.ERROR,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    elif logging_level == "CRITICAL":
        logging.basicConfig(level=logging.CRITICAL,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        raise ValueError(f"Invalid logging level: {logging_level}")
if __name__ == "__main__":
    # detect cpu or gpu
    import argparse
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # force to use cpu
    parser = argparse.ArgumentParser(description='Reaction_space_LLM Training')
    parser.add_argument(
        '--config', type=str, default='./scripts/USPTO50k_1jump_OOD.yaml',
        help='Path to the configuration file.'
    )
    parser.add_argument(
        '--device', type=str
    )
    parser.add_argument(
        '--logging_level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level.'
    )


    args = parser.parse_args()
    logging_level = args.logging_level
    set_logging(logging_level)
    
    device = torch.device(args.device)
    logging.info('device: %s', device)

    config = load_config(args.config)
    # 如果是 debug ，则输出完整的配置
    logging.debug('Configuration: %s', config)
    # directory for results
    out_dir = config['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    trained_model_dir = out_dir + 'trained_model.pt'

    # save the configuration file for future reference
    with open(out_dir + 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # training data
    dataset_dir = config['dataset_dir']
    which_vocab = config['which_vocab']
    vocab_path = config['vocab_path']
    percentage = config['percentage']

    # create dataloader
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    PADDING_IDX = config['rnn_config']['num_embeddings'] - 1
    logging.info('which vocabulary to use: %s', which_vocab)
    dataloader, train_size = dataloader_gen(
        dataset_dir, percentage, which_vocab,
        vocab_path, batch_size, PADDING_IDX,
        shuffle, drop_last=False
    )
    logging.debug('>>> loading training data finished <<<')
    # model and training configuration
    rnn_config = config['rnn_config']
    model = RNN(rnn_config).to(device)
    # 加载 ckpt : /root/retro_synthesis/Mol-RNN/results/run_reaction_0530/trained_model_800.pt
    # model.load_state_dict(torch.load(
    #     '/root/retro_synthesis/Mol-RNN/results/run_reaction_0530/trained_model_200.pt'
    # ))
    pretrained_num = 0
    logging.debug('>>> loading model finished <<<')
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    logging.debug('>>> loading model configuration finished <<<')
    # Making reduction="sum" makes huge difference
    # in valid rate of sampled molecules.
    loss_function = nn.CrossEntropyLoss(reduction='sum')

    # create optimizer
    if config['which_optimizer'] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            weight_decay=weight_decay, amsgrad=True
        )
    elif config['which_optimizer'] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate,
            weight_decay=weight_decay, momentum=0.9
        )
    else:
        raise ValueError(
            "Wrong optimizer! Select between 'adam' and 'sgd'."
        )
    logging.debug('>>> loading optimizer finished <<<')
    # learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=0.5, patience=5,
        cooldown=10, min_lr=0.0001,
        verbose=True
    )

    # vocabulary object used by the sample() function
    vocab = make_vocab(config)

    # train and validation, the results are saved.
    train_losses = []
    valid_rates = []
    best_rates = []
    best_valid_rate = 0
    num_epoch = config['num_epoch']
    logging.debug('>>> loading vocabulary finished <<<')
    logging.info('>>> start training <<<')

    for epoch in range(pretrained_num, 1 + num_epoch):
        print('>>> epoch {}/{} <<<'.format(epoch, num_epoch))
        train_begin = time.time()
        model.train()
        train_loss = 0
        for data, lengths in tqdm(dataloader):
            # the lengths are decreased by 1 because we don't
            # use <eos> for input and we don't need <sos> for
            # output during traning.
            lengths = [length - 1 for length in lengths]

            optimizer.zero_grad()
            data = data.to(device)
            preds = model(data, lengths)

            # The <sos> token is removed before packing, because
            # we don't need <sos> of output during training.
            # the image_captioning project uses the same method
            # which directly feeds the packed sequences to
            # the loss function:
            # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/train.py
            targets = pack_padded_sequence(
                data[:, 1:],
                lengths,
                batch_first=True,
                enforce_sorted=False
            ).data

            loss = loss_function(preds, targets)
            loss.backward()
            optimizer.step()

            # accumulate loss over mini-batches
            train_loss += loss.item()  # * data.size()[0]

        train_losses.append(train_loss / train_size)

        logging.info('epoch {}, train loss: {}.'.format(epoch, train_losses[-1]))

        scheduler.step(train_losses[-1])
        train_end = time.time()
        logging.info('train time: {:.2f} seconds'.format(train_end - train_begin))
        # sample 1024 SMILES each epoch
        valid_begin = time.time()
        sampled_molecules = sample(model, vocab, batch_size=512)

        # print the valid rate each epoch
        num_valid, num_invalid , components = compute_valid_template_rate(sampled_molecules)
        valid_rate = num_valid / (num_valid + num_invalid)
        valid_end = time.time()
        logging.info('valid time: {:.2f} seconds'.format(valid_end - valid_begin))
        logging.info('valid rate: {}'.format(valid_rate))
        valid_rates.append(valid_rate)

        # update the saved model upon best validation loss
        if epoch < 100 and epoch % 5 == 0:
            try:
                num = min(num_valid, 256)
                sub_components = components[:num]
                result2pdf.result_to_img_pdf(sub_components, out_dir + f'epoch_{epoch}_sampled_templates.pdf')
            except Exception as e:
                logging.debug(f"Error in generating PDF for epoch {epoch}: {e}")
                with open(out_dir + f'epoch_{epoch}_sampled_templates_error.txt', 'w') as f:
                    f.write(str(e))
        elif epoch >= 100 and epoch % 50 == 0:
            try:
                num = min(num_valid, 256)
                sub_components = components[:num]
                result2pdf.result_to_img_pdf(sub_components, out_dir + f'epoch_{epoch}_sampled_templates.pdf')
            except Exception as e:
                logging.debug(f"Error in generating PDF for epoch {epoch}: {e}")
                with open(out_dir + f'epoch_{epoch}_sampled_templates_error.txt', 'w') as f:
                    f.write(str(e))
        if epoch % 200 == 0:
            trained_model_dir = out_dir + f'trained_model_{epoch}.pt'
            logging.info('model saved at epoch {}'.format(epoch))
            torch.save(model.state_dict(), trained_model_dir)

    # save train and validation losses
    with open(out_dir + 'loss.yaml', 'w') as f:
        yaml.dump(train_losses, f)
    with open(out_dir + 'valid_rate.yaml', 'w') as f:
        yaml.dump(valid_rates, f)

    # save the final model
    trained_model_dir = out_dir + 'trained_model_last.pt'
    torch.save(model.state_dict(), trained_model_dir)