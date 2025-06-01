# python conditional_sample.py -result_dir your_output_dir -initial_string "your_initial_string" --return_probability

# Copyright: Wentao Shi, 2021
from dataloader import SELFIEVocab, RegExVocab, CharVocab
from model import RNN
import argparse
import torch
import yaml
import selfies as sf
from tqdm import tqdm
from rdkit import Chem

# suppress rdkit error
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')


def get_args():
    parser = argparse.ArgumentParser("python")
    parser.add_argument("-result_dir",
                        required=True,
                        help="directory of result files including configuration, \
                         loss, trained model, and sampled molecules"
                        )
    parser.add_argument("-batch_size",
                        required=False,
                        default=128,
                        help="number of samples to generate per mini-batch"
                        )
    parser.add_argument("-num_batches",
                        required=False,
                        default=1,
                        help="number of batches to generate"
                        )
    parser.add_argument("-initial_string",
                        required=True,
                        help="Initial string to start sampling from"
                        )
    parser.add_argument("--return_probability",
                        action='store_true',
                        help="Whether to return the probability list of each character"
                        )
    return parser.parse_args()


def conditional_sample(model, vocab, batch_size, initial_string, device, return_probability=False, max_length=140):
    """Sample a batch of SMILES from current model starting with an initial string."""
    model.eval()

    # Convert initial string to integers
    initial_ints = vocab.tokenize_smiles(initial_string)[1:-1]  # Remove <sos> and <eos>
    initial_length = len(initial_ints)

    # Create a tensor of shape [batch_size, initial_length]
    initial_tensor = torch.tensor(initial_ints, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)

    # Pass the initial tensor through the model to get the hidden state
    x = model.embedding_layer(initial_tensor)
    x, hidden = model.rnn(x)
    x = model.linear(x[:, -1:])
    x = torch.softmax(x, dim=-1)
    x = torch.multinomial(x.squeeze(), 1)

    output = [x]
    probabilities = []  # List to store probabilities

    # a tensor to indicate if the <eos> token is found
    # for all data in the mini-batch
    finish = torch.zeros(batch_size, dtype=torch.bool).to(device)

    # sample until every sequence in the mini-batch
    # has <eos> token or reaches max_length
    for _ in range(max_length - initial_length):
        # forward rnn
        x = model.embedding_layer(x)
        x, hidden = model.rnn(x, hidden)
        x = model.linear(x)
        probs = torch.softmax(x, dim=-1)  # Get probabilities

        # sample
        x = torch.multinomial(probs.squeeze(), 1)
        output.append(x)

        if return_probability:
            # Record the probability of the sampled token
            batch_probs = []
            for i in range(batch_size):
                batch_probs.append(probs[i, 0, x[i].item()].item())
            probabilities.append(batch_probs)

        # terminate if <eos> is found for every data
        eos_sampled = (x == vocab.vocab['<eos>']).data
        finish = torch.logical_or(finish, eos_sampled.squeeze())
        if torch.all(finish):
            break

    output = torch.cat([initial_tensor, torch.cat(output, -1)], -1)

    # convert integers back to SMILES
    molecules = []
    output = output.tolist()
    for ints in output:
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

    if return_probability:
        return molecules, probabilities
    else:
        return molecules


if __name__ == "__main__":
    args = get_args()
    result_dir = args.result_dir
    batch_size = int(args.batch_size)
    num_batches = int(args.num_batches)
    initial_string = args.initial_string
    return_probability = args.return_probability

    # load the configuartion file in output
    config_dir = result_dir + "config.yaml"
    with open(config_dir, 'r') as f:
        config = yaml.full_load(f)

    # detect cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    # load vocab
    which_vocab, vocab_path = config["which_vocab"], config["vocab_path"]

    if which_vocab == "selfies":
        vocab = SELFIEVocab(vocab_path)
    elif which_vocab == "regex":
        vocab = RegExVocab(vocab_path)
    elif which_vocab == "char":
        vocab = CharVocab(vocab_path)
    else:
        raise ValueError("Wrong vacab name for configuration which_vocab!")

    # load model
    rnn_config = config['rnn_config']
    model = RNN(rnn_config).to(device)
    model.load_state_dict(torch.load(
        config['out_dir'] + 'trained_model.pt',
        map_location=torch.device(device)))
    model.eval()

    # sample, filter out invalid molecules, and save the valid molecules
    out_file = open(result_dir + "sampled_molecules.out", "w")
    num_valid, num_invalid = 0, 0
    for _ in tqdm(range(num_batches)):
        if return_probability:
            molecules, probabilities = conditional_sample(model, vocab, batch_size, initial_string, device, return_probability)
        else:
            molecules = conditional_sample(model, vocab, batch_size, initial_string, device)

        # save the valid sampled SMILES to output file,
        for smiles in molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    num_invalid += 1
                else:
                    num_valid += 1
                    out_file.write(smiles + '\n')
            except:
                num_invalid += 1
                pass

    # and compute the valid rate
    print("sampled {} valid SMILES out of {}, success rate: {}".format(
        num_valid, num_valid + num_invalid, num_valid / (num_valid + num_invalid))
    )

    if return_probability:
        # You can further process or save the probabilities here
        print("Probabilities:", probabilities)