"""
Generate the vocabulary of the selfies of the smiles in the dataset
"""
import yaml
import selfies as sf
from tqdm import tqdm


def read_smiles_file(path, percentage):
    with open(path, 'r') as f:
        smiles = [line.strip("\n") for line in f.readlines()]
    num_data = len(smiles)
    return smiles[0:int(num_data * percentage)]


if __name__ == "__main__":
    # dataset_path = "../dataset/chembl28-cleaned.smi"
    dataset_path = "/root/reaction_data/USPTO/uspto_50k_smiles_data.smi"
    output_vocab = "./uspto_50k_selfies_vocab.yaml"

    smiles = read_smiles_file(dataset_path, 1)
    # smiles = ["NC(=O)c1nc2cccc([N+](=O)[O-])c2o1","*","Cc1cc(Cl)ccc1C(C/C(=N\O)c1ccc(=O)n(C)c1)c1ccc(OS(C)(=O)=O)c(F)c1"]
    selfies = []
    for x in tqdm(smiles):
        if x == "*":
            continue
        x = sf.encoder(x)
        if x is not None:
            selfies.append(x)

    print('getting alphabet from selfies...')
    vocab = sf.get_alphabet_from_selfies(selfies)

    vocab_dict = {}
    for i, token in enumerate(vocab):
        vocab_dict[token] = i
    i += 1
    vocab_dict['*'] = i
    i += 1
    vocab_dict['^'] = i
    i += 1
    vocab_dict['.'] = i
    i += 1
    vocab_dict['<eos>'] = i
    i += 1
    vocab_dict['<sos>'] = i
    i += 1
    vocab_dict['<pad>'] = i

    with open(output_vocab, 'w') as f:
        yaml.dump(vocab_dict, f)
