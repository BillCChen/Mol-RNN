# python conditional_sample.py -result_dir your_output_dir -input_file your_input.txt -temperature 0.8

# Copyright: Wentao Shi, 2021
from dataloader import SELFIEVocab, RegExVocab, CharVocab
from model import RNN
import argparse
import torch
import os
import yaml
import time
import selfies as sf
from tqdm import tqdm
from rdkit import Chem

# suppress rdkit error
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import sys
sys.path.append("/root/retro_synthesis/template_analysis")
from tools.validation_format import check_format
from tools import get_templates
from tools import result2pdf

def compute_valid_template_rate(reaction_smiles_list):
    """compute the percentage of valid SMILES given
    a list SMILES strings"""
    num_valid, num_invalid, num_best = 0, 0, 0
    components = []
    for reaction_smiles in reaction_smiles_list:
        if check_format(reaction_smiles):
            num_valid += 1
            # 注意：这里假设 reaction_smiles 已经是 "product^templates^reactants" 格式
            # 如果传入的是模型生成的单一分子，这行可能会报错或不适用
            # 对于模型生成的，我们只关心其有效性，不一定解析其组件
            try:
                product, templates, reactants = reaction_smiles.split("^")
                components.append((product, templates, reactants))
            except ValueError:
                # 捕获不能split的字符串，通常是单个分子，不加入components用于PDF
                pass
        else:
            num_invalid += 1
    return num_valid, num_invalid, components

def get_args():
    parser = argparse.ArgumentParser("python")
    parser.add_argument("-result_dir",
                        required=True,
                        help="directory of result files including configuration, \
                         loss, trained model, and sampled molecules"
                        )
    parser.add_argument("-batch_size",
                        required=False,
                        default=16,
                        type=int,
                        help="number of samples to generate per prompt_molecule"
                        )
    parser.add_argument("-num_batches",
                        required=False,
                        default=1,
                        type=int,
                        help="number of batches to generate"
                        )
    parser.add_argument("-input_file", # 新增参数
                        required=True,
                        help="Path to the input text file, each line contains product^template^reactant"
                        )
    parser.add_argument("-temperature",
                        required=True,
                        type=float,
                        help="Sampling temperature for controlling randomness"
                        )
    parser.add_argument("--return_probability",
                        action='store_true',
                        help="Whether to return the probability list of each character"
                        )
    return parser.parse_args()

def load_vocab(which_vocab, vocab_path):
    if which_vocab == "selfies":
        vocab = SELFIEVocab(vocab_path)
    elif which_vocab == "regex":
        vocab = RegExVocab(vocab_path)
    elif which_vocab == "char":
        vocab = CharVocab(vocab_path)
    else:
        raise ValueError("Wrong vocab name for configuration which_vocab!")
    print("vocab name: ", vocab.name)
    print("vocab path: ", vocab_path)
    print("vocab loaded successfully!")
    return vocab


def conditional_sample(model, vocab, batch_size, initial_string, device, \
                        strategy='probability',temperature=1.0, \
                            return_probability=False, max_length=140):
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
        # Apply temperature
        if temperature != 1.0:
            # 使用log_softmax和exp进行稳定的温度缩放
            # log_probs = torch.log(probs) / temperature
            # probs = torch.exp(log_probs)
            # probs = probs / torch.sum(probs, dim=-1, keepdim=True)
            probs = torch.pow(probs, 1.0 / temperature)
            probs = probs / torch.sum(probs, dim=-1, keepdim=True)


        # sample
        x = torch.multinomial(probs.squeeze(), 1)
        output.append(x)

        if return_probability:
            # Record the probability of the sampled token
            batch_probs = []
            for i in range(batch_size):
                # 确保索引有效
                sampled_token_idx = x[i].item()
                if 0 <= sampled_token_idx < probs.shape[-1]:
                    batch_probs.append(probs[i, 0, sampled_token_idx].item())
                else:
                    batch_probs.append(0.0) # 无效索引，给0
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

# Note: _convert_to_smiles is not directly used in the current main logic, but kept for completeness
def _convert_to_smiles(tensor, vocab):
    """Convert integer tensor to SMILES strings"""
    molecules = []
    for ints in tensor.tolist():
        molecule = []
        for x in ints:
            if vocab.int2tocken[x] == '<eos>':
                break
            molecule.append(vocab.int2tocken[x])
        molecules.append("".join(molecule))
    
    if vocab.name == 'selfies':
        molecules = [sf.decoder(x) for x in molecules]
    
    return molecules

if __name__ == "__main__":
    args = get_args()
    result_dir = args.result_dir
    batch_size = args.batch_size
    num_batches = args.num_batches
    temperature = args.temperature
    return_probability = args.return_probability
    input_file_path = args.input_file # 获取输入文件路径
    input_file_name = args.input_file.split('/')[-1].split('.')[0] # 获取文件名（不含路径和扩展名）

    # 加载配置文件
    config_dir = os.path.join(result_dir, "config.yaml")
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Configuration file not found: {config_dir}")
    with open(config_dir, 'r') as f:
        config = yaml.full_load(f)

    # 检测CPU或GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    # 加载词汇表
    which_vocab, vocab_path = config["which_vocab"], config["vocab_path"]
    vocab = load_vocab(which_vocab, vocab_path)

    # 加载模型
    epoch_ = "last"
    rnn_config = config['rnn_config']
    model = RNN(rnn_config).to(device)
    model_path = config['out_dir'] + f'trained_model_{epoch_}.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    model.load_state_dict(torch.load(
        model_path,
        map_location=torch.device(device)))
    print(f"Model loaded from {model_path}")
    model.eval()

    # 创建一个根输出目录用于存放所有子文件夹
    main_sampling_output_dir = os.path.join(result_dir, f"sampling_{input_file_name}")
    os.makedirs(main_sampling_output_dir, exist_ok=True)
    print(f"Sampling results will be saved in: {main_sampling_output_dir}")

    # 读取输入文件并逐行处理
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    with open(input_file_path, 'r') as f_in:
        all_lines = f_in.readlines()

    for i, line in enumerate(tqdm(all_lines, desc="Processing input lines")):
        line = line.strip()
        if not line: # 跳过空行
            continue

        parts = line.split("^")
        if len(parts) != 3:
            print(f"Warning: Skipping malformed line {i+1}: '{line}'. Expected 3 parts separated by '^'.")
            continue

        product_smiles = parts[0] # 产物作为 initial_string
        full_reaction_smiles = line # 整行作为 answer

        # 为当前行创建独立的子文件夹
        current_output_subdir = os.path.join(main_sampling_output_dir, str(i))
        os.makedirs(current_output_subdir, exist_ok=True)

        print(f"\n--- Processing line {i} (Product: {product_smiles}) ---")
        print(f"Saving results to: {current_output_subdir}")

        # 设置当前行的输出文件
        output_txt_path = os.path.join(current_output_subdir, 'sampled_molecules.txt')
        output_pdf_path = os.path.join(current_output_subdir, 'sampled_reactions.pdf')

        sampled_molecules_for_current_line = []
        valid_begin = time.time()

        for _ in range(num_batches):
            if return_probability:
                sampled_mols, probabilities = conditional_sample(
                    model, vocab, batch_size, product_smiles, device,
                    strategy='probability', temperature=temperature, return_probability=True
                )
                # 可以选择在这里保存或处理每个批次的probabilities
                # 例如：保存到 JSON 文件或日志
            else:
                sampled_mols = conditional_sample(
                    model, vocab, batch_size, product_smiles, device,
                    strategy='probability', temperature=temperature
                )
            sampled_molecules_for_current_line.extend(sampled_mols)

        valid_end = time.time()
        print(f"Molecules for line {i} sampled in {valid_end - valid_begin:.2f} seconds")

        # 将标准答案添加到生成结果的顶部，并写入文件
        with open(output_txt_path, 'w') as out_file:
            out_file.write(full_reaction_smiles + '\n') # 写入标准答案
            for mol in sampled_molecules_for_current_line:
                out_file.write(mol + '\n')
        
        print(f"Generated {len(sampled_molecules_for_current_line)} samples for line {i}. Results saved to {output_txt_path}")

        # 计算有效模板率并生成PDF
        # 注意：这里传入的是所有生成的molecules，它们可能不是“产物^模板^反应物”格式
        # compute_valid_template_rate 会尝试解析，如果失败则视为无效或仅统计为分子
        # 对于PDF生成，只有符合“产物^模板^反应物”格式的有效项才会被可视化
        num_valid, num_invalid, components_for_pdf = compute_valid_template_rate(sampled_molecules_for_current_line)
        print(f"Number of valid reactions for PDF generation: {len(components_for_pdf)}")
        print(f"Number of valid SMILES strings (total): {num_valid}, Number of invalid SMILES strings: {num_invalid}")

        # 确保PDF文件路径正确，并只使用能解析的组件
        if components_for_pdf: # 只有当有可供PDF可视化的组件时才生成PDF
            result2pdf.result_to_img_pdf(components_for_pdf, output_pdf_path)
            print(f"PDF report generated for line {i} at {output_pdf_path}")
        else:
            print(f"No valid reaction components found for PDF generation for line {i}. Skipping PDF output.")

        if return_probability:
            print(f"Probabilities were generated for line {i} (not explicitly saved to file by default in this example, but can be added).")

    print("\n--- All input lines processed. ---")