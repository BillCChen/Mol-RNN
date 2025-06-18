# python conditional_sample.py # 现在直接运行，无需命令行参数
# 需要在同目录下放置一个名为 'sampling_config.yaml' 的配置文件

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
# 确保此路径指向您的 template_analysis 目录，根据您的实际部署可能需要调整
# 更好的做法是，将 tools 目录作为一个可安装的Python包来处理
sys.path.append("/root/retro_synthesis/template_analysis")
from tools.validation_format import check_format
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

# 移除 get_args() 函数，改为从配置文件加载
def load_config(config_path="conditional_sample_list_config.yaml"):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded from {config_path}")
    return config

def load_vocab(which_vocab, vocab_path):
    if which_vocab == "selfies":
        vocab = SELFIEVocab(vocab_path)
    elif which_vocab == "regex":
        vocab = RegExVocab(vocab_path)
    elif which_vocab == "char":
        vocab = CharVocab(vocab_path)
    else:
        raise ValueError("Wrong vocab name for configuration which_vocab!")
    print("Vocab name: ", vocab.name)
    print("Vocab path: ", vocab_path)
    print("Vocab loaded successfully!")
    return vocab


def conditional_sample(model, vocab, batch_size, initial_string, device, \
                        strategy='probability', temperature=1.0, \
                        return_probability=False, max_length=140):
    """Sample a batch of SMILES from current model starting with an initial string."""
    model.eval()
    # initial_string 需要以^结尾，否则添加一个^的后缀
    if not initial_string.endswith('^'):
        initial_string += '^'
    # Convert initial string to integers
    initial_ints = vocab.tokenize_smiles(initial_string)[1:-1]  # Remove <sos> and <eos>
    initial_length = len(initial_ints)

    # If initial string is too long, we might need to truncate or handle differently
    if initial_length >= max_length:
        print(f"Warning: Initial string length ({initial_length}) exceeds max_length ({max_length}). Truncating initial string.")
        initial_ints = initial_ints[:max_length-1] # ensure at least one token can be generated
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
    probabilities = []  # List to store probabilities per time step

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
            probs = torch.pow(probs, 1.0 / temperature)
            # 重新归一化，确保概率和为1
            probs = probs / torch.sum(probs, dim=-1, keepdim=True)

        # sample
        x = torch.multinomial(probs.squeeze(), 1)
        output.append(x)

        if return_probability:
            # Record the probability of the sampled token for each item in the batch
            batch_probs = []
            for i_batch in range(batch_size):
                sampled_token_idx = x[i_batch].item()
                # 确保索引有效且在概率张量范围内
                if 0 <= sampled_token_idx < probs.shape[-1]:
                    batch_probs.append(probs[i_batch, 0, sampled_token_idx].item())
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
    # 加载配置文件
    config = load_config()

    # 从配置文件获取参数
    global_settings = config.get('global_settings', {})
    sampling_parameters = config.get('sampling_parameters', {})
    output_options = config.get('output_options', {})

    result_base_dir = global_settings.get('result_base_dir', "./results")
    epoch = global_settings.get('epoch', "best")
    input_file_path = global_settings.get('input_file', "input_reactions.txt")
    
    batch_size = sampling_parameters.get('batch_size', 16)
    num_batches = sampling_parameters.get('num_batches', 1)
    temperature = sampling_parameters.get('temperature', 0.8)
    max_sequence_length = sampling_parameters.get('max_sequence_length', 140)

    return_probability = output_options.get('return_probability', False)
    output_subdir_prefix = f"epoch{epoch}_sampling_" + global_settings.get('input_file').split('/')[-1].split('.')[0]

    # 从原始的result_dir加载模型和词汇表的配置
    # 假设 config.yaml 仍然在 result_base_dir 下的某个特定位置
    # 或者，您可以将模型路径和vocab路径直接定义在新的 sampling_config.yaml 中
    # 为了兼容旧逻辑，我们假设 result_base_dir 就是旧的 result_dir
    model_config_path = os.path.join(result_base_dir, "config.yaml")
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model configuration file not found: {model_config_path}. Please ensure it exists in your result_base_dir.")
    with open(model_config_path, 'r') as f:
        model_config = yaml.full_load(f)

    # 检测CPU或GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    # 加载词汇表
    which_vocab, vocab_path = model_config["which_vocab"], model_config["vocab_path"]
    vocab = load_vocab(which_vocab, vocab_path)

    # 加载模型
    epoch_ = "last"
    rnn_config = model_config['rnn_config']
    model = RNN(rnn_config).to(device)
    model_file_path = model_config['out_dir'] + f'trained_model_{epoch_}.pt'
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Trained model not found: {model_file_path}")
    model.load_state_dict(torch.load(
        model_file_path,
        map_location=torch.device(device)))
    print(f"Model loaded from {model_file_path}")
    model.eval()

    # 创建一个根输出目录用于存放所有子文件夹，命名包含时间戳和前缀
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    main_sampling_output_dir = os.path.join(result_base_dir, f"{output_subdir_prefix}")
    os.makedirs(main_sampling_output_dir, exist_ok=True)
    print(f"Sampling results will be saved in: {main_sampling_output_dir}")

    # 读取输入文件并逐行处理
    input_file_full_path = os.path.join(result_base_dir, input_file_path) # 确保从result_base_dir加载输入文件
    if not os.path.exists(input_file_full_path):
        raise FileNotFoundError(f"Input file not found: {input_file_full_path}")
    with open(input_file_full_path, 'r') as f_in:
        all_lines = f_in.readlines()

    # 存储所有行的概率（如果开启了return_probability）
    all_probabilities_data = []

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
        
        current_line_probabilities = [] # 存储当前行的所有批次的概率

        sampled_molecules_for_current_line = []
        valid_begin = time.time()

        for batch_idx in range(num_batches):
            if return_probability:
                sampled_mols, probs_batch = conditional_sample(
                    model, vocab, batch_size, product_smiles, device,
                    strategy='probability', temperature=temperature, 
                    return_probability=True, max_length=max_sequence_length
                )
                current_line_probabilities.append(probs_batch)
            else:
                sampled_mols = conditional_sample(
                    model, vocab, batch_size, product_smiles, device,
                    strategy='probability', temperature=temperature,
                    max_length=max_sequence_length
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
        # 向sampled_molecules_for_current_line第一个元素添加标准答案
        sampled_molecules_for_current_line.insert(0, full_reaction_smiles) # 将标准答案放在第一行
        # 计算有效模板率并生成PDF
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
            # 将当前行的概率数据添加到总列表中
            all_probabilities_data.append({
                'line_index': i,
                'initial_string': product_smiles,
                'probabilities': current_line_probabilities
            })
            # 如果需要，可以在这里为每一行保存一个单独的概率文件
            # 例如保存为JSON:
            # import json
            # with open(os.path.join(current_output_subdir, 'probabilities.json'), 'w') as p_file:
            #     json.dump(current_line_probabilities, p_file, indent=4)
            print(f"Probabilities generated for line {i}. (Not explicitly saved to file by default, see comments for saving option).")

    print("\n--- All input lines processed. ---")

    # # 如果需要，可以在所有任务完成后，将所有行的概率数据保存到一个总文件
    # if return_probability and all_probabilities_data:
    #     import json
    #     total_probabilities_file = os.path.join(main_sampling_output_dir, 'all_samples_probabilities.json')
    #     with open(total_probabilities_file, 'w') as f_out:
    #         json.dump(all_probabilities_data, f_out, indent=4)
    #     print(f"All probabilities data saved to {total_probabilities_file}")