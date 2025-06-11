import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter # 用于更丰富的日志记录 (可选)
import yaml
import os
import logging
from datetime import datetime
import numpy as np # 用于设置随机种子
from scipy.stats import spearmanr, pearsonr # 导入斯皮尔曼和皮尔逊相关性

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, templates_list, is_train=True):
        self.smiles_list = smiles_list
        self.templates_list = templates_list
        self.is_train = is_train

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        templates = self.templates_list[idx]

        if self.is_train:
            # For training, randomly sample templates: keep 5, discard 2
            # Ensure we have at least 7 templates to sample from
            if len(templates) < 7:
                # Handle cases where there are fewer than 7 templates if necessary
                # For this problem, it's stated sublist length is 7, so this may not be needed.
                # If it's possible, you might want to pad or handle differently.
                # For now, we'll assume len(templates) is always 7
                sampled_templates_with_indices = [(t, i) for i, t in enumerate(templates)]
            else:
                # Sample 5 templates from the 7 available
                sampled_indices = random.sample(range(7), 5)
                # Sort indices to maintain the original order for ranking loss if needed
                sampled_indices.sort()
                sampled_templates_with_indices = [(templates[i], i) for i in sampled_indices]
        else:
            # For validation, return all templates with their original indices
            sampled_templates_with_indices = [(template, i) for i, template in enumerate(templates)]

        return smiles, sampled_templates_with_indices

class LoadData:
    def __init__(self, base_dir:str,dataset_category: str):
        """
        初始化 LoadData 类。
        根据数据集类别读取对应的pkl文件。

        Args:
            dataset_category (str): 数据集的类别，例如 'USPTO50k', 'ChORISO', 'USPTO150k' 等。
                                    对应的pkl文件名为 f"{dataset_category}_dataset.pkl"。
        """
        self.base_dir = base_dir
        self.dataset_category = dataset_category
        self.smiles_data = []
        self.templates_data = []
        self._load_dataset() # 加载数据集
        self._load_global_encodings() # 加载全局编码

    def _load_dataset(self):
        """
        根据数据集类别读取对应的pkl文件。
        pkl文件存储的是一个元组，元组的第一个元素是SMILES列表，第二个元素是逆合成模板列表。
        """
        file_path = f"{self.base_dir}/{self.dataset_category}_templates.pkl"
        with open(file_path, 'rb') as f:
            data_tuple = pickle.load(f)
            self.train_smiles_data = data_tuple[0][0]
            self.train_templates_data = data_tuple[0][1]
            self.valid_smiles_data = data_tuple[1][0]
            self.valid_templates_data = data_tuple[1][1]

    def _load_global_encodings(self):
        """
        加载包含所有 SMILES 和模板编码的字典。
        假设 pkl 文件存储的是一个字典，其键是 SMILES 或模板字符串，值是其对应的 numpy 数组编码。
        """
        self.global_encoding_path = f"{self.base_dir}/{self.dataset_category}_global_encodings.pkl"
        with open(self.global_encoding_path, 'rb') as f:
            global_data = pickle.load(f)
            if isinstance(global_data, dict):
                # 假设字典结构是 { 'smiles_encodings': {smi: arr}, 'template_encodings': {tpl: arr} }
                # 或者更直接的，如果 global_data 就是包含所有编码的大字典
                if 'smiles_encodings' in global_data and 'templates_encodings' in global_data:
                    self.smiles_encodings = global_data['smiles_encodings']
                    self.template_encodings = global_data['templates_encodings']
                else:
                    raise ValueError(f"全局编码文件 {self.global_encoding_path} 的格式不正确。")
            else:
                raise ValueError(f"全局编码文件 {self.global_encoding_path} 应该是一个字典，但实际类型为 {type(global_data)}。")
    def _get_smiles_encoding(self, smiles):
        """从编码字典中获取 SMILES 的 ECFP 编码。"""
        encoding = self.smiles_encodings.get(smiles)
        if encoding is None:
            raise KeyError(f"SMILES 编码 '{smiles}' 在 {self.global_encoding_path} 中未找到。")
        return torch.from_numpy(encoding).float() # 转换为 FloatTensor

    def _get_template_encoding(self, template_str):
        """从编码字典中获取模板的 DRFP 编码。"""
        encoding = self.template_encodings.get(template_str)
        if encoding is None:
            raise KeyError(f"模板编码 '{template_str}' 在 {self.global_encoding_path} 中未找到。")
        return torch.from_numpy(encoding).float() # 转换为 FloatTensor
    @staticmethod
    def _collate_fn_train_wrapper(smiles_encodings_dict, template_encodings_dict):
        """
        训练数据加载器的collate function 工厂函数。
        创建一个闭包，以便在 collate_fn 中访问编码字典。
        """
        def collate_fn(batch):
            batch_smiles_ecfps = []
            batch_templates_drfps_nested = [] # 列表的列表，每个子列表包含该分子采样的 DRFP 张量
            batch_template_indices_nested = []

            for smiles, sampled_templates_with_indices in batch:
                # 获取 SMILES 的 ECFP 编码
                smiles_ecfp = smiles_encodings_dict.get(smiles)
                if smiles_ecfp is None:
                    raise KeyError(f"SMILES '{smiles}' 的 ECFP 编码未找到。")
                batch_smiles_ecfps.append(torch.from_numpy(smiles_ecfp).float())

                templates_drfps_for_mol = []
                indices_for_mol = []
                for template_str, index in sampled_templates_with_indices:
                    # 获取模板的 DRFP 编码
                    template_drfp = template_encodings_dict.get(template_str)
                    if template_drfp is None:
                        raise KeyError(f"模板 '{template_str}' 的 DRFP 编码未找到。")
                    templates_drfps_for_mol.append(torch.from_numpy(template_drfp).float())
                    indices_for_mol.append(index)
                
                batch_templates_drfps_nested.append(templates_drfps_for_mol)
                batch_template_indices_nested.append(indices_for_mol)
            
            # 将 SMILES ECFP 列表堆叠成一个张量
            batch_smiles_ecfps_tensor = torch.stack(batch_smiles_ecfps)

            # batch_templates_drfps_nested 仍然是列表的列表，这样才能与 MoleculeTemplateNet 的 contrastive_loss 对应
            return batch_smiles_ecfps_tensor, batch_templates_drfps_nested, batch_template_indices_nested
        return collate_fn


    @staticmethod
    def _collate_fn_valid_wrapper(smiles_encodings_dict, template_encodings_dict):
        """
        验证数据加载器的collate function 工厂函数。
        创建一个闭包，以便在 collate_fn 中访问编码字典。
        """
        def collate_fn(batch):
            batch_smiles_ecfps = []
            batch_templates_drfps_nested = []
            batch_template_indices_nested = []

            for smiles, templates_with_indices in batch:
                smiles_ecfp = smiles_encodings_dict.get(smiles)
                if smiles_ecfp is None:
                    raise KeyError(f"SMILES '{smiles}' 的 ECFP 编码未找到。")
                batch_smiles_ecfps.append(torch.from_numpy(smiles_ecfp).float())

                templates_drfps_for_mol = []
                indices_for_mol = []
                for template_str, index in templates_with_indices:
                    template_drfp = template_encodings_dict.get(template_str)
                    if template_drfp is None:
                        raise KeyError(f"模板 '{template_str}' 的 DRFP 编码未找到。")
                    templates_drfps_for_mol.append(torch.from_numpy(template_drfp).float())
                    indices_for_mol.append(index)
                
                batch_templates_drfps_nested.append(templates_drfps_for_mol)
                batch_template_indices_nested.append(indices_for_mol)
            
            batch_smiles_ecfps_tensor = torch.stack(batch_smiles_ecfps)

            return batch_smiles_ecfps_tensor, batch_templates_drfps_nested, batch_template_indices_nested
        return collate_fn

    def get_dataloaders(self, batch_size: int, num_workers: int = 0):
        """
        生成训练和验证数据加载器。

        Args:
            batch_size (int): 每个batch的分子数量。
            num_workers (int): 用于数据加载的子进程数量。

        Returns:
            tuple: 包含 (train_loader, valid_loader) 的元组。
        """
        train_dataset = MoleculeDataset(self.train_smiles_data, self.train_templates_data, is_train=True)
        valid_dataset = MoleculeDataset(self.valid_smiles_data, self.valid_templates_data, is_train=False)
        # 在这里传入编码字典给 collate_fn 的工厂函数
        train_collate_fn = self._collate_fn_train_wrapper(self.smiles_encodings, self.template_encodings)
        valid_collate_fn = self._collate_fn_valid_wrapper(self.smiles_encodings, self.template_encodings)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn, # 使用闭包生成的 collate_fn
            pin_memory=True
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=valid_collate_fn, # 使用闭包生成的 collate_fn
            pin_memory=True
        )

        return train_loader, valid_loader

class MoleculeTemplateNet(nn.Module):
    def __init__(self, model_config):
        super(MoleculeTemplateNet, self).__init__()
        self.model_config = model_config
        # 分子 ECFP 编码器
        self.smiles_encoder = nn.Sequential(
            nn.Linear(model_config.get('ecfp_dim', 1024), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, model_config.get('projection_dim', 128))
        )

        # 模板 DRFP 编码器
        self.template_encoder = nn.Sequential(
            nn.Linear(model_config.get('drfp_dim', 256), 256),
            nn.ReLU(),
            nn.Linear(256, model_config.get('projection_dim', 128))
        )

        # 预测器
        self.predictor = nn.Sequential(
            nn.Linear(model_config.get('projection_dim', 128) * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.device = "cpu"

    def to(self, device):
        """
        将模型移动到指定设备。
        Args:
            device (torch.device): 目标设备。
        """
        super(MoleculeTemplateNet, self).to(device)
        self.device = device
        self.smiles_encoder.to(device)
        self.template_encoder.to(device)
        self.predictor.to(device)
        return self

    def encode_smiles(self, ecfp_vector):
        """
        将分子的 ECFP 向量编码到低维空间。
        Args:
            ecfp_vector (torch.Tensor): 分子的 ECFP 向量。期望形状: (batch_size, ecfp_dim)
        Returns:
            torch.Tensor: 投影后的向量。期望形状: (batch_size, projection_dim)
        """
        return self.smiles_encoder(ecfp_vector)

    def encode_template(self, drfp_vector):
        """
        将模板的 DRFP 向量编码到低维空间。
        Args:
            drfp_vector (torch.Tensor): 模板的 DRFP 向量。期望形状: (batch_size, drfp_dim)
        Returns:
            torch.Tensor: 投影后的向量。期望形状: (batch_size, projection_dim)
        """
        return self.template_encoder(drfp_vector)

    def predict_relevance(self, smiles_embedding, template_embedding):
        """
        预测分子和模板之间的相关性分数。
        Args:
            smiles_embedding (torch.Tensor): 分子嵌入。期望形状: (batch_size, projection_dim)
            template_embedding (torch.Tensor): 模板嵌入。期望形状: (batch_size, projection_dim)
        Returns:
            torch.Tensor: 经过 Sigmoid 输出的标量相关性分数。期望形状: (batch_size, 1)
        """
        combined_embedding = torch.cat((smiles_embedding, template_embedding), dim=-1)
        return self.predictor(combined_embedding)

    def forward(self, ecfp_batch, drfp_batch):
        """
        前向传播，用于编码和预测相关性。
        Args:
            ecfp_batch (torch.Tensor): 批量的 ECFP 向量。
            drfp_batch (torch.Tensor): 批量的 DRFP 向量。
        Returns:
            tuple: (molecule_embeddings, template_embeddings, relevance_scores)
        """
        molecule_embeddings = self.encode_smiles(ecfp_batch)
        template_embeddings = self.encode_template(drfp_batch)
        relevance_scores = self.predict_relevance(molecule_embeddings, template_embeddings)
        return molecule_embeddings, template_embeddings, relevance_scores

    def contrastive_loss(self, smiles_ecfps, all_template_drfps_nested, template_indices_nested):
        """
        计算对比学习损失，考虑批次内正负样本。
        批次中的每个分子，其自身的 5 个模板是正例，其他分子的 5 个模板是相对于它的负例。

        Args:
            smiles_ecfps (torch.Tensor): 批量的分子 ECFP 向量。形状: (batch_size, ecfp_dim)
            all_template_drfps_nested (list of list of torch.Tensor): 批量的模板 DRFP 向量列表。
                                                                    每个内部列表对应一个分子，包含其采样的模板。
                                                                    形状: (batch_size, num_sampled_templates, drfp_dim)
            template_indices_nested (list of list of int): 批量的模板原始序号列表。
                                                          每个内部列表对应一个分子，包含其采样模板的原始序号。
                                                          形状: (batch_size, num_sampled_templates)
        Returns:
            tuple: (total_loss, contrast_loss_part1, ranking_loss_part2)
            torch.Tensor: 总对比学习损失。
        """
        batch_size = smiles_ecfps.size(0)
        num_sampled_templates = len(all_template_drfps_nested[0]) # 假设每个分子采样的模板数量相同

        margin_pos_neg = self.model_config.get('margin_pos_neg', 0.1) # 用于正负样本对比的 margin
        margin_ranking = self.model_config.get('margin_ranking', 0.05) # 用于排序损失的 margin
        
        # 编码所有分子和所有模板
        # 将 all_template_drfps_nested 从 list of list of Tensor 展平为单个 Tensor
        # (batch_size * num_sampled_templates, drfp_dim)
        flat_template_drfps = torch.cat([torch.stack(t_list) for t_list in all_template_drfps_nested], dim=0)
        
        mol_embeddings = self.encode_smiles(smiles_ecfps.to(self.device)) # (batch_size, projection_dim)
        template_embeddings_flat = self.encode_template(flat_template_drfps.to(self.device)) # (batch_size * num_sampled_templates, projection_dim)

        total_loss = 0.0
        contrast_loss_sum = 0.0
        ranking_loss_sum = 0.0
        
        # Lists to store relevance scores and original indices for correlation calculation
        all_predicted_scores = []
        all_original_indices = []

        # 遍历批次中的每一个分子作为锚点 (anchor)
        for anchor_idx in range(batch_size):
            anchor_mol_embedding = mol_embeddings[anchor_idx].unsqueeze(0) # (1, projection_dim)

            # 获取锚点分子的所有正例模板及其嵌入
            # 这 5 个模板都是相对于当前分子的“正例”
            positive_templates_drfps = all_template_drfps_nested[anchor_idx]
            # (num_sampled_templates, drfp_dim)
            positive_templates_embeddings = template_embeddings_flat[
                anchor_idx * num_sampled_templates : (anchor_idx + 1) * num_sampled_templates
            ] # (num_sampled_templates, projection_dim)

            # 获取正例模板的原始序号，用于排序损失
            positive_template_original_indices = template_indices_nested[anchor_idx]

            # 计算锚点分子与自身所有正例模板的相关性分数
            # (num_sampled_templates, 1) -> (num_sampled_templates)
            positive_scores = self.predict_relevance(
                anchor_mol_embedding.repeat(num_sampled_templates, 1),
                positive_templates_embeddings
            ).squeeze(-1)

            # --- 第一部分损失: 正负例对比损失 ---
            # 收集所有负例模板的嵌入
            # ( (batch_size - 1) * num_sampled_templates, projection_dim )
            negative_templates_embeddings = []
            for neg_mol_idx in range(batch_size):
                if neg_mol_idx == anchor_idx: # 跳过自身
                    continue
                neg_templates_embeddings_for_mol = template_embeddings_flat[
                    neg_mol_idx * num_sampled_templates : (neg_mol_idx + 1) * num_sampled_templates
                ]
                negative_templates_embeddings.append(neg_templates_embeddings_for_mol)
            
            contrast_loss_part1 = 0.0
            if len(negative_templates_embeddings) > 0: # 确保有负例
                negative_templates_embeddings = torch.cat(negative_templates_embeddings, dim=0)
                
                # 计算锚点分子与所有负例模板的相关性分数
                # ( (batch_size - 1) * num_sampled_templates, 1 ) -> ( (batch_size - 1) * num_sampled_templates )
                negative_scores = self.predict_relevance(
                    anchor_mol_embedding.repeat(negative_templates_embeddings.size(0), 1),
                    negative_templates_embeddings
                ).squeeze(-1)

                for pos_score in positive_scores:
                    contrast_loss_part1 += torch.mean(F.relu(negative_scores - pos_score + margin_pos_neg))
            
            contrast_loss_sum += contrast_loss_part1


            # --- 第二部分损失: 排序损失 ---
            ranking_loss_part2 = 0.0
            if num_sampled_templates > 1:
                # 根据原始模板序号对正例模板的分数进行排序
                # 确保 original_indices 和 scores 的对应关系正确
                # 创建一个包含 (index, score) 元组的列表，然后按 index 排序
                pairs_for_sorting = []
                for i in range(num_sampled_templates):
                    pairs_for_sorting.append((positive_template_original_indices[i], positive_scores[i]))
                
                sorted_pairs = sorted(pairs_for_sorting, key=lambda x: x[0])
                sorted_scores = torch.stack([pair[1] for pair in sorted_pairs])

                # 遍历所有可能的有序对 (k, l)，其中 k 的原始序号小于 l 的原始序号
                for k_idx in range(num_sampled_templates):
                    for l_idx in range(k_idx + 1, num_sampled_templates):
                        # 确保 sorted_scores[k_idx] 小于 sorted_scores[l_idx]
                        ranking_loss_part2 += F.relu(sorted_scores[k_idx] - sorted_scores[l_idx] + margin_ranking)
            
            ranking_loss_sum += ranking_loss_part2

            # --- 收集用于相关性计算的数据（仅在验证阶段需要，但为了统一计算在这里收集） ---
            # For validation, positive_template_original_indices will contain indices 0 to 6
            # positive_scores will contain predicted scores for these 7 templates.
            all_predicted_scores.append(positive_scores.detach().cpu().numpy())
            all_original_indices.append(np.array(positive_template_original_indices))


        total_loss = (contrast_loss_sum + ranking_loss_sum) / batch_size
        return total_loss, contrast_loss_sum / batch_size, ranking_loss_sum / batch_size, \
               all_predicted_scores, all_original_indices

# 配置日志
def setup_logging(log_dir, project_name):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = os.path.join(log_dir, f"{project_name}_{timestamp}.log")
    
    os.makedirs(log_dir, exist_ok=True) # 确保日志目录存在

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename), # 输出到文件
            logging.StreamHandler() # 输出到控制台
        ]
    )
    return logging.getLogger(__name__) , timestamp

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(config_path="scripts_yml/USPTO50k_OOD_scoring.yaml"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 设置可见的 GPU 设备
    # 1. 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 从配置中获取路径信息
    log_dir = config['general']['log_dir']
    ckpt_dir = config['general']['ckpt_dir']
    project_name = config['general']['project_name']
    
    # 确保检查点目录存在
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. 把必要的训练日志记录在指定目录下的 log 文件中
    logger , timestamp = setup_logging(log_dir, project_name)
    logger.info("--- 开始训练 ---")
    logger.info(f"加载配置: {config_path}")
    logger.info(f"当前配置:\n{yaml.dump(config, indent=2)}")

    # 设置随机种子
    set_seed(config['general']['seed'])
    logger.info(f"设置随机种子: {config['general']['seed']}")

    # 设置设备
    device = torch.device(config['general']['device'] if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 2. 数据加载
    logger.info("初始化数据加载器...")
    data_manager = LoadData(
        base_dir=config['data']['base_dir'],
        dataset_category=config['data']['dataset_category'])
    train_loader, valid_loader = data_manager.get_dataloaders(
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    logger.info(f"训练集 batch 数量: {len(train_loader)}")
    logger.info(f"验证集 batch 数量: {len(valid_loader)}")

    # 3. 初始化模型、优化器和学习率调度器
    logger.info("初始化模型...")
    model = MoleculeTemplateNet(config['model']).to(device)
    logger.info(f"模型结构:\n{model}")

    optimizer_name = config['train']['optimizer']
    learning_rate = config['train']['learning_rate']
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    logger.info(f"使用优化器: {optimizer_name}, 学习率: {learning_rate}")

    scheduler_name = config['train']['scheduler']
    scheduler_params = config['train'].get('scheduler_params', {})
    if scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_name == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_name == "None":
        scheduler = None
    else:
        raise ValueError(f"不支持的学习率调度器: {scheduler_name}")
    logger.info(f"使用学习率调度器: {scheduler_name} (参数: {scheduler_params})")


    # 4. 训练循环
    epochs = config['train']['epochs']
    log_interval = config['freq']['log_interval']
    val_interval = config['freq']['val_interval']
    ckpt_interval = config['freq']['ckpt_interval']

    best_val_loss = float('inf') # 用于保存最佳模型

    writer = SummaryWriter(os.path.join(log_dir, "runs", project_name + "_" + timestamp))
    logger.info(f"TensorBoard 日志路径: {writer.log_dir}")


    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        total_train_loss = 0
        total_train_contrast_loss = 0
        total_train_ranking_loss = 0
        logger.info(f"\n--- Epoch {epoch}/{epochs} (训练阶段) ---")
        for batch_idx, (smiles_ecfps, all_template_drfps_nested, template_indices_nested) in enumerate(train_loader):
            # 将数据移动到设备
            smiles_ecfps = smiles_ecfps.to(device)
            # all_template_drfps_nested 和 template_indices_nested 已经是列表，
            # 它们内部的 Tensor 会在 contrastive_loss 中被移动到设备
            
            optimizer.zero_grad()
            
            # 计算损失
            loss, contrast_loss, ranking_loss, _, _ = model.contrastive_loss(
                smiles_ecfps, 
                all_template_drfps_nested, 
                template_indices_nested
            )
            
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_contrast_loss += contrast_loss.item()
            total_train_ranking_loss += ranking_loss.item()

            if (batch_idx + 1) % log_interval == 0:
                avg_batch_loss = total_train_loss / (batch_idx + 1)
                avg_batch_contrast_loss = total_train_contrast_loss / (batch_idx + 1)
                avg_batch_ranking_loss = total_train_ranking_loss / (batch_idx + 1)
                logger.info(f"  Epoch {epoch}, Batch {batch_idx+1}/{len(train_loader)}, Total Loss: {avg_batch_loss:.4f}, Contrast Loss: {avg_batch_contrast_loss:.4f}, Ranking Loss: {avg_batch_ranking_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                writer.add_scalar('Loss/train_batch_total', avg_batch_loss, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/train_batch_contrast', avg_batch_contrast_loss, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/train_batch_ranking', avg_batch_ranking_loss, epoch * len(train_loader) + batch_idx)
        
        avg_train_loss_epoch = total_train_loss / len(train_loader)
        avg_train_contrast_loss_epoch = total_train_contrast_loss / len(train_loader)
        avg_train_ranking_loss_epoch = total_train_ranking_loss / len(train_loader)
        logger.info(f"Epoch {epoch} 训练平均总损失: {avg_train_loss_epoch:.4f}")
        logger.info(f"Epoch {epoch} 训练平均对比损失: {avg_train_contrast_loss_epoch:.4f}")
        logger.info(f"Epoch {epoch} 训练平均排序损失: {avg_train_ranking_loss_epoch:.4f}")
        writer.add_scalar('Loss/train_epoch_total', avg_train_loss_epoch, epoch)
        writer.add_scalar('Loss/train_epoch_contrast', avg_train_contrast_loss_epoch, epoch)
        writer.add_scalar('Loss/train_epoch_ranking', avg_train_ranking_loss_epoch, epoch)


        if scheduler:
            scheduler.step() # 学习率调度器更新

        # 验证阶段
        if epoch % val_interval == 0:
            model.eval()
            total_val_loss = 0
            total_val_contrast_loss = 0
            total_val_ranking_loss = 0
            all_spearman_coeffs = []
            all_pearson_coeffs = []

            logger.info(f"--- Epoch {epoch}/{epochs} (验证阶段) ---")
            with torch.no_grad(): # 验证阶段不需要计算梯度
                for batch_idx_val, (smiles_ecfps_val, all_template_drfps_nested_val, template_indices_nested_val) in enumerate(valid_loader):
                    smiles_ecfps_val = smiles_ecfps_val.to(device)
                    
                    val_loss, val_contrast_loss, val_ranking_loss, \
                    batch_predicted_scores_list, batch_original_indices_list = model.contrastive_loss(
                        smiles_ecfps_val, 
                        all_template_drfps_nested_val, 
                        template_indices_nested_val
                    )
                    
                    total_val_loss += val_loss.item()
                    total_val_contrast_loss += val_contrast_loss.item()
                    total_val_ranking_loss += val_ranking_loss.item()

                    # 计算每个分子的相关性
                    for i in range(len(batch_predicted_scores_list)):
                        predicted_scores = batch_predicted_scores_list[i]
                        original_indices = batch_original_indices_list[i]
                        
                        # 确保有足够的数据点来计算相关性
                        if len(predicted_scores) > 1:
                            # 斯皮尔曼相关性
                            try:
                                spearman_corr, _ = spearmanr(original_indices, predicted_scores)
                                if not np.isnan(spearman_corr):
                                    all_spearman_coeffs.append(spearman_corr)
                            except ValueError:
                                # Handle cases where data might be constant
                                pass
                            
                            # 皮尔逊相关性
                            try:
                                pearson_corr, _ = pearsonr(original_indices, predicted_scores)
                                if not np.isnan(pearson_corr):
                                    all_pearson_coeffs.append(pearson_corr)
                            except ValueError:
                                pass

            avg_val_loss = total_val_loss / len(valid_loader)
            avg_val_contrast_loss = total_val_contrast_loss / len(valid_loader)
            avg_val_ranking_loss = total_val_ranking_loss / len(valid_loader)
            
            logger.info(f"Epoch {epoch} 验证平均总损失: {avg_val_loss:.4f}")
            logger.info(f"Epoch {epoch} 验证平均对比损失: {avg_val_contrast_loss:.4f}")
            logger.info(f"Epoch {epoch} 验证平均排序损失: {avg_val_ranking_loss:.4f}")
            writer.add_scalar('Loss/val_epoch_total', avg_val_loss, epoch)
            writer.add_scalar('Loss/val_epoch_contrast', avg_val_contrast_loss, epoch)
            writer.add_scalar('Loss/val_epoch_ranking', avg_val_ranking_loss, epoch)

            if all_spearman_coeffs:
                avg_spearman_corr = np.mean(all_spearman_coeffs)
                logger.info(f"Epoch {epoch} 验证平均斯皮尔曼相关性: {avg_spearman_corr:.4f}")
                writer.add_scalar('Correlation/val_spearman', avg_spearman_corr, epoch)
            else:
                logger.info(f"Epoch {epoch} 验证斯皮尔曼相关性无法计算（数据不足或为常数）。")

            if all_pearson_coeffs:
                avg_pearson_corr = np.mean(all_pearson_coeffs)
                logger.info(f"Epoch {epoch} 验证平均皮尔逊相关性: {avg_pearson_corr:.4f}")
                writer.add_scalar('Correlation/val_pearson', avg_pearson_corr, epoch)
            else:
                logger.info(f"Epoch {epoch} 验证皮尔逊相关性无法计算（数据不足或为常数）。")


            # 保存最佳模型 (基于总验证损失)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(os.path.join(writer.log_dir , ckpt_dir),exist_ok=True) # 确保目录存在
                best_ckpt_path = os.path.join(writer.log_dir , ckpt_dir, f"{project_name}_best.pt")
                torch.save(model.state_dict(), best_ckpt_path)
                logger.info(f"  保存最佳模型到: {best_ckpt_path} (验证总损失: {best_val_loss:.4f})")

        # 每隔 ckpt_interval 个 Epoch 保存一次检查点
        if epoch % ckpt_interval == 0:
            os.makedirs(os.path.join(writer.log_dir , ckpt_dir), exist_ok=True)
            ckpt_path = os.path.join(writer.log_dir , ckpt_dir, f"{project_name}_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  保存检查点到: {ckpt_path}")

    logger.info("--- 训练完成 ---")
    writer.close() # 关闭 TensorBoard writer


if __name__ == "__main__":
    # 运行主函数
    main()