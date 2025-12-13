import json
import os
import random
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2VLProcessor

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from qwen_vl_utils import process_vision_info
from utils import format_data # 假定你的 format_data 依然可用

# 定义CSV路径
LABEL_CSV_PATH = "/hdd/common/datasets/medical-image-analysis/CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv"

########### 1. Dataset ##############

class MultiModalDataset(Dataset):
    def __init__(self, data, label_csv_path=LABEL_CSV_PATH, msg_processor=None):
        """
        Args:
            data: List of dicts (loaded from json)
            label_csv_path: Path to the CSV labels
        """
        # 1. 预处理 CSV 标签数据
        print(f"Loading label data from {label_csv_path}...")
        self.label_map = self._load_labels(label_csv_path)
        
        # 2. 构建数据对 (Positive & Negative)
        # 同时收集所有可用的视频路径构建 sample pool
        self.data = []
        self.video_path_pool = [] 
        
        print("Processing dataset entries...")
        for d in data:
            # 收集有效路径用于采样
            if "videos" in d and len(d["videos"]) > 0:
                self.video_path_pool.extend(d["videos"])
            
            # 构建正样本
            d_pos = d.copy()
            d_pos['is_noise'] = 0  # 0 represent Positive
            self.data.append(d_pos)
            
            # 构建负样本框架 (具体的视频替换在 getitem 中进行以节省内存)
            d_neg = d.copy()
            d_neg['is_noise'] = 1  # 1 represent Negative
            self.data.append(d_neg)
            
        print(f"Dataset initialized. Total samples (pos+neg): {len(self.data)}. Video pool size: {len(self.video_path_pool)}")

    def _load_labels(self, csv_path):
        """
        读取CSV并将文件名映射为numpy array标签。
        CSV Key: 'train_1_a_1.nii.gz' -> labels
        Mapping Key: 'train_1_a_1' (去除后缀) -> labels
        """
        if not os.path.exists(csv_path):
            print(f"Warning: Label CSV not found at {csv_path}. Negative sampling will be random.")
            return {}
        
        df = pd.read_csv(csv_path)
        
        print("=========================Processing Labels=============================")
        
        label_map = {}
        # 假设第一列是文件名，后面是labels
        # 使用iterrows效率较低，但只在init运行一次；若数据量过百万，可改用 vectorization
        keys = df.iloc[:, 0].values
        labels = df.iloc[:, 1:].values # shape [N, 14]
        
        for k, l in zip(keys, labels):
            # 将 'train_1_a_1.nii.gz' 转换为 trunk 'train_1_a_1'
            trunk = k.split('.')[0] 
            label_map[trunk] = l.astype(np.float32)
            
        return label_map

    def _get_start_name(self, path):
        """从路径提取用于匹配的主干名"""
        filename = os.path.basename(path)
        # 处理 .npy 或 .nii.gz
        if filename.endswith('.npy'):
            return filename[:-4]
        elif filename.endswith('.nii.gz'):
            return filename[:-7]
        else:
            return filename.split('.')[0]

    def _get_negative_video_path(self, original_video_path, threshold=7, max_retries=10):
        """
        寻找一个与 original_video_path 标签差异足够大的视频路径。
        Threshold: 汉明距离阈值（不一样的标签数量）
        """
        if not original_video_path or not self.video_path_pool:
            return original_video_path

        orig_key = self._get_start_name(original_video_path)
        orig_labels = self.label_map.get(orig_key)

        # 如果找不到原始标签，随机返回一个不同的文件
        if orig_labels is None:
            return random.choice(self.video_path_pool)

        # 尝试采样
        while(threshold):
            for _ in range(max_retries):
                candidate_path = random.choice(self.video_path_pool)
                cand_key = self._get_start_name(candidate_path)
                
                # 避免选到自己
                if cand_key == orig_key:
                    continue
                    
                cand_labels = self.label_map.get(cand_key)
                
                if cand_labels is not None:
                    # 计算差异 (汉明距离/不相等的元素个数)
                    # 假设 label 是 0/1 向量
                    diff = np.sum(np.abs(orig_labels - cand_labels))
                    if diff >= threshold:
                        print("diff = ", diff)
                        return candidate_path
            
            threshold = threshold - 1
            
        # 如果重试多次没找到足够不同的，为了效率直接返回最后一次抽样的结果
        # 或者为了严谨，可以放宽阈值，这里选择直接返回以保证速度
        print("[ERROR] Not Found For ", original_video_path)
        
        return candidate_path

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 如果是负样本，进行“查找替换”操作
        if item.get('is_noise', 0) == 1:
            # 必须深拷贝，因为我们要修改内部嵌套的 messages 结构
            # 简单的 copy() 只是浅拷贝，修改 messages 内部的 dict 会影响到缓存的 self.data
            import copy
            item = copy.deepcopy(item)
            
            # 1. 获取并替换 videos 列表（虽然 collate_fn 主要看 messages，但保持一致是个好习惯）
            original_videos = item.get("videos", [])
            new_videos = []
            
            # 创建一个从 旧路径 -> 新路径 的映射，用于更新 messages
            path_mapping = {}
            
            for v_path in original_videos:
                neg_path = self._get_negative_video_path(v_path)
                new_videos.append(neg_path)
                path_mapping[v_path] = neg_path
            
            item["videos"] = new_videos
            
            # 2. 【关键修复】遍历 messages，将里面的 video 路径替换为负样本路径
            # Qwen2-VL 的 process_vision_info 是解析 messages 里的 'video' 字段
            if "messages" in item:
                for msg in item["messages"]:
                    if "content" in msg and isinstance(msg["content"], list):
                        for content_part in msg["content"]:
                            # 查找类型为 video 的字段
                            if content_part.get("type") == "video":
                                old_vid_path = content_part.get("video")
                                # 如果这个路径在我们刚才生成的映射表中，则替换
                                if old_vid_path in path_mapping:
                                    content_part["video"] = path_mapping[old_vid_path]
                                # 如果是列表形式的路径（某些格式可能是 list），也尝试替换
                                elif isinstance(old_vid_path, list):
                                     new_paths = [path_mapping.get(p, p) for p in old_vid_path]
                                     content_part["video"] = new_paths

        # format_data 会处理 messages 和 videos 的对应关系
        return format_data(item)

########### 3. Collate Function ##############

def multimodal_collate_fn(examples: list[dict[str, Any]], processor, mode) -> dict[str, torch.Tensor]:
    texts = []
    images = []
    videos = []
    is_noises = []
    ids = []
    
    for example in examples:
        # print("example= ", example)
        texts.append(processor.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=True if mode == "eval" else False,
        ))
        image_inputs, video_inputs = process_vision_info(example["messages"])
        if image_inputs is not None:
            images.extend(image_inputs)
        if video_inputs is not None:
            # 修改点：此处不再生成噪声。
            # Dataset.__getitem__ 已经确保了如果是负样本(is_noise=1)，example中的路径已经是负向样本的路径了。
            # 我们只需要负责加载数据即可。
            loaded_videos = []
            for v in video_inputs:
                if isinstance(v, str):
                    # 加载 npy 文件
                    loaded_videos.append(torch.from_numpy(np.load(v)))
                else:
                    loaded_videos.append(v)
            videos.extend(loaded_videos)
            
        is_noises.append(example.get("is_noise", 0))
        ids.append(example["id"])

    if len(images) == 0: images = None
    if len(videos) == 0: videos = None

    # Processor 负责填充、转换
    batch = processor(
        text=texts,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )

    # Label 处理逻辑 (Pretrain / SFT)
    if mode == "pretrain":
        labels = batch["input_ids"].clone()
        
    elif mode == "sft":
        labels = torch.full_like(batch["input_ids"], -100)
        B, L = batch["input_ids"].shape
        for input_ids_cur, labels_cur in zip(batch["input_ids"], labels):
            start_idx = 0
            end_idx = 0
            while start_idx < L:
                if input_ids_cur[start_idx] == processor.tokenizer.encode("<|im_start|>")[0]:
                    if input_ids_cur[start_idx + 1] == processor.tokenizer.encode("assistant")[0]:
                        start_idx = start_idx + len(processor.tokenizer.encode("<|im_start|>assistant\n"))
                        end_idx = start_idx + 1
                        while input_ids_cur[end_idx] != processor.tokenizer.encode("<|im_end|>")[0]:
                            end_idx = end_idx + 1
                        labels_cur[start_idx:end_idx+1] = input_ids_cur[start_idx:end_idx+1]
                start_idx = start_idx + 1

    # pad others
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # 屏蔽视觉 Tokens
    visual_tokens = (
        [151652, 151653, 151655, 151656]
        if isinstance(processor, Qwen2VLProcessor)
        else [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    )
    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100
        
    batch["labels"] = labels
    batch["is_noise"] = torch.tensor(is_noises) # shape [B]
    batch["ids"] = ids # shape [B]
    
    return batch

########### 4. 自定义Trainer ##################

def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    original_shape = logits.shape[:-1]
    num_classes = logits.shape[-1]
    flat_logits = logits.reshape(-1, num_classes)
    entropies = []
    for chunk in flat_logits.split(chunk_size, dim=0):
        logps = F.log_softmax(chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        entropies.append(chunk_entropy)
    entropies = torch.cat(entropies, dim=0)
    return entropies.reshape(original_shape)

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, alpha, beta, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        
    def get_train_dataloader(self):
        train_sampler = SequentialSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False, # 必须 False 以保证 SequentialSampler 配对
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"
        alpha = getattr(self, "alpha", 1.0)
        beta = getattr(self, "beta", 0.1)

        # 1. 提取并移除辅助字段，避免传给模型导致报错
        is_noises = inputs.pop("is_noise") # shape [Batch_Size]
        # ids = inputs.pop("ids", None) # 如果 ids 不需要传给模型，建议也 pop 掉
        
        # 2. 单次前向传播 (Single Forward Pass)
        # 这解决了 DeepSpeed "Gradient computed twice" 的问题，
        # 同时也避免了手动切分 pixel_values_videos 的复杂逻辑。
        outputs = model(**inputs, return_dict=True)
        
        # 3. 手动计算 Per-Sample Loss
        # Qwen2-VL 等 Causal LM 的 Loss 计算是 Shifted Cross Entropy
        logits = outputs.logits
        labels = inputs["labels"]
        
        # Logits 预测下一个 Token，所以 Logits 取 [:-1], Labels 取 [1:]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 使用 reduction='none' 获得每个 Token 的 Loss
        # shape: [Batch_Size * (Seq_Len - 1)]
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        flat_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 还原形状为 [Batch_Size, Seq_Len - 1]
        per_token_loss = flat_loss.view(shift_labels.size(0), -1)
        
        # 计算每个样本的平均 Loss (只计算非 padding 部分)
        # CrossEntropyLoss 对 ignore_index 位置输出为 0，所以分子直接求和即可
        # 分母需要统计非 -100 的 Token 数量
        valid_token_counts = (shift_labels != -100).sum(dim=1).float()
        per_sample_loss = per_token_loss.sum(dim=1) / valid_token_counts.clamp(min=1.0)

        # 4. 根据 is_noise 掩码区分 Positive / Negative Loss
        pos_mask = (is_noises == 0)
        neg_mask = (is_noises == 1)

        # 这里处理 batch 中可能全是正样本或全是负样本的情况
        if pos_mask.any():
            loss_pos = per_sample_loss[pos_mask].mean()
        else:
            loss_pos = torch.tensor(0.0, device=model.device, dtype=per_sample_loss.dtype)

        if neg_mask.any():
            loss_neg = per_sample_loss[neg_mask].mean()
        else:
            loss_neg = torch.tensor(0.0, device=model.device, dtype=per_sample_loss.dtype)

        # 5. 组合最终 Loss
        # 注意：通常 Unlikelihood Training 是 max(0, margin - neg_loss) 或者 minimizing neg_likelihood
        # 按照您原本的公式: alpha * pos - beta * neg
        # 这意味着：如果负样本损失越大(越预测不对)，总损失越小(越好)。
        loss = alpha * loss_pos - beta * loss_neg

        # 打印调试信息 (仅在 rank0)
        if mode == "train":
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                     print(f"\r[Step] Pos: {loss_pos.item():.4f} | Neg: {loss_neg.item():.4f} | Total: {loss.item():.4f}", end="", flush=True)
            else:
                print(f"\r[Step] Pos: {loss_pos.item():.4f} | Neg: {loss_neg.item():.4f} | Total: {loss.item():.4f}", end="", flush=True)

        if return_outputs:
            # 构造一个符合 Trainer 预期的 outputs 结构
            return loss, outputs
        
        return loss

############ 5. 主流程 ##################
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    alpha = getattr(script_args, "alpha", 1.0)
    beta = getattr(script_args, "beta", 0.1)

    # ... (模型加载代码保持不变) ...
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    # ... (Gradient Checkpointing 等设置保持不变) ...
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_reentrant = False
        model.enable_input_require_grads()

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Dataset
    with open(script_args.dataset_name, "r", encoding="utf-8") as f:
        data_json = json.load(f)
        
    # 初始化 Dataset，传入 processor（如果需要），并且会自动加载 CSV
    train_dataset = MultiModalDataset(data_json, label_csv_path=LABEL_CSV_PATH)

    collate_fn = partial(
        multimodal_collate_fn,
        processor=processor,
        mode="sft",
    )
    
    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        processing_class=processor,
        peft_config=get_peft_config(model_args),
        alpha=alpha,
        beta=beta,
    )
    
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)