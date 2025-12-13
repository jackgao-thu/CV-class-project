import json
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
from transformers import AutoModelForImageTextToText, AutoProcessor

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
from utils import format_data
import torch.nn.functional as F

class MultiModalDataset(Dataset):
    def __init__(self, data):
        self.data = []
        for d in data:
            d_pos = d.copy()
            d_pos['is_noise'] = 0
            self.data.append(d_pos)
            d_neg = d.copy()
            d_neg['is_noise'] = 1
            self.data.append(d_neg)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return format_data(self.data[idx])
    
def add_noise_to_video(video_tensor):
    noise = torch.randn_like(video_tensor) * 0.1
    # video_tensor_noised = video_tensor + noise
    # return video_tensor_noised
    return noise

from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor

def multimodal_collate_fn(examples: list[dict[str, Any]], processor, mode) -> dict[str, torch.Tensor]:
    texts = []
    images = []
    videos = []
    is_noises = []
    ids = []
    for example in examples:
        texts.append(processor.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=True if mode == "eval" else False,
        ))
        image_inputs, video_inputs = process_vision_info(example["messages"])
        if image_inputs is not None:
            images.extend(image_inputs)
        if video_inputs is not None:
            if example.get("is_noise", 0) == 1:
                video_inputs = [add_noise_to_video(torch.from_numpy(np.load(v))) if isinstance(v, str) else add_noise_to_video(v)
                                for v in video_inputs]
                
            videos.extend(video_inputs)
            
        is_noises.append(example.get("is_noise", 0))
        ids.append(example["id"])

    if len(images) == 0: images = None
    if len(videos) == 0: videos = None

    batch = processor(
        text=texts,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )

    # Choice 1
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

    labels[labels == processor.tokenizer.pad_token_id] = -100
    visual_tokens = (
        [151652, 151653, 151655, 151656]
        if isinstance(processor, Qwen2VLProcessor)
        else [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    )
    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100  
    batch["labels"] = labels
    batch["is_noise"] = torch.tensor(is_noises)
    batch["ids"] = ids
    return batch

########### 4. 自定义Trainer ##################
from trl import SFTTrainer
from torch.utils.data import DataLoader, SequentialSampler


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
            shuffle=False,
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

        is_noises = inputs["is_noise"]
        pos_mask = (is_noises == 0)
        neg_mask = (is_noises == 1)

        def select_batch(inputs, mask):
            selected = {}
            b = mask.shape[0]
            idx = [i for i in range(b) if mask[i]]
            for k, v in inputs.items():
                if k == "pixel_values_videos" and "video_grid_thw" in inputs:
                    video_grids = inputs["video_grid_thw"].cpu().numpy()
                    n_frames = [int(np.prod(g)) for g in video_grids]
                    cum_frames = np.cumsum([0] + n_frames)
                    v_frames = []
                    for i in idx:
                        s, e = cum_frames[i], cum_frames[i+1]
                        v_frames.append(inputs["pixel_values_videos"][s:e])
                    selected[k] = torch.cat(v_frames, dim=0) if len(v_frames) > 1 else v_frames[0]
                elif isinstance(v, torch.Tensor) and v.shape[0] == b:
                    selected[k] = v[mask]
                elif isinstance(v, list) and len(v) == b:
                    arr = [v[i] for i in idx]
                    selected[k] = arr
                else:
                    selected[k] = v
            return selected

        inputs_pos = select_batch(inputs, pos_mask)
        inputs_neg = select_batch(inputs, neg_mask)

        loss_pos, outputs_pos = super().compute_loss(
            model, inputs_pos, return_outputs=True, num_items_in_batch=None
        )
        loss_neg, outputs_neg = super().compute_loss(
            model, inputs_neg, return_outputs=True, num_items_in_batch=None
        )

        loss = alpha * loss_pos - beta * loss_neg

        if mode == "train":
            print(f"[Loss] pos: {loss_pos.item():.4f}  neg: {loss_neg.item():.4f}  total: {loss.item():.4f}")

        if return_outputs:
            outputs = {
                "pos": outputs_pos,
                "neg": outputs_neg,
                "inputs_pos": inputs_pos,
                "inputs_neg": inputs_neg,
            }
            return loss, outputs
        else:
            return loss
        
############ 5. 主流程 ##################
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    alpha = getattr(script_args, 'alpha', 1.0)
    beta = getattr(script_args, 'beta', 0.1)
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Model
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
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_reentrant = False
        model.enable_input_require_grads()

    # Processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Dataset
    with open(script_args.dataset_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    train_dataset = MultiModalDataset(data)

    # Collate
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