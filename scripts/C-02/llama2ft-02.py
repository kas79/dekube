import argparse
import os
import json
import torch
import bitsandbytes as bnb
from datasets import load_dataset, load_from_disk
from functools import partial
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from typing import Union
import functools
import logging



def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f"{40960}MB"
    print(f"{n_gpus=}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="balanced",  # dispatch efficiently the model on the available ressources
        max_memory={i: max_memory for i in range(n_gpus)},
        load_in_8bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=not False)
    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters
    :param sample: Sample dictionnary
    """
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"
    blurb = f"{INTRO_BLURB}"
    try:
        instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    except Exception as e:
        instruction = f"{INSTRUCTION_KEY}\n{sample['question']}"
    try:
        input_context = (
            f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
        )
    except Exception as e:
        input_context = (
            f"{INPUT_KEY}\n{sample['system_prompt']}"
            if sample["system_prompt"]
            else None
        )
    response = f"{RESPONSE_KEY}\n{sample['response']}"
    end = f"{END_KEY}"
    parts = [
        part for part in [blurb, instruction, input_context, response, end] if part
    ]
    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt
    return sample


def get_max_length(model):
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def preprocess_dataset(
    tokenizer: AutoTokenizer,
    max_length: int,
    seed,
    dataset: str,
    dataset_remove_columns: list,
):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)  # , batched=True)
    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=dataset_remove_columns,
        # remove_columns=["instruction", "context", "response", "text", "category"],
    )
    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)
    return dataset


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config


def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )
    return config


def find_all_linear_names(model):
    cls = (
        bnb.nn.Linear4bit
    )  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def train(model, tokenizer, dataset, output_dir, max_steps):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()
    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)
    # Get lora module names
    modules = find_all_linear_names(model)
    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)
    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)
    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=4,
            # log_level="debug",
            log_level="info",
            sharded_ddp="simple",
            # fsdp="auto_wrap",
            # output_dir="/outputs",
            output_dir=output_dir,
            # optim="paged_adamw_8bit",
            # num_train_epochs=2
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = (
        False  # re-enable for inference to speed up predictions for similar inputs
    )
    # Verifying the datatypes before training
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)
    do_train = True
    # Launch training
    print("Training...")
    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)
    ###
    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # logger = get_default_logger()
    # dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    datasetpath = os.getenv("DEKUBE_DATASET_PATH")
    dataset = load_from_disk(f"{datasetpath}")
    print(f"Number of prompts: {len(dataset)}")
    print(f"Column names are: {dataset.column_names}")
    modelpath = os.getenv("DEKUBE_MODEL_PATH")
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(modelpath, bnb_config)
    ## Preprocess dataset
    max_length = get_max_length(model)
    # dataset_remove_columns = ['instruction','context','response','text','category']
    dataset_remove_columns = []
    dataset = preprocess_dataset(
        tokenizer, max_length, 42, dataset, dataset_remove_columns
    )
    output_dir = os.getenv("iRESULTwrk")
    max_steps = int(os.getenv("MAX_STEPS", 20))
    train(model, tokenizer, dataset, output_dir, max_steps)
