import click
import json
import functools
import datasets
import peft
import torch
import transformers


def preprocess(samples, tokenizer, system_prompt):
    full_prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": samples["prompt"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    answer = samples["answer"] + tokenizer.eos_token

    full = full_prompt + answer

    encoding = tokenizer(
        text=full,
        text_target=samples["answer"],
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True,
    )

    prompt_length = len(tokenizer(full_prompt, add_special_tokens=False)["input_ids"])
    labels = encoding.input_ids.clone()
    labels[:, :prompt_length] = -100

    return {
        "input_ids": encoding.input_ids.squeeze(),
        "attention_mask": encoding.attention_mask.squeeze(),
        "labels": labels.squeeze(),
    }


@click.command()
@click.option("--model", default="meta-llama/Llama-3.2-1B-Instruct")
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu")
@click.option("--train-config", default=None)
def main(model, device, train_config):
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model, device_map=device, quantization_config=bnb_config
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token

    model = peft.prepare_model_for_kbit_training(model)

    peft_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
    )
    model = peft.get_peft_model(model, peft_config)

    data = datasets.load_dataset(
        "parquet",
        data_files={
            "train": "data/fallout_instruction_v1/train.parquet",
            "dev": "data/fallout_instruction_v1/dev.parquet",
        },
    )

    data = data.filter(
        lambda x: x.get("prompt") is not None and x.get("answer") is not None
    )

    system_prompt = "You are a Fallout franchise expert. You answer questions related to the series universe."

    preprocess_func = functools.partial(
        preprocess, tokenizer=tokenizer, system_prompt=system_prompt
    )

    data = data.preprocess(preprocess_func)

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, padding="longest", return_tensors="pt"
    )

    if train_config is None:
        train_args = transformers.TrainingArguments()
    else:
        with open(train_config) as io:
            args = json.load(io)
        train_args = transformers.TrainingArguments(**args)

    trainer = transformers.Trainer(
        model=model,
        args=train_args,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(train_args.output_dir)
