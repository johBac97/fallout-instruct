import click
import json
import functools
import datasets
import peft
import torch
import transformers

import numpy as np
import evaluate

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


torch.set_float32_matmul_precision("medium")


def preprocess_logits_for_metrics(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    return preds


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds

    # The trainer pads the predictions with -100 which causes the batch_decode to fail
    preds[preds == -100] = tokenizer.pad_token_id

    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
    rouge_res = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return {
        "bleu": bleu_res["bleu"],
        "rouge1": rouge_res["rouge1"].item(),
        "rouge2": rouge_res["rouge2"].item(),
        "rougeL": rouge_res["rougeL"].item(),
    }


def preprocess_batch(samples, tokenizer, system_prompt):
    # Prepare batched prompts
    batch_prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in samples["prompt"]
    ]

    # Append EOS token to answers
    batch_answers = [answer + tokenizer.eos_token for answer in samples["answer"]]

    # Combine prompts and answers
    batch_full = [
        prompt + answer for prompt, answer in zip(batch_prompts, batch_answers)
    ]

    encoding = tokenizer(
        text=batch_full,
        text_target=samples["answer"],
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding="longest",
    )

    # Calculate prompt lengths for masking
    prompt_lengths = [
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in batch_prompts
    ]

    # Create labels and mask prompt tokens
    labels = encoding.input_ids.clone()
    for i, prompt_length in enumerate(prompt_lengths):
        labels[i, :prompt_length] = -100

    return {
        "input_ids": encoding.input_ids,
        "attention_mask": encoding.attention_mask,
        "labels": labels,
    }


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
@click.argument("model", type=str)
@click.argument("data", type=str)
@click.option("--train-config", default=None)
def main(model, data, train_config):
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model, device_map="auto", quantization_config=bnb_config
    )

    tokenizer.pad_token = tokenizer.eos_token

    model = peft.prepare_model_for_kbit_training(model)

    peft_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
    )
    model = peft.get_peft_model(model, peft_config)

    ds = datasets.load_dataset(data)

    ds = ds.filter(
        lambda x: x.get("prompt") is not None and x.get("answer") is not None
    )

    system_prompt = "You are a Fallout franchise expert. You answer questions related to the series universe."

    preprocess_func = functools.partial(
        preprocess_batch, tokenizer=tokenizer, system_prompt=system_prompt
    )

    ds = ds.map(preprocess_func, batched=True, num_proc=12, batch_size=8)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding="longest", pad_to_multiple_of=8
    )

    compute_metrics_func = functools.partial(compute_metrics, tokenizer=tokenizer)

    if train_config is None:
        train_args = transformers.TrainingArguments()
    else:
        with open(train_config) as io:
            args = json.load(io)
        train_args = transformers.TrainingArguments(**args)

    trainer = transformers.Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"].select(range(10)),
        data_collator=data_collator,
        compute_metrics=compute_metrics_func,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()

    trainer.save_model(train_args.output_dir)
