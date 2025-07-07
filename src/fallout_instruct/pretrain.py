import click
import json
import functools
import datasets
import transformers
from pathlib import Path


def preprocess(samples, tokenizer):
    return tokenizer(
        samples["text"],
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_attention_mask=True,
    )


@click.command()
@click.argument("model", type=str)
@click.argument("data", type=str)
@click.option("--train-config", type=Path)
def main(model, data, train_config):
    device = "cpu"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(model, device_map=device)

    model.gradient_checkpointing_enable()

    data = datasets.load_dataset(
        "json",
        data_files={"train": data},
    )

    preprocess_func = functools.partial(
        preprocess,
        tokenizer=tokenizer,
    )

    data = data.map(preprocess_func, batched=True)

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, padding="longest", return_tensors="pt"
    )

    if train_config is None:
        train_args = transformers.TrainingArguments()
    else:
        with open(train_config) as io:
            args = json.load(io)
        train_args = transformers.TrainingArguments(**args)

    import pdb

    pdb.set_trace()
    trainer = transformers.Trainer(
        model=model,
        args=train_args,
        train_dataset=data["train"],
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(train_args.output_dir)
