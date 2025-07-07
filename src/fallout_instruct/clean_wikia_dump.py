import click
import mwparserfromhell
import json
import tqdm
import transformers
from pathlib import Path


def clean_text(text):
    code = mwparserfromhell.parse(text)

    cleaned_text = "\n".join(
        [x.strip() for x in code.strip_code(normalize=True, collapse=True).splitlines()]
    )

    return cleaned_text


@click.command()
@click.argument("raw", type=Path)
@click.argument("output", type=Path)
@click.argument("model", type=str)
@click.option("--max-length", type=int, default=-1)
@click.option("--stride", type=int, default=256)
@click.option("--min-length", type=int, default=64)
def main(raw, output, model, max_length, stride, min_length):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)

    if max_length != -1:
        tokenizer.max_length = max_length

    with raw.open() as fin, output.open("w") as fout:
        for line in tqdm.tqdm(fin.readlines()):
            data = json.loads(line)

            text = data.get("revision", {}).get("text")

            text = clean_text(text)

            if not text:
                continue

            ids = tokenizer(text, add_special_tokens=False)["input_ids"]

            length = len(ids)

            if length < min_length:
                continue

            # Use sliding window techniques to divide large pages into multiple samples
            for index in range(0, length, stride):
                chunk_ids = ids[index : index + stride]

                if len(chunk_ids) < min_length:
                    break

                chunk_text = tokenizer.decode(
                    chunk_ids, clean_up_tokenization_spaces=True
                )

                sample = {"text": chunk_text, "title": data.get("title", "")}
                fout.write(json.dumps(sample) + "\n")
