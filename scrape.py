import requests
import random
import json
import tqdm
from bs4 import BeautifulSoup
import argparse
import re
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--seed-pages", type=Path, default=None)

    return parser.parse_args()


def clean_text(text):
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    boilerplate_patterns = [
        r"Community content is available under CC-BY-SA.*",
        r"This page was last edited on.*",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\[\d+\]", "", text)

    text = re.sub(r"\[\[.*?\]\]", "", text)  # Remove [[link]] style links
    text = re.sub(r"\{\{.*?\}\}", "", text)  # Remove {{template}} style markup

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{2,}", "\n", text)

    text = text.strip()

    return text


def parse_page(page):
    base_url = "https://fallout.fandom.com/"
    url = f"{base_url}wiki/{page}"

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    body_class = "mw-content-ltr mw-parser-output"
    body = soup.find("div", class_=body_class)
    if not body:
        return None
    elements = body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"], recursive=False)

    body_text = clean_text("\n".join([x.text for x in elements]))

    link_elements = body.find_all("a")
    links = []
    for e in link_elements:
        if not e["href"].startswith("/wiki/"):
            # external link or some image or something skip
            continue
        links.append(e["href"].removeprefix("/wiki/"))

    return {
        "page": page,
        "url": url,
        "text": body_text,
        "links": links,
    }


def main():
    args = _parse_args()

    if args.seed_pages:
        with args.seed_pages.open() as io:
            seed_pages = json.load(io)
    else:
        seed_pages = ["Nick_Valentine"]

    pages = seed_pages
    visited = set()
    page = None
    results = []

    for page_index in tqdm.tqdm(range(args.max_pages), total=args.max_pages):
        while page is None or page in visited:
            page = pages.pop(random.randint(0, len(pages) - 1))

        try:
            result = parse_page(page)
        except Exception as e:
            print(f"An error occurred for page {page}: {e}")
            result = None

        visited.add(page)
        if result is not None:
            pages.extend(result["links"])
            results.append(result)

    with args.output.open("w") as io:
        json.dump(results, io)


if __name__ == "__main__":
    main()
