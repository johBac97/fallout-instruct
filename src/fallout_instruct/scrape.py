import asyncio
import aiohttp
import json
import tqdm.asyncio
from bs4 import BeautifulSoup
import argparse
import re
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import aiofiles


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="Path to output JSON Lines file")
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--seed-pages", type=Path, default=None)
    parser.add_argument(
        "--concurrency", type=int, default=5, help="Number of concurrent workers"
    )
    return parser.parse_args()


def clean_text(text):
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    patterns = [
        r"Community content is available under CC-BY-SA.*",
        r"This page was last edited on.*",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[\[.*?\]\]", "", text)
    text = re.sub(r"\{\{.*?\}\}", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=5, max=30),
    reraise=True,
)
async def fetch_url(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url, timeout=10) as response:
        if response.status == 429:
            raise Exception("HTTP 429 Too Many Requests")
        response.raise_for_status()
        return await response.text()


async def worker(
    name: int,
    session: aiohttp.ClientSession,
    queue: asyncio.Queue,
    visited: set,
    lock: asyncio.Lock,
    out_path: Path,
    max_pages: int,
    pbar: tqdm.asyncio.tqdm,
):
    while True:
        async with lock:
            if len(visited) >= max_pages:
                break
        try:
            page = await queue.get()
        except asyncio.CancelledError:
            break

        # Remove retireval of images
        invalid_page = any([x in page for x in [".jpg", ".webp", ".png", "&"]])

        async with lock:
            if page in visited or invalid_page:
                queue.task_done()
                continue
            visited.add(page)

        url = f"https://fallout.fandom.com/wiki/{page}"
        try:
            html = await fetch_url(session, url)
        except Exception as e:
            print(f"Worker {name}: error fetching {page}: {e}")
            queue.task_done()
            await asyncio.sleep(1)
            continue

        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("div", class_="mw-content-ltr mw-parser-output")
        if body:
            elems = body.find_all(
                ["h1", "h2", "h3", "h4", "h5", "h6", "p"], recursive=False
            )
            text = clean_text("\n".join([e.get_text() for e in elems]))
            links = [
                a["href"].removeprefix("/wiki/")
                for a in body.find_all("a", href=True)
                if a["href"].startswith("/wiki/") and "#" not in a["href"]
            ]

            result = {"page": page, "url": url, "text": text, "links": links}
            async with lock:
                async with aiofiles.open(out_path, "a", encoding="utf-8") as f:
                    await f.write(json.dumps(result, ensure_ascii=False) + "\n")
                for ln in links:
                    if ln not in visited:
                        queue.put_nowait(ln)

        queue.task_done()
        pbar.update(1)


async def main_scrape():
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        args.output.touch()

    if args.seed_pages:
        with args.seed_pages.open() as f:
            seed = json.load(f)
    else:
        seed = ["Starlight_Drive_In"]

    queue = asyncio.Queue()
    for page in seed:
        queue.put_nowait(page)
    visited = set()
    lock = asyncio.Lock()

    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        with tqdm.asyncio.tqdm(total=args.max_pages) as pbar:
            workers = [
                asyncio.create_task(
                    worker(
                        i,
                        session,
                        queue,
                        visited,
                        lock,
                        args.output,
                        args.max_pages,
                        pbar,
                    )
                )
                for i in range(args.concurrency)
            ]
            await queue.join()
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    print(f"Scraping complete. {len(visited)} pages saved to {args.output}")

def main():
    asyncio.run(main_scrape())

if __name__ == "__main__":
    main()
