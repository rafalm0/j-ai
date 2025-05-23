from pathlib import Path
import re
import json

# ---- SETTINGS ----
CHUNK_SIZE = 350
STEP_SIZE = 300

# Define your source files
source_files = [
    {"path": "RAG-source/nyt_1999_raw.txt", "output": "RAG-processed/nyt_1999_full_clean.jsonl", "year": "1999"},
    {"path": "RAG-source/nyt_2024_raw.txt", "output": "RAG-processed/nyt_2024_full_clean.jsonl", "year": "2024"},
]


# ---- UTILITY FUNCTIONS ----

def clean_text(text: str) -> str:
    """Remove unwanted headers, footers, etc."""
    text = re.sub(r'Page \d+ of \d+ © 2025 Factiva, Inc. All rights reserved\.', '\n', text)
    text = re.sub(r'© \d{4} The New York Times Company\. All Rights Reserved\.', '\n', text)
    text = re.sub(r'NYTimes\.com Feed|NYTFEED|English|Document [A-Z0-9]+', '\n', text)
    text = re.sub("'", '', text)
    # text = re.sub('\(c\) 1999 New York Times Company', '\n', text)
    # text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_articles_by_author(text: str):
    """Split text into articles using 'By [AUTHOR]' as markers."""
    parts = re.split(r'\nDocument nytf', text)[:-1]
    articles = []

    for i in range(0, len(parts)):
        article = parts[i]
        article = article.split('\n\n\n\n\x0c')[1][1:]
        headers = article.split('\n')[:12]
        author = None
        title = None
        try:
            body = article.split(" New York Times Company")[1]
        except IndexError:
            body = article.split("International New York Times")[1]

        for j, header in enumerate(headers):
            if 'By ' in header:
                author = header.split('By ')[-1]
                if headers[j - 1] != ' ':
                    title = headers[j - 1]
                else:
                    title = headers[j - 2]
                break
            if ('words' in header) and (author is None): # backup author aquiring by looking for words
                author = headers[j - 1].split('By ')[-1]
                if headers[j - 2] != ' ':
                    title = headers[j - 2]
                else:
                    title = headers[j - 3]

        if (author == None) or (title == None):
            print("Author or title not found")

        if author == 'BUSINESS DIGEST':
            title = 'BUSINESS DIGEST'

        if body == '':
            print(f'Error on article {title} by author')
        articles.append({
            "author": author,
            "title": title,
            "body": body
        })

    return articles


def chunk_text(text: str):
    """Chunk text into overlapping word windows."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), STEP_SIZE):
        chunk_words = words[i:i + CHUNK_SIZE]
        if len(" ".join(chunk_words)) > 510:
            chunk_words_1 = chunk_words[:len(chunk_words) // 2 + 20]
            chunk_words_2 = chunk_words[len(chunk_words) // 2 - 20:]
            chunks.append(" ".join(chunk_words_1))
            chunks.append(" ".join(chunk_words_2))
        elif len(chunk_words) < 100:
            continue
        else:
            chunks.append(" ".join(chunk_words))

    return chunks


# ---- MAIN PROCESSING ----

for source in source_files:
    print(f"Processing {source['path']}...")

    path = Path(source["path"])
    output_path = Path(source["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_text = path.read_text(encoding="utf-8")
    cleaned = clean_text(raw_text)

    articles_info = split_articles_by_author(cleaned)

    # Save
    with output_path.open("w", encoding="utf-8") as f:
        for idx, article in enumerate(articles_info):
            chunks = chunk_text(article["body"])
            for chunk_idx, chunk in enumerate(chunks):
                record = {
                    "id": f"nyt_{source['year']}_{idx + 1:04}_chunk_{chunk_idx + 1}",
                    "title": article["title"],
                    "author": article["author"],
                    "chunk": chunk
                }
                json.dump(record, f)
                f.write("\n")

    print(f"✅ Saved {len(articles_info)} articles to {output_path}")
