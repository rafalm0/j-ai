from pathlib import Path
import re
import json

# ---- SETTINGS ----
CHUNK_SIZE = 350
STEP_SIZE = 300

# Define your source files
source_files = [
    {"path": "RAG-source/nyt_1999_raw.txt", "output": "RAG-processed/nyt_1999_full_clean.jsonl", "year": "1999",
     "batch": "1"},
    {"path": "RAG-source/nyt_2024_raw.txt", "output": "RAG-processed/nyt_2024_full_clean.jsonl", "year": "2024",
     "batch": "1"},
    {"path": "RAG-source/nyt_1999_raw-2.txt", "output": "RAG-processed/nyt_1999_full_clean-2.jsonl", "year": "1999",
     "batch": "2"},
    {"path": "RAG-source/nyt_2024_raw-2.txt", "output": "RAG-processed/nyt_2024_full_clean-2.jsonl", "year": "2024",
     "batch": "2"},
]


# number of articles : 80/88

# ---- UTILITY FUNCTIONS ----

def clean_text(text: str) -> str:
    """Remove unwanted headers, footers, etc."""
    text = re.sub(r'Page \d+ of \d+ © 2025 Factiva, Inc. All rights reserved\.', '', text)
    text = re.sub(r'© \d{4} The New York Times Company\. All Rights Reserved\.', '', text)
    text = re.sub(r'Document TOR', 'Document tor', text)
    text = re.sub(r'NYTimes\.com Feed|NYTFEED|English', '', text)
    text = re.sub("'", '', text)
    text = re.sub(r'\nFactiva\n', '', text).strip()
    text = re.sub(r'\d{1,4}/\d{1,4}\n', '', text).strip()
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{1,2},\s\d{1,2}:\d{1,2}\s(PM|AM)', '', text).strip()
    text = re.sub(r'https://.*', '', text).strip()
    # text = re.sub('\(c\) 1999 New York Times Company', '\n', text)
    # text = re.sub(r'Copyright ... 1999 The Toronto Star', '', text)
    # text = re.sub(r'"All material copyright Bell Globemedia Publishing Inc. and its licensors. All rights reserved. "', '\n', text)

    # text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_author_title(Headers: list[str]) -> (str, str):
    author = None
    title = None

    # Clean headers first
    Headers = [scrub_header(h) for h in Headers if h.strip()]

    for j, header in enumerate(Headers):
        if 'By ' in header:  # found author, perfect, then the title is before it normally
            author = header.split('By ')[-1]
            if Headers[j - 1] != ' ':
                title = Headers[j - 1]
            else:
                title = Headers[j - 2]
            break

        if ('words' in header) and (not author):  # backup author acquiring by looking for words
            if 'By ' in Headers[j - 1]:
                author = Headers[j - 1].split('By ')[-1]
            elif j < 3:  # probably no author since there is normally section + title and then author
                author = None
                title = Headers[j - 1]
            else:
                author = Headers[j - 1]
                title = ' '.join(Headers[1:j - 1])  # removing the first one since its probably section, then merge

    if author and ("<p>" in author):
        author = author.split("<p>")[0]
    if author and ('BY ' in author.upper()):
        author = author.upper().split('BY ')[-1]
    return author, title


def scrub_header(h: str) -> str:
    if '\x0c' in h:
        h.replace('\x0c', '')
    return h


def split_articles_by_author(text: str, font='nyt'):
    """Split text into articles using 'By [AUTHOR]' as markers."""
    parts = re.split(r'\nDocument nyt', text)[:-1]  # for the v1 of the documents
    if len(parts) < 10:  # now we are in the version 2 of documents
        parts = re.split(r'0000', text)[:-1]

    articles = []

    for i in range(0, len(parts)):

        article = parts[i]
        if len(article) < 100:  # sometimes articles incredbly short or simply a bug of the split by 0000, skip
            continue
        if '\n\n\n\n\x0c' in article[:100]:
            article = article.split('\n\n\n\n\x0c')[1]

        body_start = re.search(
            r".*?(Copyright\s....\.{0,1}\sThe\sNew\sYork\sTimes\sCompany\..{0,1}All\sRights\sReserved\.{0,1}|\d{4} New York Times Company|Copyright ... .... The Toronto Star|Copyright . .... Montreal Gazette|..... The Globe and Mail Inc\. All Rights Reserved\.|International New York Times|All material copyright Bell Globemedia Publishing Inc\. and its licensors\. All rights reserved\.)",
            article).end()
        body = article[body_start:]
        headers = article[:body_start].split("\n")

        while '' in headers:
            headers.remove('')

        author, title = get_author_title(headers)

        if (author == 'BUSINESS DIGEST') or (not author and title == 'BUSINESS DIGEST'):
            title = 'BUSINESS DIGEST'
            author = 'BUSINESS DIGEST'

        if not (author and title):
            print("Author or title not found, skipping...")
            continue

        if body == '':
            print(f'Error on article {title}, skipping....')
            continue
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
                    "id": f"nyt_{source['year']}_{idx + 1:04}_chunk_{chunk_idx + 1}_{source['batch']}",
                    "title": article["title"],
                    "author": article["author"],
                    "chunk": chunk
                }
                json.dump(record, f)
                f.write("\n")

    print(f"✅ Saved {len(articles_info)} articles to {output_path}")
