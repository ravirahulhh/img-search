#!/usr/bin/env python3
"""
Standalone crawler: collects href from <a class="display d-block"> on category pages.
Usage: pip install requests beautifulsoup4
  python crawl_links.py <base_url> <num_pages> [output_file]
Example:
  python crawl_links.py "https://9a07j.com/videos/categories/japan-korea/" 5 urls.txt
"""

import argparse
import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
TIMEOUT = 30


def make_page_url(base_url: str, page: int) -> str:
    """Build URL for a page. Page 1 = base_url, page N = base_url/N."""
    base = base_url.rstrip("/")
    if page <= 1:
        return base if base else base_url
    return f"{base}/{page}"


def fetch_page(url: str, session: requests.Session) -> str | None:
    """Fetch HTML of a page. Returns None on failure."""
    try:
        r = session.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}", file=sys.stderr)
        return None


def extract_links(html: str, base_url: str) -> list[str]:
    """Find all <a class='display d-block'> and return absolute hrefs."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", class_=lambda c: c and "display" in c and "d-block" in c):
        href = a.get("href")
        if not href or not href.strip():
            continue
        full = urljoin(base_url, href.strip())
        links.append(full)
    return links


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl category pages for video links (class='display d-block')."
    )
    parser.add_argument(
        "base_url",
        help="Category base URL, e.g. https://9a07j.com/videos/categories/japan-korea/",
    )
    parser.add_argument(
        "num_pages",
        type=int,
        help="Number of pages to crawl (1, 2, ... num_pages)",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="crawled_links.txt",
        help="Output file (one URL per line). Default: crawled_links.txt",
    )
    args = parser.parse_args()

    if args.num_pages < 1:
        print("num_pages must be >= 1", file=sys.stderr)
        sys.exit(1)

    seen: set[str] = set()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    for page in range(1, args.num_pages + 1):
        url = make_page_url(args.base_url, page)
        print(f"Crawling page {page}: {url}", file=sys.stderr)
        html = fetch_page(url, session)
        if not html:
            continue
        for link in extract_links(html, url):
            if link not in seen:
                seen.add(link)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for link in sorted(seen):
            f.write(link + "\n")

    print(f"Wrote {len(seen)} links to {args.output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
