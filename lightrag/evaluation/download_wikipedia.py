#!/usr/bin/env python3
"""
Download Wikipedia articles for LightRAG ingestion testing.

This script fetches plain text from Wikipedia articles across diverse domains
to create a test dataset with intentional entity overlap for testing:
- Entity merging and summarization
- Cross-domain relationships
- Parallel processing optimizations

Usage:
    python lightrag/evaluation/download_wikipedia.py
    python lightrag/evaluation/download_wikipedia.py --output wiki_docs/
    python lightrag/evaluation/download_wikipedia.py --domains medical,climate
"""

import argparse
import asyncio
from pathlib import Path

import httpx

# Wikipedia API endpoint (no auth required)
WIKI_API = 'https://en.wikipedia.org/w/api.php'

# User-Agent required by Wikipedia API policy
# See: https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = 'LightRAG-Test-Downloader/1.0 (https://github.com/HKUDS/LightRAG; claude@example.com)'

# Article selection by domain - chosen for entity overlap
# WHO → Medical + Climate
# Carbon/Emissions → Climate + Finance (ESG)
# Germany/Brazil → Sports + general knowledge
ARTICLES = {
    'medical': ['Diabetes', 'COVID-19'],
    'finance': ['Stock_market', 'Cryptocurrency'],
    'climate': ['Climate_change', 'Renewable_energy'],
    'sports': ['FIFA_World_Cup', 'Olympic_Games'],
}


async def fetch_article(title: str, client: httpx.AsyncClient) -> dict | None:
    """Fetch Wikipedia article text via API.

    Args:
        title: Wikipedia article title (use underscores for spaces)
        client: Async HTTP client

    Returns:
        Dict with title, content, and source; or None if not found
    """
    params = {
        'action': 'query',
        'titles': title,
        'prop': 'extracts',
        'explaintext': True,  # Plain text, no HTML
        'format': 'json',
    }
    response = await client.get(WIKI_API, params=params)

    # Check for HTTP errors
    if response.status_code != 200:
        print(f'    HTTP {response.status_code} for {title}')
        return None

    # Handle empty response
    if not response.content:
        print(f'    Empty response for {title}')
        return None

    try:
        data = response.json()
    except Exception as e:
        print(f'    JSON parse error for {title}: {e}')
        return None

    pages = data.get('query', {}).get('pages', {})

    for page_id, page in pages.items():
        if page_id != '-1':  # -1 = not found
            return {
                'title': page.get('title', title),
                'content': page.get('extract', ''),
                'source': f'wikipedia_{title}',
            }
    return None


async def download_articles(
    domains: list[str],
    output_dir: Path,
) -> list[dict]:
    """Download all articles for selected domains.

    Args:
        domains: List of domain names (e.g., ["medical", "climate"])
        output_dir: Directory to save downloaded articles

    Returns:
        List of article metadata dicts
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    articles = []

    headers = {'User-Agent': USER_AGENT}
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        for domain in domains:
            titles = ARTICLES.get(domain, [])
            if not titles:
                print(f'[{domain.upper()}] Unknown domain, skipping')
                continue

            print(f'[{domain.upper()}] Downloading {len(titles)} articles...')

            for title in titles:
                article = await fetch_article(title, client)
                if article:
                    # Save to file
                    filename = f'{domain}_{title.lower().replace(" ", "_")}.txt'
                    filepath = output_dir / filename
                    filepath.write_text(article['content'])

                    word_count = len(article['content'].split())
                    print(f'  ✓ {title}: {word_count:,} words')

                    articles.append(
                        {
                            'domain': domain,
                            'title': article['title'],
                            'file': str(filepath),
                            'words': word_count,
                            'source': article['source'],
                        }
                    )
                else:
                    print(f'  ✗ {title}: Not found')

    return articles


async def main():
    parser = argparse.ArgumentParser(description='Download Wikipedia test articles')
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='lightrag/evaluation/wiki_documents',
        help='Output directory for downloaded articles',
    )
    parser.add_argument(
        '--domains',
        '-d',
        type=str,
        default='medical,finance,climate,sports',
        help='Comma-separated domains to download',
    )
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(',')]
    output_dir = Path(args.output)

    print('=== Wikipedia Article Downloader ===')
    print(f'Domains: {", ".join(domains)}')
    print(f'Output: {output_dir}/')
    print()

    articles = await download_articles(domains, output_dir)

    total_words = sum(a['words'] for a in articles)
    print()
    print(f'✓ Downloaded {len(articles)} articles ({total_words:,} words total)')
    print(f'  Output: {output_dir}/')

    # Print summary by domain
    print('\nBy domain:')
    for domain in domains:
        domain_articles = [a for a in articles if a['domain'] == domain]
        domain_words = sum(a['words'] for a in domain_articles)
        print(f'  {domain}: {len(domain_articles)} articles, {domain_words:,} words')


if __name__ == '__main__':
    asyncio.run(main())
