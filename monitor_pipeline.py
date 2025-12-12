#!/usr/bin/env python3
"""Monitor LightRAG pipeline processing status with timeouts and retries."""

import argparse
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


def _fetch_json(url: str, timeout: float) -> dict[str, Any]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def monitor(
    api_url: str,
    poll_interval: float = 10.0,
    request_timeout: float = 10.0,
    timeout_seconds: float = 600.0,
    max_retries: int = 5,
) -> int:
    """
    Poll the pipeline status endpoint until complete or timeout.

    Returns exit code: 0 success, 1 timeout, 2 status retries exceeded, 3 doc fetch failed.
    """
    logger.info('Monitoring LightRAG pipeline... api_url=%s', api_url)
    start = time.time()
    consecutive_errors = 0

    while True:
        elapsed = time.time() - start
        if elapsed > timeout_seconds:
            logger.warning('Monitoring timed out after %.0fs', elapsed)
            return 1

        try:
            status = _fetch_json(f'{api_url}/documents/pipeline_status', timeout=request_timeout)
        except requests.RequestException as e:
            consecutive_errors += 1
            logger.error('Failed to fetch pipeline status (%d/%d): %s', consecutive_errors, max_retries, e)
            if consecutive_errors >= max_retries:
                return 2
            time.sleep(poll_interval)
            continue
        except ValueError as e:
            consecutive_errors += 1
            logger.error('Invalid JSON from pipeline status (%d/%d): %s', consecutive_errors, max_retries, e)
            if consecutive_errors >= max_retries:
                return 2
            time.sleep(poll_interval)
            continue

        consecutive_errors = 0
        busy = bool(status.get('busy', False))
        pending = bool(status.get('request_pending', False))
        msg = str(status.get('latest_message', ''))[:80]
        batch = f'{status.get("cur_batch", 0)}/{status.get("batchs", 0)}'
        logger.info('[%s] batch=%s busy=%s pending=%s | %s', time.strftime('%H:%M:%S'), batch, busy, pending, msg)

        if not busy and not pending:
            try:
                docs = _fetch_json(f'{api_url}/documents', timeout=request_timeout)
                doc_count = len(docs.get('documents', []))
                logger.info('Pipeline complete. Documents indexed: %d', doc_count)
                return 0
            except (requests.RequestException, ValueError) as e:
                logger.error('Pipeline finished but failed to fetch documents: %s', e)
                return 3

        time.sleep(poll_interval)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Monitor LightRAG pipeline processing status.')
    parser.add_argument(
        '--api-url',
        default=os.getenv('API_URL', 'http://localhost:9621'),
        help='Base URL of the LightRAG API (default: env API_URL or http://localhost:9621)',
    )
    parser.add_argument('--interval', type=float, default=10.0, help='Polling interval in seconds (default: 10)')
    parser.add_argument('--request-timeout', type=float, default=10.0, help='Per-request timeout in seconds')
    parser.add_argument('--timeout', type=float, default=600.0, help='Overall timeout in seconds (default: 600)')
    parser.add_argument(
        '--max-retries',
        type=int,
        default=5,
        help='Maximum consecutive request failures before exiting (default: 5)',
    )
    parser.add_argument(
        '--log-level',
        default=os.getenv('LOG_LEVEL', 'INFO'),
        help='Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    exit_code = monitor(
        api_url=args.api_url,
        poll_interval=args.interval,
        request_timeout=args.request_timeout,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )
    raise SystemExit(exit_code)
