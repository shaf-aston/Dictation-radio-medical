#!/usr/bin/env python3
"""
Download medical reference PDFs from authorized sources.
Supports fallback URLs and progress tracking.
"""

import os
import sys
import json
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import time

def load_metadata(metadata_path):
    """Load reference metadata from JSON."""
    with open(metadata_path) as f:
        return json.load(f)

def download_pdf(url, output_path, max_retries=3, timeout=30):
    """
    Download PDF with retries and progress.
    Returns True on success, False on failure.
    """
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}: {url}")

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            req = Request(url, headers=headers)

            with urlopen(req, timeout=timeout) as response:
                # Check content length
                content_length = response.headers.get('Content-Length')
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    print(f"    Size: {size_mb:.1f} MB")

                # Download with progress
                downloaded = 0
                chunk_size = 8192
                with open(output_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Simple progress indicator
                        if content_length and downloaded % (100 * chunk_size) == 0:
                            percent = (downloaded / int(content_length)) * 100
                            print(f"    Progress: {percent:.0f}%", end='\r')

                if os.path.getsize(output_path) > 0:
                    print(f"  [OK] Downloaded: {output_path} ({os.path.getsize(output_path) / (1024*1024):.1f} MB)")
                    return True
                else:
                    print("  [FAIL] Downloaded file is empty")
                    os.remove(output_path)

        except (HTTPError, URLError) as e:
            print(f"  [FAIL] HTTP Error: {e}")
            if attempt < max_retries - 1:
                print(f"    Waiting {2**attempt}s before retry...")
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  [FAIL] Error: {e}")

    return False

def main():
    repo_root = Path(__file__).parent.parent
    metadata_path = repo_root / "data" / "medical_reference" / "metadata.json"
    output_dir = repo_root / "data" / "medical_reference"

    if not metadata_path.exists():
        print(f"Error: metadata.json not found at {metadata_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(metadata_path)

    print("=" * 70)
    print("Medical Reference PDF Downloader")
    print("=" * 70)

    downloaded = 0
    failed = 0

    for item in metadata['medical_reference_corpus']['collection']:
        if item['status'] != 'pending':
            print(f"\n[SKIP] {item['title']} (already {item['status']})")
            continue

        if 'source' not in item:
            print(f"\n[SKIP] {item['title']} (no source URL)")
            continue

        output_file = output_dir / item['file']
        if output_file.exists() and output_file.stat().st_size > 1000:
            print(f"\n[OK] {item['title']} (file exists)")
            continue

        print(f"\n[DOWNLOAD] {item['title']}")
        print(f"  Authors: {', '.join(item.get('authors', ['Unknown']))}")

        urls = [item['source']] + item.get('alternate_sources', [])

        success = False
        for url in urls:
            if download_pdf(url, str(output_file)):
                success = True
                downloaded += 1
                break

        if not success:
            print("  [FAIL] All download attempts failed")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Summary: {downloaded} downloaded, {failed} failed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
