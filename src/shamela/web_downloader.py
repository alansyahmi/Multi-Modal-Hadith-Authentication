r"""
Web Downloader — resilient page-by-page scraper for shamela.ws.

Cloudflare bypass strategy:

  Tier 1 (RECOMMENDED): Connect to your REAL Chrome browser via CDP
    1. Close all Chrome windows
    2. Run this to start Chrome with remote debugging:
         "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
    3. Open https://shamela.ws and solve the Cloudflare challenge ONCE
    4. Leave that tab open. Run the downloader:
         python run_shamela_extract.py --book-id 3722 --method web
    The scraper connects to your authenticated Chrome — Cloudflare sees a real browser.

  Tier 2: Load the Shamela Chrome extension into an automated browser
    Uses Playwright with the pre-loaded extension to export via the extension's API.
    (Slower, requires more setup.)

Features:
  - SQLite checkpointing for resume after interruption
  - Configurable request delay (default 1.5s)
  - Per-page raw HTML caching
  - HTML-to-text extraction using BeautifulSoup
  - Per-volume text grouping
  - Auto-detects Cloudflare challenges and pauses gracefully
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Optional

from .checkpoint import CheckpointStore
from .models import BookMeta, ExtractionConfig


# ── constants ──────────────────────────────────────────────────

SHAMELA_BASE = "https://shamela.ws"
BOOK_PATH = "book"

BATCH_LOG_INTERVAL = 25  # Print progress every N pages

# Path to the Chrome extension in the project
CHROME_EXT_PATH = Path(__file__).resolve().parent.parent.parent / "shamela-epub-exporter-1.4.2-chrome"


# ── session creation (Tiers 1 & 2) ─────────────────────────────

def _create_session(use_browser: bool = False) -> Any:
    """
    Create a browser session for shamela.ws.

    Tier 1: Playwright automated browser (useful for solving Cloudflare interactively).
    Tier 2: CDP to user's existing Chrome.
    """
    if use_browser:
        return _create_extension_session()

    # Connect to user's running Chrome via CDP
    if _try_cdp_connection(port=9222):
        return _create_cdp_session(port=9222)

    print("  CDP connection to existing Chrome not found on port 9222.")
    print()
    instructions = r"""
  === HOW TO START CHROME WITH REMOTE DEBUGGING ===
  1. Close ALL Chrome windows
  2. Press Win+R, paste this, and press Enter:

     "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222

  3. In the new Chrome window that opens, go to https://shamela.ws/book/3722/1
  4. Solve the Cloudflare challenge if needed (just once)
  5. Leave that tab open, come back here, and re-run.
"""
    print(instructions)

    raise RuntimeError("No connection to running Chrome. Follow the instructions above or run with --browser.")


# ── Tier 1: CDP (Chrome DevTools Protocol) ─────────────────────

def _try_cdp_connection(port: int = 9222) -> bool:
    """Check if a Chrome instance is listening on the debugging port."""
    import urllib.request
    import urllib.error
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=3)
        return resp.status == 200
    except (urllib.error.URLError, ConnectionRefusedError, Exception):
        return False


def _create_cdp_session(port: int = 9222) -> Any:
    """
    Connect to the user's running Chrome via CDP.

    Uses Playwright's connect_over_cdp to attach to the already-authenticated
    browser. Cloudflare trusts it because it IS the user's real browser.
    """
    from playwright.sync_api import sync_playwright, BrowserContext, Page

    class _CdpSession:
        def __init__(self, port: int):
            self._pw = sync_playwright().start()
            print(f"  Connecting to Chrome on port {port}...")
            self._browser = self._pw.chromium.connect_over_cdp(
                f"http://127.0.0.1:{port}"
            )

            # Use an existing context or create a new one
            contexts = self._browser.contexts
            if contexts:
                self._context: BrowserContext = contexts[0]
                print(f"  Attached to existing browser context")
            else:
                self._context = self._browser.new_context()
                print(f"  Created new browser context")

        def get(self, url: str, *, headers=None, timeout: int = 30):
            page = self._context.new_page()
            try:
                if headers:
                    page.set_extra_http_headers(headers)
                page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
                # Wait a moment for the .nass content to load
                try:
                    page.wait_for_selector(".nass", timeout=5000)
                except Exception:
                    pass
                html = page.content()
                return _Response(html, 200)
            finally:
                page.close()

        def close(self):
            self._pw.stop()

    class _Response:
        def __init__(self, text: str, status: int):
            self.text = text
            self.status_code = status

    print("  Connected! Using your real Chrome browser.\n")
    return _CdpSession(port)


# ── Tier 2: Playwright with embedded extension ─────────────────

def _create_extension_session() -> Any:
    """
    Launch Playwright Chromium with the Shamela EPUB Exporter extension loaded.

    The extension provides content scripts that match https://shamela.ws/book/*,
    giving us a more reliable scraping path.
    """
    from playwright.sync_api import sync_playwright, BrowserContext

    ext_path = str(CHROME_EXT_PATH.resolve())

    if not CHROME_EXT_PATH.is_dir():
        print(f"  Extension not found at: {ext_path}")
        print("  Falling back to plain Playwright...")

    class _ExtensionSession:
        def __init__(self):
            self._pw = sync_playwright().start()
            launch_args = [
                f"--disable-extensions-except={ext_path}",
                f"--load-extension={ext_path}",
            ]
            if CHROME_EXT_PATH.is_dir():
                print(f"  Launching Chrome with Shamela extension loaded...")
            else:
                print(f"  Launching plain Playwright Chromium...")
                launch_args = []

            self._browser = self._pw.chromium.launch(
                headless=False,
                args=launch_args,
            )
            self._context: BrowserContext = self._browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                locale="ar",
            )

        def get(self, url: str, *, headers=None, timeout: int = 30):
            page = self._context.new_page()
            try:
                try:
                    from playwright_stealth import stealth_sync
                    stealth_sync(page)
                except ImportError:
                    pass
                if headers:
                    page.set_extra_http_headers(headers)
                try:
                    page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
                except Exception:
                    # Cloudflare challenge page often causes navigation timeouts, continue to check content
                    pass
                # Wait for content
                try:
                    page.wait_for_selector(".nass", timeout=5000)
                except Exception:
                    pass
                try:
                    html = page.content()
                except Exception:
                    html = ""
                
                # Check for Cloudflare challenge and wait if detected
                if is_challenge_page(html):
                    print("\n  [CLOUDFLARE DETECTED] Please solve the challenge in the browser window...")
                    start_time = time.time()
                    # Wait up to 120 seconds
                    while is_challenge_page(html) and (time.time() - start_time) < 120:
                        time.sleep(2)
                        try:
                            html = page.content()
                        except Exception:
                            html = ""
                    
                    if is_challenge_page(html):
                        print("  [TIMEOUT] Cloudflare challenge not solved in 120 seconds.")
                    else:
                        print("  [SUCCESS] Challenge solved! Continuing...")
                        time.sleep(2)
                        try:
                            html = page.content()
                        except Exception:
                            html = ""
                
                return _Response(html, 200)
            finally:
                page.close()

        def close(self):
            self._browser.close()
            self._pw.stop()

    class _Response:
        def __init__(self, text: str, status: int):
            self.text = text
            self.status_code = status

    return _ExtensionSession()


# ── page fetching ──────────────────────────────────────────────

def _fetch_page(
    session: Any,
    book_id: int,
    page: int,
) -> tuple[int, str]:
    """
    Fetch a single shamela.ws book page using the browser session.

    Returns (status_code, html_text).
    Status is always 200 for browser-based sessions — we check
    content validity via is_challenge_page().
    """
    url = f"{SHAMELA_BASE}/{BOOK_PATH}/{book_id}/{page}"
    referrer = f"{SHAMELA_BASE}/{BOOK_PATH}/{book_id}"

    try:
        resp = session.get(url, headers={"Referer": referrer}, timeout=30)
        return resp.status_code, resp.text
    except Exception as e:
        return -1, str(e)


# ── page count discovery ───────────────────────────────────────

def _discover_total_pages(session: Any, book_id: int) -> int:
    """
    Fetch page 1 and extract the total page count from pagination.
    """
    import re

    print("  Fetching page 1...")
    status, html = _fetch_page(session, book_id, 1)
    if status != 200 or not html:
        raise RuntimeError(
            f"Cannot fetch page 1 of book {book_id}. "
            f"Got status {status}."
        )

    if is_challenge_page(html):
        raise RuntimeError(
            "Got a Cloudflare challenge page instead of book content.\n"
            "  Make sure you opened https://shamela.ws in your debug Chrome\n"
            "  and solved the Cloudflare challenge before running this."
        )

    # Strategy 1: Find all page number links in pagination
    matches = re.findall(rf'/book/{book_id}/(\d+)', html)
    if matches:
        page_nums = sorted(set(int(m) for m in matches))
        if page_nums:
            total = page_nums[-1]
            print(f"  Found page nav links: max = {total}")
            return total

    # Strategy 2: Input field value
    match = re.search(r'value="(\d+)"[^>]*id="?fld_goto_bottom"?', html)
    if match:
        total = int(match.group(1))
        print(f"  Found input field: max = {total}")
        return total

    return 1


# ── HTML-to-text extraction ────────────────────────────────────

def extract_text_from_html(html: str) -> str:
    """
    Extract the main book text from a shamela.ws HTML page.

    Uses the .nass selector (matching the Chrome extension's approach),
    stripping navigation, copy buttons, and empty elements.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    nass = soup.select_one(".nass")
    if not nass:
        for selector in ("#book-text", ".BookBody", ".entry-content", "article"):
            nass = soup.select_one(selector)
            if nass:
                break
        if not nass:
            nass = soup.body
            if not nass:
                return ""

    # Remove navigation and UI elements
    for tag in nass.select(
        "a.btn_tag, div.text-left, div.betaka-index, "
        ".s-nav, script, style, nav, "
        "input, button, select, "
        "span:empty, div:empty"
    ):
        tag.decompose()

    # Remove inline font-size styles
    for tag in nass.find_all(style=True):
        style = tag.get("style", "")
        if "font-size" in style.lower():
            del tag["style"]

    # Get paragraphs
    paragraphs: list[str] = []
    for p in nass.select("p, div.nass > div"):
        text = p.get_text(separator=" ", strip=True)
        if text and len(text) > 2:
            paragraphs.append(text)

    if not paragraphs:
        full_text = nass.get_text(separator="\n", strip=True)
        paragraphs = [l for l in full_text.split("\n") if l.strip()]

    return "\n\n".join(paragraphs)


# ── main download orchestration ────────────────────────────────

def download_book(
    config: ExtractionConfig,
    checkpoint: CheckpointStore,
) -> tuple[dict[int, str], BookMeta]:
    """
    Download a complete book from shamela.ws with checkpointing.

    Returns (volumes_dict, book_meta) where volumes_dict maps
    volume_number → raw_text.
    """
    book_id = config.book_id
    output_dir = config.output_dir

    print(f"\n{'='*60}")
    print(f"  Downloading book {book_id} from shamela.ws")
    print(f"  Output: {output_dir}")
    print(f"  Delay:  {config.delay_seconds}s between requests")
    print(f"  Resume: {'yes' if config.resume else 'no'}")
    print(f"{'='*60}\n")

    # 1. Create session — this connects to your browser or raises
    session = _create_session(use_browser=config.use_browser)
    print()

    # 2. Discover total pages
    print("  Discovering total pages...")
    total_pages = _discover_total_pages(session, book_id)
    print(f"  Total pages: {total_pages:,}")

    if config.end_page and config.end_page < total_pages:
        total_pages = config.end_page
        print(f"  Capped to:   {total_pages:,} (--end-page)")

    # 3. Initialize checkpoint
    start_page = config.start_page
    if not config.resume:
        checkpoint.reset_all(book_id)
        print("  Fresh start (--no-resume).")

    for p in range(start_page, total_pages + 1):
        if not checkpoint.is_downloaded(book_id, p):
            checkpoint.mark_pending(book_id, p)

    done, total, failed = checkpoint.get_progress(book_id)
    print(f"  State: {done} done, {failed} failed, {total - done - failed} pending")

    # 4. Download pages
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    consecutive_failures = 0
    last_log = 0

    print("\n  Downloading pages...")

    while True:
        page = checkpoint.get_next_pending(book_id)
        if page is None:
            break

        page_file = pages_dir / f"page_{page:05d}.html"

        status, html = _fetch_page(session, book_id, page)

        if status == 200 and html and not is_challenge_page(html):
            page_file.write_text(html, encoding="utf-8")
            content_hash = hashlib.md5(html.encode()).hexdigest()
            checkpoint.mark_downloaded(book_id, page, content_hash)
            consecutive_failures = 0
        else:
            reason = "Cloudflare challenge" if is_challenge_page(html) else f"HTTP {status}"
            checkpoint.mark_failed(book_id, page, reason)
            consecutive_failures += 1
            print(f"    Page {page}: {reason}")
            if consecutive_failures >= 5:
                print(
                    f"\n  ABORTING: {consecutive_failures} consecutive failures. "
                    f"Your Cloudflare session may have expired."
                )
                print(f"  Refresh the browser tab and run with --resume.")
                break

        # Progress logging
        done_now, _, _ = checkpoint.get_progress(book_id)
        if done_now - last_log >= BATCH_LOG_INTERVAL or page == total_pages:
            pct = done_now / total_pages * 100
            print(f"    Page {page}/{total_pages} ({pct:.1f}%) — {done_now} saved")
            last_log = done_now

        # Rate-limit delay
        if config.delay_seconds > 0:
            time.sleep(config.delay_seconds)

    # Close session
    if hasattr(session, "close"):
        session.close()

    done_final, total_final, failed_final = checkpoint.get_progress(book_id)
    print(f"\n  Download complete: {done_final} pages, {failed_final} failed")

    # 5. Extract text from downloaded HTML
    print("\n  Extracting text from HTML...")
    volumes = _build_volumes_from_pages(pages_dir, book_id, done_final)

    # 6. Build metadata
    meta = BookMeta(
        book_id=book_id,
        title=f"Book {book_id}",
        author="Unknown",
        volumes=len(volumes),
        total_pages=done_final,
        source_url=f"{SHAMELA_BASE}/{BOOK_PATH}/{book_id}",
    )

    total_chars = sum(len(t) for t in volumes.values())
    print(f"  Volumes: {len(volumes)}")
    print(f"  Total chars: {total_chars:,}\n")

    return volumes, meta


def is_challenge_page(html: str) -> bool:
    """Check if the HTML is a Cloudflare challenge page, not book content."""
    if not html:
        return True
    markers = [
        "cf-challenge",
        "cf-wrapper",
        "cf-browser-verification",
        "Just a moment",
        "لحظة",
        "التحقق من الأمان",
        "Checking your browser",
    ]
    lower = html.lower()
    return any(m.lower() in lower for m in markers)


def _build_volumes_from_pages(
    pages_dir: Path,
    book_id: int,
    page_count: int,
) -> dict[int, str]:
    """Read saved HTML pages, extract text, and group by detected volume."""
    from bs4 import BeautifulSoup
    import re

    volumes: dict[int, list[str]] = {}
    current_volume = 1

    for p in range(1, page_count + 1):
        page_file = pages_dir / f"page_{p:05d}.html"
        if not page_file.exists():
            continue

        html = page_file.read_text(encoding="utf-8")

        # Detect volume from HTML
        try:
            soup = BeautifulSoup(html, "lxml")
            part_btn = soup.select_one("#fld_part_top ~ div button, .part-btn")
            if part_btn:
                vol_text = part_btn.get_text(strip=True)
                nums = re.findall(r'\d+', vol_text)
                if nums:
                    current_volume = int(nums[0])
        except Exception:
            pass

        text = extract_text_from_html(html)
        if text and not is_challenge_page(html):
            if current_volume not in volumes:
                volumes[current_volume] = []
            volumes[current_volume].append(text)

    return {
        vol: "\n\n".join(texts)
        for vol, texts in sorted(volumes.items())
    }
