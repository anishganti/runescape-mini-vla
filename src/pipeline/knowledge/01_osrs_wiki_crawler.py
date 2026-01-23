import requests
import time
import os
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("OSRS_Wiki_Scraper")
logger.setLevel(logging.DEBUG)

console_fmt = logging.Formatter('%(levelname)s: %(message)s')
file_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(console_fmt)

fh = RotatingFileHandler("knowledge.log", maxBytes=5*1024*1024, backupCount=2)
fh.setLevel(logging.DEBUG)
fh.setFormatter(file_fmt)

logger.addHandler(ch)
logger.addHandler(fh)

WIKI_API = "https://oldschool.runescape.wiki/api.php"
HEADERS = {"User-Agent": "LLM_Project_Scraper/1.0 (contact: anishgantis@utexs.edu)"}


def call_api(params):
    params.update({"format": "json", "maxlag": 5})
    try:
        resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"API returned status {resp.status_code}")
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
    return None

def list_pages(continue_token=None):
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": 0,
        "apfilterredir": "nonredirects",
        "aplimit": 500
    }
    if continue_token:
        params["apcontinue"] = continue_token
    
    data = call_api(params)
    if not data: return [], None
    
    titles = [p['title'] for p in data.get("query", {}).get("allpages", [])]
    next_token = data.get("continue", {}).get("apcontinue")
    return titles, next_token

def scrape_pages(titles, start_count):
    """
    titles: list of 500 titles
    start_count: how many pages we've already scraped in this session
    """
    current_count = start_count
    for i in range(0, len(titles), 50):
        batch = titles[i : i + 50]
        params = {
            "action": "query",
            "prop": "revisions",
            "titles": "|".join(batch),
            "rvprop": "content"
        }
        
        data = call_api(params)
        if data:
            pages = data.get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                if 'revisions' in pdata:
                    title = pdata['title']
                    content = pdata['revisions'][0]['*']
                    save_to_disk(title, content)
                    
                    current_count += 1
                    logger.info(f"Scraped page {current_count}: {title}")
                else:
                    logger.debug(f"Skipping {pdata.get('title')} - No content.")
        
        time.sleep(0.2)
    return current_count

def save_to_disk(title, content):
    os.makedirs("wiki_data", exist_ok=True)
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).strip()
    with open(f"wiki_data/{safe_title}.txt", "w", encoding="utf-8") as f:
        f.write(content)

def run_pipeline():
    continue_token = None
    total_scraped_session = 0
    
    if os.path.exists("checkpoint.txt"):
        with open("checkpoint.txt", "r") as f:
            continue_token = f.read().strip() or None
            logger.info(f"Resuming from token: {continue_token}")

    logger.info("Initializing OSRS Wiki Scrape...")
    
    try:
        while True:
            titles, next_token = list_pages(continue_token)
            if not titles:
                logger.info("No more titles found. Scrape complete.")
                break
            
            total_scraped_session = scrape_pages(titles, total_scraped_session)
            
            continue_token = next_token
            if continue_token:
                with open("checkpoint.txt", "w") as f:
                    f.write(continue_token)
            else:
                break
                
    except KeyboardInterrupt:
        logger.info(f"Paused. Total pages this session: {total_scraped_session}. Checkpoint saved.")

if __name__ == "__main__":
    run_pipeline()