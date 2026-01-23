import os
import re
import json
import sys
import time
from logger import get_logger

log = get_logger("clean_wiki_text", "test.log")

HANDLERS = {
    "GEP": lambda x: "[GE_PRICE]",
    "plink": lambda x: x,
    "Coins": lambda x: f"{x} GP",
}

def handle_template(match):
    full_content = match.group(1)
    parts = [p.strip() for p in full_content.split('|')]
    command = parts[0]
    args = parts[1] if len(parts) > 1 else ""

    if command not in HANDLERS:
        log.critical(f"STOP: No handler for '{{{{ {command} }}}}' in {match.string[:50]}...")
        sys.exit(1) 

    return HANDLERS[command](args)

def clean_wiki_text(text):
    text = re.split(r'==\s*(Gallery|Trivia|Changes|External)\s*==', text, flags=re.I)[0]
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\{\{([^{}]+)\}\}', lambda m: handle_template(m), text)
    return " ".join(text.split())

def get_cleaned_file_names(output_path):
    """Returns a set of file names already present in the output file."""
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["id"])
                except: continue
    return processed

def get_remaining_file_names(input_dir, output_path):
    """Returns file IDs that have not been cleaned"""
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    cleaned_files = get_cleaned_file_names(output_path)
    rem_files = [f for f in files if f not in cleaned_files]
    log.info(f"CLEANING START: {len(files)} total | {len(cleaned_files)} skipped | {len(rem_files)} to clean.")
    return rem_files

def main():
    input_dir = "/Users/anishganti/runescape_mini_vla/data/knowledge/01_raw"
    output_path = "/Users/anishganti/runescape_mini_vla/data/knowledge/02_cleaned/cleaned_osrs_wiki_dump.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    #files = get_remaining_file_names(input_dir, output_path)
    #print(files)
    files = ['Gold bar.txt']

    start_time = time.time()
    success_count = 0
    
    with open(output_path, "a", encoding="utf-8") as f_out:
        for i, filename in enumerate(files):
            try:
                with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f_in:
                    cleaned = clean_wiki_text(f_in.read())
                    f_out.write(json.dumps({"id": filename, "text": cleaned}) + "\n")
                    success_count += 1
                
                if (i + 1) % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    log.info(f"Metrics: Cleaned {i+1}/{len(to_process)} | Speed: {rate:.2f} files/sec")

            except Exception as e:
                log.error(f"Cleaning failed on {filename}: {str(e)}")
                sys.exit(1)

    total_time = time.time() - start_time
    log.info(f"CLEANING COMPLETE: {success_count} new files cleaned in {total_time:.2f}s.")

if __name__ == "__main__":
    main()
    