import os
import json
import asyncio
from datetime import datetime
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from logger import get_logger

logger = get_logger('classify_wiki_text', 'knowledge.log')
client = AsyncOpenAI(api_key="sk-60eac209b2ab445ebf15446fd33766bf", base_url="https://api.deepseek.com")

RAW_DIR = "/Users/anishganti/runescape_mini_vla/data/knowledge/01_raw"
META_DIR = "/Users/anishganti/runescape_mini_vla/metadata"
OUTPUT_FILE = os.path.join(META_DIR, "classification.jsonl")

CONCURRENCY_LIMIT = 100
SYSTEM_PROMPT = """
### ROLE
You are a RuneScape Data Auditor. Your job is to classify Wiki pages into "KEEP" (data worth extracting for a LLM training set) or "SKIP" (noise, meta-pages, or low-information stubs).

### CLASSIFICATION CRITERIA
- KEEP: Pages containing specific stats (levels, combat bonuses), drop tables, quest steps, or lore paragraphs.
- SKIP: 
  1. Disambiguation pages (lists of other pages).
  2. User profiles or "Talk" pages.
  3. Sparse stubs (fewer than 3 unique facts).
  4. Wiki categories or technical templates.

### HANDLING AMBIGUITY
If a file contains ONLY an item name (e.g., "Abyssal whip") with no stats or descriptions, classify as SKIP. We only want pages with context.

### EXAMPLES
Input: "Abyssal whip - an iconic weapon requiring 70 attack." -> KEEP (Contains stats/requirements)
Input: "Abyssal whip (disambiguation) - see also Abyssal whip (item), Abyssal whip (cosmetic)." -> SKIP (Meta-page)
Input: "User:Zezima - I like to play RS." -> SKIP (User profile)
Input: "Iron Ore - A rock found in Rimmington." -> KEEP (Location data)

### RESPONSE FORMAT (JSON)
{
  "reasoning": "Briefly explain if facts are present or if it's a meta/sparse page.",
  "label": "KEEP/SKIP"
}
"""

write_queue = asyncio.Queue()

async def get_processed_files():
    if not os.path.exists(OUTPUT_FILE): return set()
    with open(OUTPUT_FILE, 'r') as f:
        return {json.loads(line)['filename'] for line in f}

async def file_writer_worker():
    """Background worker that pulls from the queue and writes to disk."""
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        while True:
            result = await write_queue.get()
            if result is None: # 
                break
            
            meta_entry = {
                "filename": result["filename"],
                "label": result["label"],
                "reasoning": result["reasoning"]
            }
            f.write(json.dumps(meta_entry) + "\n")
            f.flush() 
            
            m = result['metrics']
            logger.info(f"Classified: {result['filename']} | Cost: ${m['cost_usd']:.5f} | Hit: {m['cache_hit']}")
            
            hit_ratio = (m['cache_hit'] / m['total_input']) * 100 if m['total_input'] > 0 else 0
            if hit_ratio < 50:
                logger.warning(f"Low cache hit for {result['filename']}: {hit_ratio:.1f}%")
            
            write_queue.task_done()

async def classify_page_async(filename, text, semaphore):
    """Producer: Performs the API call with rate limiting."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"FILENAME: {filename}\nCONTENT: {text[:2500]}"}
                ],
                response_format={'type': 'json_object'}
            )
            
            usage = response.usage
            hit = getattr(usage, 'prompt_cache_hit_tokens', 0)
            miss = getattr(usage, 'prompt_cache_miss_tokens', 0)
            cost = (hit * 0.000000028) + (miss * 0.00000028) + (usage.completion_tokens * 0.00000042)
            
            res_data = json.loads(response.choices[0].message.content)
                        
            result = {
                "filename": filename,
                "label": res_data.get("label", "SKIP").upper(),
                "reasoning": res_data.get("reasoning", "N/A"),
                "metrics": {
                    "cache_hit": hit,
                    "cache_miss": miss,
                    "total_input": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "cost_usd": cost 
                }
            }
            await write_queue.put(result)
            
        except Exception as e:
            logger.error(f"Failed to process {filename}: {str(e)}")

async def main():
    os.makedirs(META_DIR, exist_ok=True)
    processed = await get_processed_files()
    all_files = [f for f in os.listdir(RAW_DIR) if f not in processed]
    
    logger.info(f"Starting async classification. Total files: {len(all_files)}")

    writer_task = asyncio.create_task(file_writer_worker())
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    tasks = []
    for filename in all_files:
        with open(os.path.join(RAW_DIR, filename), 'r', encoding='utf-8') as f:
            content = f.read()
        tasks.append(classify_page_async(filename, content, semaphore))

    await tqdm.gather(*tasks)
    await write_queue.put(None)
    await writer_task
    
    logger.info("Classification complete.")

if __name__ == "__main__":
    asyncio.run(main())