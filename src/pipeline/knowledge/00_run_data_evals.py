import json
from collections import Counter, defaultdict
import mwparserfromhell as mwp
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = '/Users/anishganti/runescape_mini_vla/data/knowledge/01_raw'

def run_parser_eval():
    input_path = '/Users/anishganti/runescape_mini_vla/metadata/templates.txt'
    output_path = '/Users/anishganti/runescape_mini_vla/metadata/parsers.txt'
    counts = defaultdict(int)

    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                num_str, text = line.split(None, 1)
                count = int(num_str)
            except ValueError:
                continue  

            if not text.startswith("#") or ":" not in text:
                continue

            parser = text[1:].split(":", 1)[0]

            counts[parser] += count

    with open(output_path, "w") as f:
        for key, total in sorted(counts.items()):
            f.write(f"{total} {key}\n")

def extract_template_names_from_file(filepath):
    """Extract just template names from a single wiki file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        wikicode = mwp.parse(content)
        template_names = []
        
        for template in wikicode.filter_templates(recursive=True):
            template_name = str(template.name).strip()
            if "#vardefine" in template_name:
                print(filepath)
            template_names.append(template_name)            
        
        return template_names
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []

def collect_all_template_names(directory):
    """Collect all template names from all wiki files"""
    files = list(Path(directory).glob('*.txt'))
    
    template_counter = Counter()
    
    print(f"Processing {len(files)} files...")
    
    for filepath in tqdm(files):
        template_names = extract_template_names_from_file(filepath)
        template_counter.update(template_names)
    
    return template_counter

def run_template_eval():
    wiki_directory = "/Users/anishganti/runescape_mini_vla/data/knowledge/01_raw"
    
    template_counter = collect_all_template_names(wiki_directory)
    
    output_file = "templates.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for name, count in template_counter.most_common():
            f.write(f"{count}\t{name}\n")
    
    print(f"\n{'='*60}")
    print(f"Found {len(template_counter)} unique template names")
    print(f"Total template instances: {sum(template_counter.values()):,}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")
    
    print(f"\nTop 20 most common templates:")
    for name, count in template_counter.most_common(20):
        print(f"{count:>6}\t{name}")

def run_classification_eval(file_path):
    counts = Counter()
    invalid_entries = []
    valid_labels = {"KEEP", "SKIP"}
    
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                
                try:
                    data = json.loads(line)
                    label = data.get('label', 'MISSING').upper()
                    
                    # Validation Logic
                    if label not in valid_labels:
                        invalid_entries.append({"line": i + 1, "found": label})
                    
                    counts[label] += 1
                except json.JSONDecodeError:
                    invalid_entries.append({"line": i + 1, "found": "MALFORMED_JSON"})

        total = sum(counts.values())
        
        # --- Visual Breakdown ---
        print(f"\n{'='*40}")
        print(f"CLASSIFICATION BREAKDOWN (Total: {total})")
        print(f"{'='*40}")
        
        for label, count in counts.items():
            percent = (count / total) if total > 0 else 0
            bar_length = int(percent * 20)
            bar = "█" * bar_length
            print(f"{label:<10} | {bar:<20} | {count} ({percent:.1%})")
            
        # --- Validation Alerts ---
        if invalid_entries:
            print(f"\n[!] ALERT: Found {len(invalid_entries)} invalid or missing labels:")
            for entry in invalid_entries[:5]:  # Show first 5
                print(f"    - Line {entry['line']}: {entry['found']}")
            if len(invalid_entries) > 5:
                print(f"    ... and {len(invalid_entries) - 5} more.")
        else:
            print("\n[✓] Data Integrity: All labels are valid (KEEP/SKIP).")
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

#run_classification_eval('/Users/anishganti/runescape_mini_vla/metadata/classification.jsonl')
#run_parser_eval()