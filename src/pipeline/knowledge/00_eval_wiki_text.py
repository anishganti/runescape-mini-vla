import mwparserfromhell as mwp
from pathlib import Path
from collections import Counter
from tqdm import tqdm

def extract_template_names_from_file(filepath):
    """Extract just template names from a single wiki file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        wikicode = mwp.parse(content)
        template_names = []
        
        for template in wikicode.filter_templates(recursive=True):
            template_name = str(template.name).strip()
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

# Run the extraction
if __name__ == "__main__":
    wiki_directory = "/Users/anishganti/runescape_mini_vla/data/knowledge/01_raw"
    
    template_counter = collect_all_template_names(wiki_directory)
    
    # Save to templates.txt sorted by frequency
    output_file = "templates.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for name, count in template_counter.most_common():
            f.write(f"{count}\t{name}\n")
    
    print(f"\n{'='*60}")
    print(f"Found {len(template_counter)} unique template names")
    print(f"Total template instances: {sum(template_counter.values()):,}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")
    
    # Show top 20
    print(f"\nTop 20 most common templates:")
    for name, count in template_counter.most_common(20):
        print(f"{count:>6}\t{name}")
