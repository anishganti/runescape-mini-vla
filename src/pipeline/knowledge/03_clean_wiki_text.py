import re
import json
import numexpr
import mwparserfromhell

#https://runescape.wiki/w/RuneScape:Grand_Exchange_Market_Watch/Usage_and_APIs

def load_prices():
    with open('/Users/anishganti/runescape_mini_vla/metadata/prices.json') as f:
        prices = json.load(f)
        return prices

def load_page(path):
    with open(path, 'r') as file:
        content = file.read()
        return content

def get_wikicode(page):
    return mwparserfromhell.parse(page)

var = {}
prices = load_prices()

PARSER_FUNCTIONS = [
    "#dpl",
    "#explode",
    "#expr",
    "#if",
    "#ifeq",
    "#iferror",
    "#invoke",
    "#switch",
    "#tag",
    "#time",
    "#var",
    "#var_final",
    "#vardefine",
    "#vardefineecho",
]

PRICE_TEMPLATES = [
    "GEP", 
    "GEPT",
    "Alch",
    "GEPrice",
    "GETotal", 
    "GETotalqty", 
]

VAR_TEMPLATES = [
    "#vardefine", 
    "#vardefineecho",
    "#var", 
    "#var_final"
]

EXPRESSION_TEMPLATES = [
    "#expr",
    "fe", 
    "Coins", 
    "NoCoins", 
    "#explode"
]

CONDITIONAL_TEMPLATES = [
    "if", 
    "ifeq", 
    "iferror", 
    "switch"
]

MISCELLANEOUS_TEMPLATES = [
    "plink", 
    "SCP"
]

IGNORE_TEMPLATES = [
    "External", 
    "ItemSpawnTableHead", 
    "ItemSpawnTableBottom", 
    "reflist", 
    "Subject changes header", 
    "Subject changes footer"
]

TEMPLATE_GROUPS = ["VAR", "PRICE", "EXPRESSION", "MISC", "IGNORE"]

TEMPLATE_MAP = {
    "PRICE": PRICE_TEMPLATES, 
    "EXPRESSION": EXPRESSION_TEMPLATES, 
    "VAR": VAR_TEMPLATES, 
    "CONDITIONAL" : CONDITIONAL_TEMPLATES, 
    "MISC" : MISCELLANEOUS_TEMPLATES, 
    "IGNORE" : IGNORE_TEMPLATES
}

def is_function_parser(name):
    for func in PARSER_FUNCTIONS:
        if func in name:
            return True 

    return False

def match_template(name, template_type):
    if is_function_parser(str(name)):
        return True

    for pt in TEMPLATE_MAP[template_type]: 
        if name.matches(pt): 
            return True
    
    return False

def parse_template_args(template):
    name = str(template.name)

    if name.startswith("GE"):
        return ["GE"] + [str(param) for param in template.params]
    elif name.startswith("Alch"):
        return ["Alch"] + [str(param) for param in template.params]
    elif "#vardefine" in name:
        return name, name.split(":", 1)[1], str(template.params[0])
    elif "#var" in name:
        return name, name.split(":", 1)[1]
    elif name == "SCP" or name == "plink":
        return [str(param) for param in template.params]
    elif name == "Coins" or name == "NoCoins":
        return [str(param) for param in template.params]

def price_lookup(type, name):
    if type == 'GE':
        return prices[name]['price']
    elif type == 'Alch':
        return prices[name]['highalch']
    else:
        return 'Invalid price type'

def get_value(price_type, item, qty=1):
    return price_lookup(price_type, item) * qty

def get_total_value(price_type, items):
    return sum(get_value("GE", item) for item in items)

def eval_expr(expr):
    return numexpr.evaluate(expr)

def eval_coins(expr, c=False):
    return str(eval_expr(expr)) + ("coins" if c else "")

def format_num():
    pass

def get_var(func, key, value=None):
    if "#vardefine" in func: 
        var[key] = value

    if "#vardefine" in func and "#vardefineecho" not in func: 
        value = ""
    else: 
        value = var.get(key, "")
    return value

def resolve_scp(skill, level=None, link=None):
    return f"Level {level} {skill}" if level else skill

def resolve_plink(item, pic=None, text=None): 
    return text if text else item

def strip_links(wikicode):
    wikicode = re.sub(r'\[\[File:.*?\]\]', '', wikicode)
    wikicode = wikicode.replace('[[', '').replace(']]', '')
    return wikicode
    
template_handlers = {
    "Alch": get_value,
    "GEP": get_value,
    "GEPrice": get_value, 
    "GETotal": get_total_value, 
    "GETotalqty": get_total_value, 
    "#expr": eval_expr,
    "formatnum": format_num, 
    "fe": eval_expr, 
    "Coins": eval_expr, 
    "NoCoins": eval_expr, 
    "#vardefine" : get_var,
    "#vardefineecho" : get_var,
    "#var" : get_var,
    "#var_final" : get_var,
    "SCP" : resolve_scp, 
    "plink" : resolve_plink, 
    "Coins" : eval_coins, 
    "NoCoins": eval_coins,
}

def dispatch(template):
    args = parse_template_args(template)
    name = str(template.name).split(":", 1)[0]
    return template_handlers[name](*args)

def handle_template_group(wikicode, group):
    templates = wikicode.filter_templates()
    for template in templates:
        if match_template(template.name, group): 
            value = dispatch(template)                
            wikicode.replace(template, value)
   

page = load_page('/Users/anishganti/runescape_mini_vla/data/knowledge/01_raw/Gold bar.txt')
#page = load_page('/Users/anishganti/runescape_mini_vla/data/knowledge/01_raw/Magic weapons.txt')

def main(): 
    var = {}
    wikicode = get_wikicode(page)

    for group in TEMPLATE_GROUPS:
        handle_template_group(wikicode, group)

    print(strip_links(str(wikicode)))

main()