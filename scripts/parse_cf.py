import json
with open('../scripts/create_figures.ipynb', 'r') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        print("".join(cell['source']))
