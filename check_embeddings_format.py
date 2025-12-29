import json

with open('embeddings/documents.json') as f:
    data = json.load(f)

print(f'Type: {type(data)}')

if isinstance(data, dict):
    print(f'Keys: {list(data.keys())}')
    print(f'Documents count: {len(data.get("documents", []))}')
else:
    print('Format: List')
    print(f'Documents count: {len(data)}')