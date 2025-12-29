# check_doc_format.py
import json

with open('embeddings/documents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

docs = data['documents']

print(f"Total: {len(docs)}")
print(f"\nFirst doc type: {type(docs[0])}")
print(f"\nFirst doc content:")
print(docs[0])

print(f"\n\nLast doc type: {type(docs[-1])}")
print(f"\nLast doc content:")
print(docs[-1])