import requests

url = (
    "https://api.mavedb.org/api/v1/score-sets/"
    "urn:mavedb:00001199-a-1"
)

response = requests.get(url, timeout=30)
response.raise_for_status()

record = response.json()

print(type(record))
print(record.keys())

print("\n--- Experiment Metadata ---")

print("Title:", record["title"])
print("Accession:", record["urn"])
print("Variants:", record["numVariants"])
print("Published:", record["publishedDate"])
print("Description:", record["shortDescription"])

print("\n--- Data Structure Inspection ---")

print("targetGenes:", type(record["targetGenes"]))
print("datasetColumns:", type(record["datasetColumns"]))

print("\n--- Target Genes ---")

genes = record["targetGenes"]

print("Number of gene records:", len(genes))
print("First gene record type:", type(genes[0]))


print("\n--- Dataset Columns ---")

columns = record["datasetColumns"]

print("Column definition keys:", list(columns.keys()))
