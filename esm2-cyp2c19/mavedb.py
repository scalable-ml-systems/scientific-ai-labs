import requests

from Bio.Seq import Seq
from io import StringIO
from Bio import SeqIO

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

print("\n--- Target Gene Fields ---")

gene = record["targetGenes"][0]

print("Gene keys:", list(gene.keys()))


print("\n--- Score Column Structure ---")

columns = record["datasetColumns"]

print("scoreColumns type:", type(columns["scoreColumns"]))
print("countColumns type:", type(columns["countColumns"]))
print("recordType value:", columns["recordType"])

gene = record["targetGenes"][0]

print("\n--- Biological Target ---")

print("Name:", gene["name"])
print("Category:", gene["category"])
print("UniProt ID:", gene["uniprotIdFromMappedMetadata"])
print("Target sequence type:", type(gene["targetSequence"]))

columns = record["datasetColumns"]

scores = columns["scoreColumns"]
counts = columns["countColumns"]

print("\n--Experimental Data Definitions ---")

print("Number of score columns:", len(scores))
print("Number of count columns:", len(counts))

if scores:
    print("First score definition type:", type(scores[0]))


scores = record["datasetColumns"]["scoreColumns"]

print("\n--- Experimental Score Columns ---")

for score in scores:
    print(score)

sequence_info = gene["targetSequence"]

print("\n--- Reference Sequence Structure ---")

print("Keys:", list(sequence_info.keys()))

sequence_info = gene["targetSequence"]

print("\n--- Reference Sequence ---")

print("Sequence type:", sequence_info["sequenceType"])
print("Sequence label:", sequence_info["label"])

sequence = sequence_info["sequence"]

print("Python type:", type(sequence))

if isinstance(sequence, str):
    print("Sequence length:", len(sequence))
    print("First 30 characters:", sequence[:30])

dna = gene["targetSequence"]["sequence"]

print("\n--- DNA to Protein ---")

print("DNA length:", len(dna))
print("Complete codons:", len(dna) // 3)
print("Remainder:", len(dna) % 3)

protein = str(Seq(dna).translate())

print("Protein length:", len(protein))
print("First 30 amino acids:", protein[:30])
print("Stop codons:", protein.count("*"))

def translate_target(dna: str) -> str:
    dna = dna.strip().upper()

    if len(dna) % 3 != 0:
        raise ValueError("DNA length is not divisible by 3")

    invalid = set(dna) - set("ACGT")

    if invalid:
        raise ValueError(f"Unexpected nucleotide symbols: {invalid}")

    translated = str(Seq(dna).translate(table=1))

    if "*" in translated[:-1]:
        raise ValueError("Internal stop codon detected")

    if translated.endswith("*"):
        translated = translated[:-1]

    return translated

sequence_info = record["targetGenes"][0]["targetSequence"]

if sequence_info["sequenceType"] != "dna":
    raise ValueError("Expected a DNA target sequence")

dna = sequence_info["sequence"]
protein = translate_target(dna)

print("\n--- Validated Translation ---")
print("DNA length:", len(dna))
print("Protein length:", len(protein))
print("First 30 amino acids:", protein[:30])
print("Last 10 amino acids:", protein[-10:])
print("Unexpected stop symbols:", protein.count("*"))

url = "https://rest.uniprot.org/uniprotkb/P33261.fasta"

response = requests.get(url, timeout=30)
response.raise_for_status()

reference = SeqIO.read(StringIO(response.text), "fasta")
uniprot_protein = str(reference.seq)

print("\n--- Reference Comparison ---")

print("UniProt identifier:", reference.id)
print("UniProt length:", len(uniprot_protein))
print("MaveDB length:", len(protein))

print("Exact sequence match:", protein == uniprot_protein)

url = "https://rest.uniprot.org/uniprotkb/P33261.fasta"

response = requests.get(url, timeout=30)
response.raise_for_status()

reference = SeqIO.read(StringIO(response.text), "fasta")

uniprot_protein = str(reference.seq)

print("\n--- Independent Reference Validation ---")

print("UniProt ID:", reference.id)
print("UniProt length:", len(uniprot_protein))
print("MaveDB length:", len(protein))
print("Exact match:", protein == uniprot_protein)
