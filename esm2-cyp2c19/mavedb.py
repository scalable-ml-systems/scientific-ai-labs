
import requests
from io import StringIO
from Bio.Seq import Seq
from Bio import SeqIO

SCORE_SET = "urn:mavedb:00001199-a-1"
MAVEDB_URL = f"https://api.mavedb.org/api/v1/score-sets/{SCORE_SET}"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/P33261.fasta"


def translate_target(dna: str) -> str:
    dna = dna.strip().upper()

    if len(dna) % 3 != 0:
        raise ValueError("DNA length is not divisible by 3")

    invalid = set(dna) - set("ACGT")
    if invalid:
        raise ValueError(f"Unexpected nucleotide symbols: {invalid}")

    protein = str(Seq(dna).translate(table=1))

    if "*" in protein[:-1]:
        raise ValueError("Internal stop codon detected")

    if protein.endswith("*"):
        protein = protein[:-1]

    return protein


def get_reference_protein() -> str:
    response = requests.get(MAVEDB_URL, timeout=30)
    response.raise_for_status()

    record = response.json()
    gene = record["targetGenes"][0]
    sequence_info = gene["targetSequence"]

    if gene["name"] != "CYP2C19":
        raise ValueError("Unexpected target gene")

    if sequence_info["sequenceType"] != "dna":
        raise ValueError("Expected DNA target sequence")

    protein = translate_target(sequence_info["sequence"])

    response = requests.get(UNIPROT_URL, timeout=30)
    response.raise_for_status()

    reference = SeqIO.read(StringIO(response.text), "fasta")
    uniprot_protein = str(reference.seq)

    if protein != uniprot_protein:
        raise ValueError("MaveDB and UniProt sequences do not match")

    if len(protein) != 490:
        raise ValueError("Unexpected CYP2C19 protein length")

    return protein


if __name__ == "__main__":
    protein = get_reference_protein()
    print("Protein length:", len(protein))
    print("First 30 amino acids:", protein[:30])
    print("Reference validation: PASS")

