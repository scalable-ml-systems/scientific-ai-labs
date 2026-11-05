import re

AMINO_ACIDS = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
SINGLE_SUBSTITUTION = re.compile(r"p\.([A-Z][a-z]{2})([1-9][0-9]*)([A-Z][a-z]{2}|\*)")


def validate_protein_variant(protein_variant: str, reference_protein_sequence: str) -> dict:
    result = {
        "validation_status": "excluded",
        "validation_reason": "unsupported_variant_notation",
        "position_1based": None,
        "reference_amino_acid": None,
        "alternate_amino_acid": None,
        "variant_protein_sequence": None,
    }
    if not isinstance(protein_variant, str) or not protein_variant.strip():
        result.update(validation_status="invalid", validation_reason="missing_protein_annotation")
        return result

    match = SINGLE_SUBSTITUTION.fullmatch(protein_variant.strip())
    if match is None:
        return result

    reference_code, position_text, alternate_code = match.groups()
    if alternate_code in {"Ter", "*"}:
        result["validation_reason"] = "nonsense_variant"
        return result
    if reference_code not in AMINO_ACIDS or alternate_code not in AMINO_ACIDS:
        result["validation_reason"] = "noncanonical_amino_acid"
        return result

    position_1based = int(position_text)
    reference_amino_acid = AMINO_ACIDS[reference_code]
    alternate_amino_acid = AMINO_ACIDS[alternate_code]
    result.update(
        position_1based=position_1based,
        reference_amino_acid=reference_amino_acid,
        alternate_amino_acid=alternate_amino_acid,
    )

    index_0based = position_1based - 1
    if index_0based >= len(reference_protein_sequence):
        result.update(validation_status="invalid", validation_reason="position_out_of_range")
        return result
    if reference_protein_sequence[index_0based] != reference_amino_acid:
        result.update(validation_status="invalid", validation_reason="reference_residue_mismatch")
        return result
    if reference_amino_acid == alternate_amino_acid:
        result["validation_reason"] = "no_amino_acid_change"
        return result

    result.update(
        validation_status="accepted",
        validation_reason="",
        variant_protein_sequence=(
            reference_protein_sequence[:index_0based]
            + alternate_amino_acid
            + reference_protein_sequence[index_0based + 1:]
        ),
    )
    return result

