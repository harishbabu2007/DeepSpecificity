import requests
from pyjaspar import jaspardb


def get_metadata_from_pdb(pdb_id):
    """
    Fetches both UniProt IDs and Gene Symbols from the free PDBe API.
    """
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    uniprot_ids = []
    gene_symbols = []

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if not data:
                return uniprot_ids, gene_symbols

            pdb_key = list(data.keys())[0]
            uniprot_data = data[pdb_key].get("UniProt", {})

            for up_id, info in uniprot_data.items():
                uniprot_ids.append(up_id)
                # PDBe returns identifiers like 'CREB1_MOUSE'.
                # Splitting by '_' gives us the clean Gene Symbol 'CREB1'
                if "identifier" in info:
                    gene = info["identifier"].split("_")[0]
                    gene_symbols.append(gene.upper())

    except Exception as e:
        print(f"Error fetching PDBe mapping for {pdb_id}: {e}")

    return list(set(uniprot_ids)), list(set(gene_symbols))


def build_jaspar_index(release="JASPAR2024"):
    """
    Builds two fast in-memory dictionaries mapping UniProt and Genes to Motifs.
    """
    print(f"Initializing pyJASPAR ({release}) and building local indices...")
    jdb_obj = jaspardb(release=release)

    # Using default fetch_motifs() pulls the curated CORE collection cleanly
    all_motifs = jdb_obj.fetch_motifs()

    uniprot_to_motifs = {}
    gene_to_motifs = {}

    for motif in all_motifs:
        # 1. Index by UniProt ID
        if motif.acc:
            for up_id in motif.acc:
                if up_id not in uniprot_to_motifs:
                    uniprot_to_motifs[up_id] = []
                uniprot_to_motifs[up_id].append(motif)

        # 2. Index by Upper-case Gene Name
        if motif.name:
            gene_upper = motif.name.upper()
            if gene_upper not in gene_to_motifs:
                gene_to_motifs[gene_upper] = []
            gene_to_motifs[gene_upper].append(motif)

    print(
        f"Index built! Maps {len(uniprot_to_motifs)} UniProt IDs and {len(gene_to_motifs)} Gene Symbols."
    )
    return {"by_uniprot": uniprot_to_motifs, "by_gene": gene_to_motifs}


def get_motifs_for_pdb(pdb_id, indices):
    """
    Finds JASPAR motifs for a PDB ID, falling back to Gene Symbol if UniProt misses.
    """
    uniprot_ids, gene_symbols = get_metadata_from_pdb(pdb_id)
    matched_motifs = []

    # Strategy 1: Try strict matching using UniProt ID
    for up_id in uniprot_ids:
        if up_id in indices["by_uniprot"]:
            matched_motifs.extend(indices["by_uniprot"][up_id])

    # Strategy 2: If nothing found, fall back to Gene Symbol (Cross-species fix)
    if not matched_motifs:
        for gene in gene_symbols:
            if gene in indices["by_gene"]:
                matched_motifs.extend(indices["by_gene"][gene])

    # Remove any potential duplicates
    unique_motifs = {m.matrix_id: m for m in matched_motifs}.values()
    return list(unique_motifs)


# --- Execution Loop ---
if __name__ == "__main__":
    # Build your lookup indices once
    jaspar_indices = build_jaspar_index(release="JASPAR2024")

    # Query your target PDB
    example_pdb = "1gat"
    print(f"\nSearching motifs for PDB: {example_pdb}")

    motifs = get_motifs_for_pdb(example_pdb, jaspar_indices)

    if not motifs:
        print(f"No JASPAR motifs found for PDB {example_pdb}.")

    for motif in motifs:
        print(f"\n=========================================")
        print(f"SUCCESS! Found Motif ID: {motif.matrix_id} | Name: {motif.name}")
        print(f"Class: {motif.tf_class} | Family: {motif.tf_family}")
        print(f"=========================================")

        print("\n--- Position Frequency Matrix (PFM) ---")
        print(motif.counts)

        print("\n--- Position Weight Matrix (PWM / Log-Odds) ---")
        try:
            ppm = motif.counts.normalize(pseudocounts=0.5)
            pwm = ppm.log_odds()
            print(pwm)
        except Exception as e:
            print(f"Could not compute log-odds matrix: {e}")
