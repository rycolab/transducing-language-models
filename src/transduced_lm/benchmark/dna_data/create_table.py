from pathlib import Path
from typing import List


def parse_accessions_from_fasta(path: str) -> List[str]:
    seen = set()
    order = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            parts = line[1:].strip().split("|")
            if len(parts) >= 2:
                acc = parts[1].strip()
                if acc and acc not in seen:
                    seen.add(acc)
                    order.append(acc)
    return order


def latex_table(accessions: List[str], cols: int = 1) -> str:

    if cols < 1:
        cols = 1
    rows = [accessions[i : i + cols] for i in range(0, len(accessions), cols)]
    colspec = " ".join(["l"] * cols)
    body_lines = []
    for r in rows:
        if len(r) < cols:
            r = r + [""] * (cols - len(r))
        body_lines.append(" & ".join(r) + r" \\")

    return (
        r"\begin{tabular}{" + colspec + "}\n"
        r"\hline"
        + "\n"
        + "\n".join(body_lines)
        + "\n"
        + r"\hline"
        + "\n"
        + r"\end{tabular}"
    )


def main():
    fasta_file = str(
        Path(__file__).parent
        / "uniprotkb_accession_A0A0A0MT78_OR_access_2025_08_20.fasta"
    )
    accessions = parse_accessions_from_fasta(fasta_file)
    table = latex_table(accessions, cols=5)
    print(table)


if __name__ == "__main__":
    main()
