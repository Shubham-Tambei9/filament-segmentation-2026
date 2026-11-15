"""Pack the report sources into report_overleaf.zip for upload to Overleaf (official competition template).

    python report/build_overleaf_zip.py
"""
import zipfile
from pathlib import Path

REPORT = Path(__file__).resolve().parent
FILES = ["main.tex", "preamble.tex", "main.bib", "numbers.tex", "official_numbers.tex", "efficiency_numbers.tex",
         "uncertainty_numbers.tex"]
FIGURES = ["pipeline.pdf", "qualitative.pdf", "official_eval.pdf", "size_confidence.pdf"]


def main():
    out = REPORT.parent / "report_overleaf.zip"
    missing = [f for f in FILES if not (REPORT / f).exists()] + \
              [f for f in FIGURES if not (REPORT / "figures" / f).exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(REPORT / f, f)
        for f in FIGURES:
            z.write(REPORT / "figures" / f, f"figures/{f}")
    print("wrote", out, "with", len(FILES) + len(FIGURES), "files")


if __name__ == "__main__":
    main()
