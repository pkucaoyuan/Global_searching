#!/usr/bin/env python3
"""
Extract all theorem-like environments from paper .tex files and compile
a standalone theory document. Content is verbatim identical to the
main paper — no modification.

Usage:
    python extract_theory.py              # auto-detect from main.tex
    python extract_theory.py --compile    # also run pdflatex
"""
import re
import sys
import subprocess
from pathlib import Path

PAPER_DIR = Path(__file__).parent
MAIN_TEX = PAPER_DIR / "main.tex"
OUTPUT_TEX = PAPER_DIR / "theory_extract.tex"

# Environments to extract (order matters for display)
THEORY_ENVS = [
    "assumption", "definition", "theorem", "proposition",
    "lemma", "corollary", "example", "remark",
]
# Also extract proofs that follow theorem-like environments
PROOF_ENV = "proof"

# Regex for \begin{env}...\end{env} including nested braces
def build_env_pattern(env_name):
    return re.compile(
        rf"(\\begin\{{{env_name}\}}" r".*?" rf"\\end\{{{env_name}\}})",
        re.DOTALL,
    )

def extract_preamble(main_text):
    """Extract everything from documentclass to \\begin{document}."""
    m = re.search(r"(\\documentclass.*?)\\begin\{document\}", main_text, re.DOTALL)
    if not m:
        raise ValueError("Cannot find preamble in main.tex")
    return m.group(1)

def resolve_inputs(main_text):
    """Return ordered list of .tex files from \\input{} commands."""
    files = []
    for m in re.finditer(r"\\input\{([^}]+)\}", main_text):
        path = m.group(1)
        if not path.endswith(".tex"):
            path += ".tex"
        files.append(PAPER_DIR / path)
    return files

def extract_blocks(tex_content, source_file):
    """Extract all theorem-like environments and their following proofs."""
    blocks = []

    # Build a combined pattern: any theory env, optionally followed by proof
    all_envs = THEORY_ENVS + [PROOF_ENV]
    patterns = {env: build_env_pattern(env) for env in all_envs}

    # Find all theorem-like environment positions
    env_matches = []
    for env in THEORY_ENVS:
        for m in patterns[env].finditer(tex_content):
            env_matches.append((m.start(), m.end(), env, m.group(0)))

    # Find all proof positions
    proof_matches = []
    for m in patterns[PROOF_ENV].finditer(tex_content):
        proof_matches.append((m.start(), m.end(), m.group(0)))

    # Sort by position
    env_matches.sort(key=lambda x: x[0])

    for start, end, env, text in env_matches:
        block = {"env": env, "text": text, "source": source_file.name}

        # Check if a proof follows (within 200 chars of whitespace/comments)
        after = tex_content[end:]
        # Skip whitespace and comments
        after_stripped = re.match(r"(\s*(?:%[^\n]*\n\s*)*)", after)
        gap = after_stripped.group(0) if after_stripped else ""

        # Look for proof starting right after
        proof_start_pos = end + len(gap)
        for ps, pe, proof_text in proof_matches:
            if ps == proof_start_pos:
                block["proof"] = proof_text
                break

        blocks.append(block)

    return blocks

def collect_all_labels(input_files):
    """Scan all source files and collect every \\label{...} definition."""
    all_labels = {}
    for f in input_files:
        if not f.exists():
            continue
        content = f.read_text(encoding="utf-8")
        for m in re.finditer(r"\\label\{([^}]+)\}", content):
            all_labels[m.group(1)] = f.name
    return all_labels

def find_missing_refs(doc_text, defined_labels):
    """Find all referenced labels not defined in the extracted doc."""
    # Collect all labels defined in doc_text
    doc_labels = set(re.findall(r"\\label\{([^}]+)\}", doc_text))
    # Collect all referenced labels
    refs = set()
    for m in re.finditer(r"\\(?:cref|ref|eqref)\{([^}]+)\}", doc_text):
        # Handle comma-separated labels like \cref{asm:smooth,asm:local-search}
        for label in m.group(1).split(","):
            refs.add(label.strip())
    return refs - doc_labels

def generate_stub_labels(missing_labels, all_paper_labels):
    """Generate stub \\label{} commands for missing references."""
    stubs = []
    # Categorize by prefix for readable stub sections
    categories = {}
    for label in sorted(missing_labels):
        prefix = label.split(":")[0] if ":" in label else "misc"
        categories.setdefault(prefix, []).append(label)

    for prefix, labels in sorted(categories.items()):
        for label in labels:
            src = all_paper_labels.get(label, "unknown")
            if prefix == "eq":
                stubs.append(f"% Stub equation from {src}")
                stubs.append(f"\\begin{{equation}} \\text{{(see main paper)}} \\label{{{label}}} \\end{{equation}}")
            elif prefix == "fig":
                stubs.append(f"% Stub figure from {src}")
                stubs.append(f"\\begin{{figure}}[h]\\centering\\caption{{See main paper.}}\\label{{{label}}}\\end{{figure}}")
            elif prefix == "sec":
                stubs.append(f"% Stub section from {src}")
                stubs.append(f"\\subsection{{(See main paper)}} \\label{{{label}}}")
            elif prefix == "alg":
                stubs.append(f"% Stub algorithm from {src}")
                stubs.append(f"\\begin{{algorithm}}[h]\\caption{{See main paper.}}\\label{{{label}}}\\end{{algorithm}}")
            else:
                stubs.append(f"% Stub: {label} from {src}")
                stubs.append(f"\\label{{{label}}}")
    return stubs

def generate_theory_doc(preamble, all_blocks, section_map,
                        missing_labels=None, all_paper_labels=None):
    """Generate the standalone theory .tex file."""
    lines = []
    lines.append(preamble.rstrip())
    lines.append("")

    # Add extra packages for standalone
    lines.append("% === Theory Extract: auto-generated, do not edit ===")
    lines.append(r"\usepackage{xcolor}")
    lines.append(r"\definecolor{srcgray}{gray}{0.5}")
    lines.append(r"\newcommand{\srcfile}[1]{\hfill{\tiny\color{srcgray}[\detokenize{#1}]}}")
    lines.append("")

    lines.append(r"\title{Theory Extract: GAINS}")
    lines.append(r"\author{Auto-generated from main paper}")
    lines.append(r"\date{\today}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append(r"\maketitle")
    lines.append("")
    lines.append(r"\tableofcontents")
    lines.append(r"\newpage")
    lines.append("")

    # Generate stub labels for cross-references to non-theory parts
    if missing_labels:
        lines.append("% ============================================")
        lines.append("% Stub labels for cross-references to main paper")
        lines.append("% (equations, figures, algorithms, sections)")
        lines.append("% ============================================")
        lines.append(r"\section*{Cross-Reference Stubs}")
        lines.append(r"{\small The following items are referenced by the")
        lines.append(r"theory but defined in the main paper.}")
        lines.append(r"\medskip")
        lines.append("")
        stubs = generate_stub_labels(missing_labels, all_paper_labels or {})
        lines.extend(stubs)
        lines.append(r"\newpage")
        lines.append("")

    # Group by source file, preserve section structure
    current_source = None
    for block in all_blocks:
        src = block["source"]
        if src != current_source:
            current_source = src
            section_title = section_map.get(src, src.replace(".tex", "").replace("_", " ").title())
            lines.append(f"\\section{{{section_title}}}")
            lines.append(f"\\srcfile{{{src}}}")
            lines.append("")

        # Output the environment verbatim
        lines.append(block["text"])
        lines.append("")

        # Output proof if present
        if "proof" in block:
            lines.append(block["proof"])
            lines.append("")

    # Bibliography (reuse main paper's .bib)
    lines.append(r"\bibliographystyle{plainnat}")
    lines.append(r"\bibliography{main}")
    lines.append("")
    lines.append(r"\end{document}")
    return "\n".join(lines)

def main():
    compile_flag = "--compile" in sys.argv

    # Read main.tex
    main_text = MAIN_TEX.read_text(encoding="utf-8")

    # Extract preamble
    preamble = extract_preamble(main_text)

    # Resolve input files
    input_files = resolve_inputs(main_text)

    # Section name mapping
    section_map = {
        "abstract.tex": "Abstract",
        "introduction.tex": "Introduction",
        "preliminaries.tex": "Preliminaries",
        "framework.tex": "Two-Level Framework",
        "algorithm.tex": "GAINS Algorithm and Theory",
        "general_local_search.tex": "General Local Search Operators",
        "experiments.tex": "Experiments",
        "conclusion.tex": "Conclusion",
        "appendix_mdp.tex": "Appendix: MDP Formulation",
        "appendix_proofs.tex": "Appendix: Proofs",
    }

    # Extract from each file
    all_blocks = []
    files_with_theory = []
    for f in input_files:
        if not f.exists():
            print(f"  [skip] {f.name} (not found)")
            continue
        content = f.read_text(encoding="utf-8")
        blocks = extract_blocks(content, f)
        if blocks:
            all_blocks.extend(blocks)
            files_with_theory.append(f.name)

    # Summary
    env_counts = {}
    for b in all_blocks:
        env_counts[b["env"]] = env_counts.get(b["env"], 0) + 1
    proof_count = sum(1 for b in all_blocks if "proof" in b)

    print(f"Extracted {len(all_blocks)} theory blocks from {len(files_with_theory)} files:")
    for env, count in sorted(env_counts.items()):
        print(f"  {env}: {count}")
    print(f"  proofs attached: {proof_count}")
    print(f"  source files: {', '.join(files_with_theory)}")

    # Collect all labels from the full paper
    all_paper_labels = collect_all_labels(input_files)

    # First pass: generate doc without stubs to find missing refs
    doc_draft = generate_theory_doc(preamble, all_blocks, section_map)
    missing = find_missing_refs(doc_draft, all_paper_labels)

    if missing:
        print(f"\n  cross-ref stubs: {len(missing)} ({', '.join(sorted(missing))})")

    # Second pass: generate with stubs
    doc = generate_theory_doc(preamble, all_blocks, section_map,
                              missing_labels=missing,
                              all_paper_labels=all_paper_labels)
    OUTPUT_TEX.write_text(doc, encoding="utf-8")
    print(f"\nWritten: {OUTPUT_TEX}")

    # Compile if requested
    if compile_flag:
        print("\nCompiling theory_extract.tex ...")
        # pdflatex → bibtex → pdflatex → pdflatex
        compile_cmds = [
            ["pdflatex", "-interaction=nonstopmode", "theory_extract.tex"],
            ["bibtex", "theory_extract"],
            ["pdflatex", "-interaction=nonstopmode", "theory_extract.tex"],
            ["pdflatex", "-interaction=nonstopmode", "theory_extract.tex"],
        ]
        for cmd in compile_cmds:
            result = subprocess.run(
                cmd, cwd=PAPER_DIR, capture_output=True, text=True,
            )
        # Check result
        if result.returncode == 0:
            # Check for errors in log
            errors = [l for l in result.stdout.split("\n") if "Error" in l and "LaTeX" in l]
            if errors:
                print("Warnings/Errors:")
                for e in errors:
                    print(f"  {e}")
            else:
                print("Compilation successful!")
            # Page count
            for line in result.stdout.split("\n"):
                if "Output written" in line:
                    print(f"  {line.strip()}")
        else:
            print("Compilation failed. Check theory_extract.log")
            # Show last few lines of output
            for line in result.stdout.split("\n")[-10:]:
                if line.strip():
                    print(f"  {line}")

if __name__ == "__main__":
    main()
