#!/bin/sh
# Render the paper and show it in a VS Code tab (INF-2).
#
#   paper/render.sh            build paper/main.pdf and open it
#   paper/render.sh --no-open  build only
#
# All LaTeX by-products (.aux, .log, .bbl, ...) go to a scratch directory outside the repo.
# The only file written to the repo is paper/main.pdf (gitignored).
# Needs pdflatex and bibtex (MacTeX / BasicTeX) and new-aiaa.cls in paper/ (see main.tex).
set -eu

PAPER=$(cd "$(dirname "$0")" && pwd)
BUILD="${TMPDIR:-/tmp}/gnn-marl-paper-build"
PATH="/Library/TeX/texbin:$PATH"
export PATH

if ! command -v pdflatex >/dev/null || ! command -v bibtex >/dev/null; then
    echo "render: pdflatex/bibtex not found (expected in /Library/TeX/texbin)" >&2
    exit 1
fi
if [ ! -f "$PAPER/new-aiaa.cls" ] && [ -z "$(kpsewhich new-aiaa.cls)" ]; then
    echo "render: new-aiaa.cls not found. Download the AIAA LaTeX template and copy" >&2
    echo "        new-aiaa.cls and new-aiaa.bst into $PAPER/" >&2
    exit 1
fi

mkdir -p "$BUILD"
cd "$PAPER"

run_latex() {
    if ! pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
            -output-directory="$BUILD" main.tex >/dev/null; then
        echo "render: pdflatex failed:" >&2
        grep -E -A 3 '^(!|[^ ]+:[0-9]+:)' "$BUILD/main.log" >&2 || tail -n 20 "$BUILD/main.log" >&2
        if grep -q "File \`.*' not found" "$BUILD/main.log"; then
            echo "render: install a missing file with: tlmgr --usermode install <package> (see README)" >&2
        fi
        exit 1
    fi
}

run_latex
if grep -q '\\citation' "$BUILD/main.aux"; then
    if ! (cd "$BUILD" && BIBINPUTS="$PAPER:" BSTINPUTS="$PAPER:" bibtex main >/dev/null); then
        echo "render: bibtex failed:" >&2
        cat "$BUILD/main.blg" >&2
        exit 1
    fi
    run_latex
fi
run_latex

cp "$BUILD/main.pdf" "$PAPER/main.pdf"
grep -E 'LaTeX Warning: (Citation|Reference).*undefined|^Warning--' \
    "$BUILD/main.log" "$BUILD/main.blg" 2>/dev/null | sed 's|^.*/main\.[a-z]*:||' | sort -u || true
echo "render: wrote $PAPER/main.pdf"

if [ "${1:-}" != "--no-open" ]; then
    if command -v code >/dev/null; then
        code --reuse-window "$PAPER/main.pdf"
    else
        open "$PAPER/main.pdf"
    fi
fi
