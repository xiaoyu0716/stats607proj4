#!/bin/bash
# Compile report-Qiu.md to PDF using pandoc

set -e  # Exit on error

cd "$(dirname "$0")"

echo "Compiling report-Qiu.md to PDF..."

# Check if pandoc is available
if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc is not installed. Please install pandoc first."
    exit 1
fi

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex is not installed. Please install LaTeX first."
    exit 1
fi

# Check if images exist
if [ ! -f "exps/coverage.png" ]; then
    echo "Warning: exps/coverage.png not found"
fi

if [ ! -f "exps/recon_output.png" ]; then
    echo "Warning: exps/recon_output.png not found"
fi

# Compile with pandoc
pandoc report-Qiu.md \
    -o report-Qiu.pdf \
    --pdf-engine=pdflatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V documentclass=article \
    --toc-depth=1 \
    --standalone

if [ -f "report-Qiu.pdf" ]; then
    echo "✅ Successfully compiled report-Qiu.pdf"
    ls -lh report-Qiu.pdf
else
    echo "❌ Failed to compile PDF"
    exit 1
fi

