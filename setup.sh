#!/bin/bash

# Setup script for OMINIMO Insurance Chatbot
# Run this script to set up the environment and build the vector database

set -e

echo "=================================="
echo "OMINIMO Insurance Chatbot Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo ""
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ API key already configured (using shared key)"
else
    echo "✅ .env file found - API key already configured"
fi

# Check for PDF files
echo ""
echo "Checking for PDF documents..."
if [ ! -d "data" ]; then
    mkdir -p data
fi

pdf_count=$(find data -name "*.pdf" 2>/dev/null | wc -l)

if [ $pdf_count -eq 0 ]; then
    echo ""
    echo "⚠️  No PDF files found in data/ directory."
    echo "Please add the following PDF files to the data/ directory:"
    echo "  - MTPL_Product_Info.pdf"
    echo "  - User_Regulations.pdf"
    echo "  - Terms_and_Conditions.pdf"
    echo ""
    read -p "Press Enter after you've added the PDF files..."
fi

# Build vector database
echo ""
echo "Building vector database..."
cd src
python vector_store.py
cd ..

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To run the chatbot:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "To run evaluation:"
echo "  source venv/bin/activate"
echo "  cd src"
echo "  python evaluation.py"
echo ""
