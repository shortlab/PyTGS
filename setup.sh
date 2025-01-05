#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    VENV_ACTIVATE="tgs-venv\Scripts\activate"
    PYTHON_CMD="python"
else
    VENV_ACTIVATE="source tgs-venv/bin/activate"
    PYTHON_CMD="python3"
fi

echo -e "${BLUE}Setting up TGS Analysis Tool...${NC}"

if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo -e "${BLUE}Creating virtual environment..${NC}"
$PYTHON_CMD -m venv tgs-venv

if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    . tgs-venv/Scripts/activate
else
    source tgs-venv/bin/activate
fi

echo -e "${BLUE}Upgrading pip...${NC}"
python -m pip install --upgrade pip

echo -e "${BLUE}Installing dependencies...${NC}"
if [ -f "pyproject.toml" ] || [ -f "setup.py" ]; then
    pip install -e .
else
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo "Error: Neither package files (pyproject.toml/setup.py) nor requirements.txt found"
        exit 1
    fi
fi

echo -e "${GREEN}Setup complete! To use the tool:${NC}"
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo -e "1. Activate the virtual environment with: ${BLUE}tgs-venv\Scripts\activate${NC}"
else
    echo -e "1. Activate the virtual environment with: ${BLUE}source tgs-venv/bin/activate${NC}"
fi
echo -e "2. Edit config.yaml with your data path and desired fitting parameters"
echo -e "3. Run the analysis with: ${BLUE}python main.py${NC}"
