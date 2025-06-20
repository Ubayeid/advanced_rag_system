SHELL := /bin/bash # Changed to bash for standard Linux commands within WSL2

.PHONY: clean create_venv install_system_deps install_python_deps setup run all

VENV_DIR = rag_env
PYTHON_EXE = $(VENV_DIR)/bin/python # Path for Linux virtual env
PIP_EXE = $(VENV_DIR)/bin/pip    # Path for Linux virtual env
REQUIREMENTS_FILE = requirements.txt

# List of core Python dependencies
PYTHON_DEPS_STRING = setuptools wheel numpy scikit-learn spacy openai>=1.14.0 python-dotenv faiss-cpu networkx plotly pandas tiktoken matplotlib sentence-transformers

clean:
	@echo "--- Cleaning up old virtual environment and generated data ---"
	@rm -rf $(VENV_DIR) # Linux equivalent of Remove-Item -Recurse -Force
	@rm -rf storage # Linux equivalent
	@rm -f knowledge_gap_report.md # Linux equivalent

create_venv:
	@echo "--- Checking/Creating virtual environment ---"
	@python3 -m venv $(VENV_DIR)

install_system_deps:
	@echo "--- Installing/Updating system-level build dependencies (build-essential, gfortran) ---"
	@sudo apt update
	@sudo apt install -y build-essential gfortran python3-venv python3-pip

install_python_deps: create_venv install_system_deps
	@echo "--- Activating virtual environment for dependency installation ---"
	@source $(VENV_DIR)/bin/activate && \
	echo "--- Generating $(REQUIREMENTS_FILE) ---" && \
	echo "$(PYTHON_DEPS_STRING)" | tr ' ' '\n' > "$(REQUIREMENTS_FILE)" && \
	echo "--- Upgrading pip, setuptools, and wheel in the environment ---" && \
	$(PIP_EXE) install --upgrade pip setuptools wheel && \
	echo "--- Clearing pip cache ---" && \
	$(PIP_EXE) cache purge && \
	echo "--- Installing/Updating Python dependencies from $(REQUIREMENTS_FILE) ---" && \
	$(PIP_EXE) install -r $(REQUIREMENTS_FILE) --ignore-installed # Added --ignore-installed

setup: clean install_python_deps
	@echo "--- Activating virtual environment for spaCy download ---"
	@source $(VENV_DIR)/bin/activate && \
	echo "--- Downloading spaCy model 'en_core_web_sm' (if not already present) ---" && \
	$(PYTHON_EXE) -m spacy download en_core_web_sm

run:
	@echo "--- Activating virtual environment for running main.py ---"
	@source $(VENV_DIR)/bin/activate && \
	echo "--- Running main.py ---" && \
	$(PYTHON_EXE) main.py

all: setup run



