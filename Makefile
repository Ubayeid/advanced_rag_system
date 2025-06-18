# SHELL := powershell.exe

# .PHONY: clean setup run all

# VENV_DIR = rag_env
# PYTHON_EXE = $(VENV_DIR)/Scripts/python.exe
# PIP_EXE = $(VENV_DIR)/Scripts/pip.exe

# clean:
# 	@echo "--- Cleaning up old virtual environment (if any) ---"
# 	@if (Test-Path $(VENV_DIR)) { Remove-Item -Recurse -Force $(VENV_DIR) }

# setup:
# 	@echo "--- Checking/Creating virtual environment ---"
# 	@if (-not (Test-Path $(VENV_DIR))) { python -m venv $(VENV_DIR) }
# 	@echo "--- Upgrading pip, setuptools, and wheel in the environment ---"
# 	@$(PYTHON_EXE) -m pip install --upgrade pip setuptools wheel
# 	@echo "--- Clearing pip cache ---"
# 	@$(PIP_EXE) cache purge
# 	@echo "--- Installing/Updating dependencies from requirements.txt ---"
# 	@$(PIP_EXE) install -r requirements.txt
# 	@echo "--- Downloading spaCy model 'en_core_web_sm' (if not already present) ---"
# 	@$(PYTHON_EXE) -m spacy download en_core_web_sm

# run:
# 	@echo "--- Running main.py ---"
# 	@$(PYTHON_EXE) main.py

# all: setup run

SHELL := /bin/bash # Changed to bash for standard Linux commands within WSL2

.PHONY: clean create_venv install_system_deps install_python_deps setup run all

VENV_DIR = rag_env
PYTHON_EXE = $(VENV_DIR)/bin/python # Path for Linux virtual env
PIP_EXE = $(VENV_DIR)/bin/pip    # Path for Linux virtual env
REQUIREMENTS_FILE = requirements.txt

# List of core Python dependencies
PYTHON_DEPS_STRING = setuptools wheel numpy scikit-learn spacy openai>=1.14.0 python-dotenv faiss-cpu networkx plotly pandas tiktoken matplotlib

clean:
	@echo "--- Cleaning up old virtual environment and generated data ---"
	@rm -rf $(VENV_DIR) # Linux equivalent of Remove-Item -Recurse -Force
	@rm -rf storage # Linux equivalent
	@rm -f knowledge_gap_report.md # Linux equivalent
	@echo "--- Ensuring empty data directory ---"
	@rm -rf data && mkdir data || mkdir data # Linux equivalent

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



