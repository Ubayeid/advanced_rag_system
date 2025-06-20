# Use bash shell for commands
SHELL := /bin/bash

# Define common variables
VENV_DIR = rag_env
PYTHON_BIN = $(VENV_DIR)/bin/python
PIP_BIN = $(VENV_DIR)/bin/pip
REQUIREMENTS_FILE = requirements.txt

# Phony targets to prevent conflicts with file names
.PHONY: all clean venv_create install_system_deps install_python_deps spacy_download run

# Default target
all: setup run

# Cleans up the virtual environment and generated data
clean:
	@echo "--- Cleaning up old virtual environment and generated data ---"
	@rm -rf $(VENV_DIR) storage knowledge_gap_report.md

# Creates the Python virtual environment
venv_create:
	@echo "--- Checking/Creating virtual environment ---"
	@python3 -m venv $(VENV_DIR)

# Installs system-level dependencies for Python packages
install_system_deps:
	@echo "--- Installing/Updating system-level build dependencies ---"
	@sudo apt update
	@sudo apt install -y build-essential gfortran python3-venv python3-pip

# Installs Python dependencies from requirements.txt
install_python_deps: venv_create install_system_deps
	@echo "--- Activating virtual environment for Python dependency installation ---"
	# All commands here are on one logical line, preceded by a single TAB
	@$(PIP_BIN) install --upgrade pip setuptools wheel && \
	$(PIP_BIN) cache purge && \
	echo "--- Installing Python dependencies from $(REQUIREMENTS_FILE) ---" && \
	$(PIP_BIN) install -r $(REQUIREMENTS_FILE) --ignore-installed

# Downloads the spaCy model
spacy_download:
	@echo "--- Activating virtual environment for spaCy model download ---"
	# All commands here are on one logical line, preceded by a single TAB
	@$(PYTHON_BIN) -m spacy download en_core_web_sm

# Sets up the entire environment (combines venv creation, python deps, and spaCy download)
setup: clean install_python_deps spacy_download

# Runs the main application
run:
	@echo "--- Activating virtual environment for running main.py ---"
	# All commands here are on one logical line, preceded by a single TAB
	@echo "--- Running main.py ---" && \
	$(PYTHON_BIN) main.py