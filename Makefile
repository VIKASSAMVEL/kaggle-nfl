"""
NFL Big Data Bowl 2026 - Makefile for Development Workflow
Enterprise-grade build automation for competitive machine learning
"""

# Default shell and configuration
SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
.DEFAULT_GOAL := help

# Project configuration
PROJECT_NAME := nfl-trajectory
PYTHON_VERSION := 3.11
CONDA_ENV := $(PROJECT_NAME)

# Directories
SRC_DIR := src
TESTS_DIR := tests  
DOCS_DIR := docs
DATA_DIR := data
MODELS_DIR := models
NOTEBOOKS_DIR := notebooks
CONFIG_DIR := configs

# Docker configuration
DOCKER_REGISTRY := your-registry.com
DOCKER_IMAGE := $(PROJECT_NAME)
DOCKER_TAG := latest

# Competition configuration  
KAGGLE_COMPETITION := nfl-big-data-bowl-2026-prediction
SUBMISSION_DIR := submissions

# Color codes for output
RED := \033[31m
GREEN := \033[32m  
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

##@ Help
.PHONY: help
help: ## Display this help message
	@echo "$(BLUE)NFL Big Data Bowl 2026 - Development Makefile$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make $(YELLOW)<target>$(RESET)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Environment Setup
.PHONY: setup-env setup-conda setup-pip setup-dev setup-gpu
setup-env: ## Complete environment setup (conda + dependencies + tools)
	@echo "$(BLUE)Setting up complete development environment...$(RESET)"
	$(MAKE) setup-conda
	$(MAKE) setup-dev
	$(MAKE) setup-gpu
	$(MAKE) validate-env
	@echo "$(GREEN)✅ Environment setup completed!$(RESET)"

setup-conda: ## Create and configure conda environment
	@echo "$(BLUE)Creating conda environment: $(CONDA_ENV)$(RESET)"
	conda create -n $(CONDA_ENV) python=$(PYTHON_VERSION) -y
	conda activate $(CONDA_ENV) && \
	conda install -c conda-forge numpy pandas scikit-learn jupyter -y
	@echo "$(GREEN)✅ Conda environment created$(RESET)"

setup-pip: ## Install Python dependencies
	@echo "$(BLUE)Installing Python dependencies...$(RESET)"
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	pip install -e .
	@echo "$(GREEN)✅ Python dependencies installed$(RESET)"

setup-dev: ## Install development dependencies and tools
	@echo "$(BLUE)Installing development tools...$(RESET)"
	pip install -r docker/requirements/dev.txt
	pre-commit install
	pre-commit autoupdate
	@echo "$(GREEN)✅ Development tools installed$(RESET)"

setup-gpu: ## Setup GPU dependencies (CUDA, PyTorch)
	@echo "$(BLUE)Installing GPU dependencies...$(RESET)"
	pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
	pip install torch-geometric pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
	@echo "$(GREEN)✅ GPU dependencies installed$(RESET)"

##@ Data Management
.PHONY: download-data validate-data preprocess-data generate-features clean-data
download-data: ## Download competition data from Kaggle
	@echo "$(BLUE)Downloading Kaggle competition data...$(RESET)"
	python $(SRC_DIR)/scripts/download_data.py --competition $(KAGGLE_COMPETITION)
	@echo "$(GREEN)✅ Data downloaded to $(DATA_DIR)/raw/$(RESET)"

validate-data: ## Validate downloaded data integrity
	@echo "$(BLUE)Validating data integrity...$(RESET)"
	python $(SRC_DIR)/scripts/validate_data.py
	@echo "$(GREEN)✅ Data validation completed$(RESET)"

preprocess-data: ## Run data preprocessing pipeline
	@echo "$(BLUE)Running data preprocessing...$(RESET)"
	python $(SRC_DIR)/scripts/preprocess_data.py --config $(CONFIG_DIR)/preprocessing.yaml
	@echo "$(GREEN)✅ Data preprocessing completed$(RESET)"

generate-features: ## Generate feature engineering pipeline
	@echo "$(BLUE)Generating engineered features...$(RESET)"
	python $(SRC_DIR)/scripts/generate_features.py --config $(CONFIG_DIR)/feature_configs/
	@echo "$(GREEN)✅ Feature generation completed$(RESET)"

clean-data: ## Clean processed data and cache
	@echo "$(YELLOW)Cleaning processed data and cache...$(RESET)"
	rm -rf $(DATA_DIR)/processed/*
	rm -rf $(DATA_DIR)/cache/*
	@echo "$(GREEN)✅ Data cleaned$(RESET)"

##@ Model Training
.PHONY: train-baseline train-advanced train-ensemble train-all hyperopt
train-baseline: ## Train baseline models (XGBoost, Linear)
	@echo "$(BLUE)Training baseline models...$(RESET)"
	python $(SRC_DIR)/scripts/train_model.py --config $(CONFIG_DIR)/model_configs/baseline.yaml
	@echo "$(GREEN)✅ Baseline models trained$(RESET)"

train-advanced: ## Train advanced deep learning models
	@echo "$(BLUE)Training advanced models...$(RESET)"
	python $(SRC_DIR)/scripts/train_model.py --config $(CONFIG_DIR)/model_configs/lstm.yaml
	python $(SRC_DIR)/scripts/train_model.py --config $(CONFIG_DIR)/model_configs/transformer.yaml
	python $(SRC_DIR)/scripts/train_model.py --config $(CONFIG_DIR)/model_configs/gnn.yaml
	@echo "$(GREEN)✅ Advanced models trained$(RESET)"

train-ensemble: ## Train ensemble models
	@echo "$(BLUE)Training ensemble models...$(RESET)"
	python $(SRC_DIR)/scripts/train_model.py --config $(CONFIG_DIR)/model_configs/ensemble.yaml
	@echo "$(GREEN)✅ Ensemble models trained$(RESET)"

train-all: ## Train all model types
	$(MAKE) train-baseline
	$(MAKE) train-advanced  
	$(MAKE) train-ensemble

hyperopt: ## Run hyperparameter optimization
	@echo "$(BLUE)Running hyperparameter optimization...$(RESET)"
	python $(SRC_DIR)/scripts/hyperopt.py --config $(CONFIG_DIR)/hyperopt.yaml
	@echo "$(GREEN)✅ Hyperparameter optimization completed$(RESET)"

##@ Model Evaluation
.PHONY: evaluate validate cross-validate analyze-errors
evaluate: ## Evaluate trained models
	@echo "$(BLUE)Evaluating models...$(RESET)"
	python $(SRC_DIR)/scripts/evaluate_model.py --model-dir $(MODELS_DIR)/final_models/
	@echo "$(GREEN)✅ Model evaluation completed$(RESET)"

validate: ## Run cross-validation on models
	@echo "$(BLUE)Running cross-validation...$(RESET)"
	python $(SRC_DIR)/scripts/cross_validate.py --config $(CONFIG_DIR)/validation.yaml
	@echo "$(GREEN)✅ Cross-validation completed$(RESET)"

cross-validate: validate ## Alias for validate

analyze-errors: ## Perform error analysis on predictions
	@echo "$(BLUE)Analyzing prediction errors...$(RESET)"
	python $(SRC_DIR)/scripts/error_analysis.py
	@echo "$(GREEN)✅ Error analysis completed$(RESET)"

##@ Submission Generation
.PHONY: submit generate-submission test-submission validate-submission
generate-submission: ## Generate Kaggle submission file
	@echo "$(BLUE)Generating submission file...$(RESET)"
	python $(SRC_DIR)/scripts/generate_submission.py --model $(MODELS_DIR)/final_models/ensemble.pkl
	@echo "$(GREEN)✅ Submission generated: $(SUBMISSION_DIR)/submission_$(shell date +%Y%m%d_%H%M%S).csv$(RESET)"

test-submission: ## Test submission format and validity
	@echo "$(BLUE)Testing submission format...$(RESET)"
	python $(SRC_DIR)/scripts/test_submission.py --file $(SUBMISSION_DIR)/latest_submission.csv
	@echo "$(GREEN)✅ Submission format validated$(RESET)"

validate-submission: test-submission ## Alias for test-submission

submit: generate-submission test-submission ## Generate and submit to Kaggle
	@echo "$(BLUE)Submitting to Kaggle...$(RESET)"
	kaggle competitions submit -c $(KAGGLE_COMPETITION) -f $(SUBMISSION_DIR)/latest_submission.csv -m "Automated submission $(shell date)"
	@echo "$(GREEN)✅ Submission uploaded to Kaggle$(RESET)"

##@ Code Quality
.PHONY: format lint type-check test test-unit test-integration coverage
format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	black $(SRC_DIR)/ $(TESTS_DIR)/
	isort $(SRC_DIR)/ $(TESTS_DIR)/
	@echo "$(GREEN)✅ Code formatted$(RESET)"

lint: ## Run linting with flake8 and pylint
	@echo "$(BLUE)Running linters...$(RESET)"
	flake8 $(SRC_DIR)/ --max-line-length=88 --extend-ignore=E203,W503
	pylint $(SRC_DIR)/ --disable=C0103,R0913,R0914
	@echo "$(GREEN)✅ Linting completed$(RESET)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checking...$(RESET)"
	mypy $(SRC_DIR)/ --ignore-missing-imports
	@echo "$(GREEN)✅ Type checking completed$(RESET)"

test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(RESET)"
	pytest $(TESTS_DIR)/ -v --cov=$(SRC_DIR) --cov-report=term-missing
	@echo "$(GREEN)✅ All tests passed$(RESET)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	pytest $(TESTS_DIR)/unit/ -v
	@echo "$(GREEN)✅ Unit tests passed$(RESET)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	pytest $(TESTS_DIR)/integration/ -v
	@echo "$(GREEN)✅ Integration tests passed$(RESET)"

coverage: ## Generate test coverage report
	@echo "$(BLUE)Generating coverage report...$(RESET)"
	coverage run -m pytest $(TESTS_DIR)/
	coverage report -m
	coverage html
	@echo "$(GREEN)✅ Coverage report generated: htmlcov/index.html$(RESET)"

##@ Docker Operations
.PHONY: docker-build docker-run docker-push docker-dev docker-prod
docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(RESET)"
	docker build -f docker/Dockerfile.gpu -t $(DOCKER_IMAGE):gpu-$(DOCKER_TAG) .
	docker build -f docker/Dockerfile.cpu -t $(DOCKER_IMAGE):cpu-$(DOCKER_TAG) .
	@echo "$(GREEN)✅ Docker images built$(RESET)"

docker-run: ## Run development Docker container
	@echo "$(BLUE)Starting Docker development environment...$(RESET)"
	docker-compose up -d
	@echo "$(GREEN)✅ Docker environment started$(RESET)"

docker-push: ## Push Docker images to registry
	@echo "$(BLUE)Pushing Docker images...$(RESET)"
	docker tag $(DOCKER_IMAGE):gpu-$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):gpu-$(DOCKER_TAG)
	docker tag $(DOCKER_IMAGE):cpu-$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):cpu-$(DOCKER_TAG)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):gpu-$(DOCKER_TAG)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):cpu-$(DOCKER_TAG)
	@echo "$(GREEN)✅ Docker images pushed$(RESET)"

docker-dev: ## Start development environment
	docker-compose -f docker-compose.dev.yml up -d

docker-prod: ## Start production environment  
	docker-compose -f docker-compose.prod.yml up -d

##@ Documentation
.PHONY: docs docs-build docs-serve docs-clean
docs: docs-build ## Generate documentation

docs-build: ## Build documentation with Sphinx
	@echo "$(BLUE)Building documentation...$(RESET)"
	cd $(DOCS_DIR) && sphinx-build -b html . _build/html
	@echo "$(GREEN)✅ Documentation built: $(DOCS_DIR)/_build/html/index.html$(RESET)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(RESET)"
	cd $(DOCS_DIR)/_build/html && python -m http.server 8000

docs-clean: ## Clean documentation build
	@echo "$(YELLOW)Cleaning documentation build...$(RESET)"
	rm -rf $(DOCS_DIR)/_build/
	@echo "$(GREEN)✅ Documentation cleaned$(RESET)"

##@ Monitoring & Analysis
.PHONY: monitor profile benchmark visualize
monitor: ## Start model performance monitoring
	@echo "$(BLUE)Starting performance monitoring...$(RESET)"
	python $(SRC_DIR)/tools/monitoring/model_monitor.py
	@echo "$(GREEN)✅ Monitoring started$(RESET)"

profile: ## Profile model performance
	@echo "$(BLUE)Profiling model performance...$(RESET)"
	python $(SRC_DIR)/tools/monitoring/profiler.py
	@echo "$(GREEN)✅ Profiling completed$(RESET)"

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	python $(SRC_DIR)/tools/monitoring/benchmark.py
	@echo "$(GREEN)✅ Benchmarking completed$(RESET)"

visualize: ## Generate visualizations and plots
	@echo "$(BLUE)Generating visualizations...$(RESET)"
	python $(SRC_DIR)/tools/visualization/generate_plots.py
	@echo "$(GREEN)✅ Visualizations generated$(RESET)"

##@ Cleanup & Maintenance
.PHONY: clean clean-all clean-cache clean-models clean-logs reset
clean: ## Clean temporary files and cache
	@echo "$(YELLOW)Cleaning temporary files...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	@echo "$(GREEN)✅ Temporary files cleaned$(RESET)"

clean-cache: ## Clean data processing cache
	@echo "$(YELLOW)Cleaning data cache...$(RESET)"
	rm -rf $(DATA_DIR)/cache/*
	@echo "$(GREEN)✅ Data cache cleaned$(RESET)"

clean-models: ## Clean trained models (DANGEROUS!)
	@echo "$(RED)WARNING: This will delete all trained models!$(RESET)"
	@read -p "Are you sure? (y/N): " confirm && [[ $$confirm == [yY] ]] || exit 1
	rm -rf $(MODELS_DIR)/*
	@echo "$(GREEN)✅ Models cleaned$(RESET)"

clean-logs: ## Clean log files
	@echo "$(YELLOW)Cleaning log files...$(RESET)"
	rm -rf logs/*
	@echo "$(GREEN)✅ Log files cleaned$(RESET)"

clean-all: clean clean-cache clean-logs ## Clean everything except models
	@echo "$(GREEN)✅ Full cleanup completed$(RESET)"

reset: clean-all ## Reset project to initial state (DANGEROUS!)
	@echo "$(RED)WARNING: This will reset the project to initial state!$(RESET)"
	@read -p "Are you sure? (y/N): " confirm && [[ $$confirm == [yY] ]] || exit 1
	rm -rf $(DATA_DIR)/processed/*
	rm -rf $(MODELS_DIR)/*
	rm -rf experiments/*
	@echo "$(GREEN)✅ Project reset completed$(RESET)"

##@ Validation & Testing
.PHONY: validate-env test-pipeline quick-test integration-test
validate-env: ## Validate development environment
	@echo "$(BLUE)Validating environment...$(RESET)"
	python $(SRC_DIR)/scripts/validate_environment.py
	@echo "$(GREEN)✅ Environment validation completed$(RESET)"

test-pipeline: ## Test complete ML pipeline
	@echo "$(BLUE)Testing ML pipeline...$(RESET)"
	python $(SRC_DIR)/scripts/test_pipeline.py --quick
	@echo "$(GREEN)✅ Pipeline test completed$(RESET)"

quick-test: ## Quick development test
	@echo "$(BLUE)Running quick development test...$(RESET)"
	pytest $(TESTS_DIR)/unit/ -x --tb=short
	@echo "$(GREEN)✅ Quick test completed$(RESET)"

integration-test: ## Full integration test
	@echo "$(BLUE)Running integration tests...$(RESET)"
	pytest $(TESTS_DIR)/integration/ -v --tb=long
	@echo "$(GREEN)✅ Integration tests completed$(RESET)"

##@ Competition Workflow
.PHONY: daily-workflow weekly-workflow competition-prep final-submission
daily-workflow: ## Daily development workflow
	@echo "$(BLUE)Running daily workflow...$(RESET)"
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test-unit
	$(MAKE) train-baseline
	@echo "$(GREEN)✅ Daily workflow completed$(RESET)"

weekly-workflow: ## Weekly competition workflow  
	@echo "$(BLUE)Running weekly workflow...$(RESET)"
	$(MAKE) validate-data
	$(MAKE) generate-features
	$(MAKE) train-all
	$(MAKE) evaluate
	$(MAKE) generate-submission
	@echo "$(GREEN)✅ Weekly workflow completed$(RESET)"

competition-prep: ## Final competition preparation
	@echo "$(BLUE)Preparing for competition deadline...$(RESET)"
	$(MAKE) validate-env
	$(MAKE) test-pipeline
	$(MAKE) train-ensemble
	$(MAKE) validate-submission
	@echo "$(GREEN)✅ Competition preparation completed$(RESET)"

final-submission: ## Generate final competition submission
	@echo "$(BLUE)Generating final submission...$(RESET)"
	$(MAKE) train-ensemble
	$(MAKE) generate-submission
	$(MAKE) test-submission
	@echo "$(GREEN)✅ Final submission ready!$(RESET)"
	@echo "$(YELLOW)Remember to manually submit before Dec 3, 2025 deadline!$(RESET)"

##@ Utilities
.PHONY: info status gpu-status disk-usage
info: ## Show project information
	@echo "$(BLUE)Project Information:$(RESET)"
	@echo "Name: $(PROJECT_NAME)"
	@echo "Python: $(PYTHON_VERSION)"
	@echo "Environment: $(CONDA_ENV)"
	@echo "Competition: $(KAGGLE_COMPETITION)"

status: ## Show current project status
	@echo "$(BLUE)Project Status:$(RESET)"
	@echo "Git branch: $(shell git branch --show-current)"
	@echo "Git commit: $(shell git rev-parse --short HEAD)"
	@echo "Conda env: $(shell conda info --envs | grep '*' | awk '{print $$1}')"
	@echo "GPU available: $(shell python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"

gpu-status: ## Show GPU status and usage
	@echo "$(BLUE)GPU Status:$(RESET)"
	nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
	awk -F', ' '{printf "GPU: %-20s Memory: %s/%s MB (%.1f%%) Utilization: %s%%\n", $$1, $$2, $$3, ($$2/$$3)*100, $$4}'

disk-usage: ## Show disk usage for project directories
	@echo "$(BLUE)Disk Usage:$(RESET)"
	du -sh $(DATA_DIR) $(MODELS_DIR) $(NOTEBOOKS_DIR) logs/ 2>/dev/null || true

# Include custom makefiles if they exist
-include Makefile.local