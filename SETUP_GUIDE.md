# ⚙️ NFL Big Data Bowl 2026 - Development Setup Guide

## 🎯 Environment Requirements

### Hardware Requirements
```yaml
Minimum:
  CPU: 8 cores (Intel i7/AMD Ryzen 7)
  RAM: 32GB DDR4
  GPU: RTX 3080 (10GB VRAM) or equivalent
  Storage: 500GB NVMe SSD

Recommended:
  CPU: 16+ cores (Intel i9/AMD Ryzen 9)  
  RAM: 64GB DDR4/DDR5
  GPU: RTX 4090 (24GB VRAM) or A100 (40GB)
  Storage: 1TB+ NVMe SSD
```

### Software Requirements
```yaml
Operating System: Ubuntu 20.04+ / Windows 11 / macOS 12+
Python: 3.11+
CUDA: 11.8+ (for GPU training)
Docker: 20.10+
Git: 2.35+
```

## 🐍 Python Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-username/kaggle-nfl.git
cd kaggle-nfl

# Setup Git LFS for large files
git lfs install
git lfs track "*.pkl" "*.pth" "*.h5" "*.parquet"
```

### 2. Conda Environment (Recommended)
```bash
# Create environment from file
conda env create -f environment.yml
conda activate nfl-prediction

# Or create manually
conda create -n nfl-prediction python=3.11
conda activate nfl-prediction
```

### 3. Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies  
pip install -r docker/requirements/dev.txt

# Install package in development mode
pip install -e .
```

### 4. GPU Setup (CUDA)
```bash
# Install PyTorch with CUDA support
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install additional GPU libraries
pip install cupy-cuda11x
pip install rapids-cudf-cu11 --extra-index-url=https://pypi.nvidia.com

# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🔧 Development Tools Setup

### 1. Code Quality Tools
```bash
# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/
isort src/
mypy src/

# Run linting
flake8 src/
pylint src/
```

### 2. Testing Framework
```bash
# Run full test suite
pytest tests/ -v --cov=src/

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v

# Generate coverage report
coverage report -m
coverage html
```

### 3. Jupyter Lab Setup
```bash
# Install Jupyter extensions
pip install jupyterlab
jupyter lab build

# Install useful extensions
jupyter labextension install @jupyterlab/git
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

## 🗄️ Data Setup

### 1. Kaggle API Configuration
```bash
# Install Kaggle API
pip install kaggle

# Setup API credentials (get from kaggle.com/account)
mkdir -p ~/.kaggle
echo '{"username":"your-username","key":"your-api-key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Download Competition Data
```bash
# Download all competition data
python src/scripts/download_data.py

# Verify data integrity
python src/scripts/validate_data.py

# Generate data summary
python src/scripts/data_summary.py
```

### 3. Data Preprocessing
```bash
# Run initial data preprocessing
python src/scripts/preprocess_data.py --config configs/preprocessing.yaml

# Generate feature cache
python src/scripts/generate_features.py --cache-only
```

## 🧪 Experiment Tracking Setup

### 1. MLflow Setup
```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./experiments/mlflow

# Or use hosted MLflow
export MLFLOW_TRACKING_URI="https://your-mlflow-server.com"
```

### 2. Weights & Biases Setup
```bash
# Login to W&B
wandb login

# Initialize project
wandb init --project nfl-big-data-bowl-2026 --entity your-team
```

### 3. Configuration Management
```bash
# Validate configuration files
python -c "from src.nfl_trajectory.utils.config import load_config; load_config('configs/model_configs/baseline.yaml')"

# Generate configuration templates
python src/scripts/generate_config_templates.py
```

## 🐳 Docker Environment

### 1. Build Docker Images
```bash
# Build GPU training image
docker build -f docker/Dockerfile.gpu -t nfl-prediction:gpu .

# Build CPU inference image  
docker build -f docker/Dockerfile.cpu -t nfl-prediction:cpu .
```

### 2. Run Development Container
```bash
# Start development environment
docker-compose up -d

# Access development container
docker-compose exec gpu-trainer bash

# Run Jupyter in container
docker-compose exec gpu-trainer jupyter lab --ip=0.0.0.0 --allow-root
```

### 3. Production Deployment
```bash
# Build production image
docker build --target production -t nfl-prediction:prod .

# Run inference server
docker run -p 8080:8080 -v $(pwd)/models:/app/models nfl-prediction:prod
```

## ☁️ Cloud Setup (Optional)

### 1. AWS Setup
```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Setup S3 bucket for data
aws s3 mb s3://nfl-prediction-data

# Launch EC2 training instance
aws ec2 run-instances --image-id ami-xxx --instance-type p3.2xlarge
```

### 2. Google Cloud Setup
```bash
# Install GCloud CLI
curl https://sdk.cloud.google.com | bash

# Authenticate and set project
gcloud auth login
gcloud config set project your-project-id

# Setup Vertex AI training
gcloud ai custom-jobs create --config=deployment/gcp/training-job.yaml
```

### 3. Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/kubernetes/

# Monitor training job
kubectl get jobs -w

# Check logs
kubectl logs -f job/nfl-training-job
```

## 📊 Performance Monitoring

### 1. System Monitoring
```bash
# Install monitoring tools
pip install nvidia-ml-py3 psutil

# Monitor GPU usage
nvidia-smi -l 1

# Monitor system resources
python src/tools/monitoring/system_monitor.py
```

### 2. Model Performance Tracking
```bash
# Start model monitoring
python src/tools/monitoring/model_monitor.py --config configs/monitoring.yaml

# Generate performance reports
python src/tools/monitoring/generate_report.py
```

## 🚀 Quick Validation

### 1. Environment Check
```bash
# Run environment validation script
python src/scripts/validate_environment.py

# Expected output:
# ✅ Python 3.11+ detected
# ✅ CUDA 11.8+ available
# ✅ GPU memory: 24GB available
# ✅ All dependencies installed
# ✅ Kaggle API configured
# ✅ Data directory accessible
```

### 2. Pipeline Test
```bash
# Run end-to-end pipeline test
make test-pipeline

# Expected output:
# ✅ Data loading: PASSED
# ✅ Feature engineering: PASSED  
# ✅ Model training: PASSED
# ✅ Inference: PASSED
# ✅ Submission generation: PASSED
```

### 3. Baseline Model Training
```bash
# Train and validate baseline model
python src/scripts/train_model.py --config configs/model_configs/baseline.yaml --quick-test

# Expected output:
# Training completed in ~10 minutes
# Validation RMSE: ~2.5 (baseline target)
# Model saved to: models/checkpoints/baseline_v1.pkl
```

## 🔍 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size in config
# Enable gradient checkpointing
# Use mixed precision training
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**Data Loading Slow**
```bash
# Use SSD storage for data
# Increase num_workers in DataLoader
# Enable data caching
# Use memory mapping for large files
```

**Import Errors**
```bash
# Reinstall package in development mode
pip uninstall nfl-trajectory
pip install -e .

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Performance Optimization
```bash
# Enable mixed precision training
export TORCH_CUDNN_V8_API_ENABLED=1

# Optimize CPU performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Enable fast data loading
export TORCH_USE_CUDA_DSA=1
```

## 📚 Additional Resources

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Kaggle API Guide](https://github.com/Kaggle/kaggle-api)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Weights & Biases Guide](https://docs.wandb.ai/)

### Useful Commands Reference
```bash
# Development workflow
make setup        # Setup environment
make test         # Run tests  
make lint         # Run linting
make format       # Format code
make docs         # Generate docs

# Training workflow  
make download-data    # Download competition data
make preprocess      # Preprocess data
make train-baseline  # Train baseline model
make train-advanced  # Train advanced models
make evaluate       # Evaluate models
make submit         # Generate submission
```

This setup guide ensures you have a robust, scalable development environment optimized for competitive machine learning and capable of handling the computational demands of advanced deep learning models.