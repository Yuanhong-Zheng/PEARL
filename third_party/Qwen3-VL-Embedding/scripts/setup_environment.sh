#!/bin/bash

set -e  # Exit on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if a virtual environment is currently activated
check_virtual_env() {
    if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_DEFAULT_ENV" ]; then
        print_error "A virtual environment is already activated!"
        if [ -n "$VIRTUAL_ENV" ]; then
            print_error "Active virtualenv: $VIRTUAL_ENV"
        fi
        if [ -n "$CONDA_DEFAULT_ENV" ]; then
            print_error "Active conda environment: $CONDA_DEFAULT_ENV"
        fi
        print_error "Please deactivate the current environment first by running:"
        print_error "  deactivate  (for virtualenv)"
        print_error "  conda deactivate  (for conda)"
        exit 1
    fi
}

# Check and install uv
install_uv() {
    if command_exists uv; then
        print_info "uv is already installed. Version: $(uv --version)"
    else
        print_warning "uv not detected. Installing..."
        
        # Install uv for Linux
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
        # Add uv to PATH for current shell
        export PATH="$HOME/.cargo/bin:$PATH"
        
        if command_exists uv; then
            print_info "uv installed successfully! Version: $(uv --version)"
        else
            print_error "Failed to install uv. Please install manually: https://github.com/astral-sh/uv"
            exit 1
        fi
    fi
}

# Setup environment
setup_environment() {
    print_info "Setting up project environment..."
    
    # Check if pyproject.toml exists
    if [ ! -f "pyproject.toml" ]; then
        print_error "pyproject.toml not found. Please run this script from the project root directory"
        exit 1
    fi
    
    # Sync dependencies (installs all dependencies except flash-attn)
    print_info "Installing project dependencies..."
    uv sync
    
    print_info "Base dependencies installed successfully"
}

# Install flash-attn with special handling
install_flash_attn() {
    print_info "Installing flash-attn (this may take several minutes)..."
    
    uv pip install flash-attn --no-build-isolation
}

# Verify installation
verify_installation() {
    print_info "Verifying environment setup..."
    
    # Display Python version
    if uv run python --version >/dev/null 2>&1; then
        PYTHON_VERSION=$(uv run python --version)
        print_info "Python version: $PYTHON_VERSION"
    else
        print_error "Python environment setup failed"
        exit 1
    fi
    
    # List installed packages
    print_info "Installed packages:"
    uv pip list
}

# Main function
main() {
    echo "=========================================="
    echo "  Environment Setup"
    echo "=========================================="
    echo ""
    
    # 0. Check if virtual environment is already activated
    check_virtual_env
    echo ""
    
    # 1. Check and install uv
    install_uv
    echo ""
    
    # 2. Setup environment
    setup_environment
    echo ""
    
    # 3. Install flash-attn
    install_flash_attn
    echo ""
    
    # 4. Verify installation
    verify_installation
    echo ""
    
    print_info "=========================================="
    print_info "Environment setup completed successfully!"
    print_info "=========================================="
    print_info ""
    print_info "To activate the environment, run:"
    print_info "  source .venv/bin/activate"
}

# Run main function
main