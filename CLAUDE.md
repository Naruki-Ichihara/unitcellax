# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

unitcellax is a GPU-accelerated numerical material testing framework built with JAX for computational physics and scientific computing. The project uses a containerized development approach with NVIDIA JAX, NumPy, and the FEniCS ecosystem for finite element analysis.

## Container-First Development

This project is designed to be developed entirely within a Docker container. All development commands should be executed inside the container environment.

### Essential Commands

**Container Management:**
```bash
# Build and start development environment
docker compose build --no-cache
docker compose up -d

# Enter the container for development
docker exec -it unitcellax /bin/bash

# Stop the environment
docker compose down
```

**Inside Container Setup:**
```bash
# Install FEniCS (required manual step due to MPI issues)
apt install -y fenicsx

# Verify GPU access
nvidia-smi

# Package installation
pip install -e .

# Install pytest for testing
pip install pytest
```

### Python Package Structure

The project uses setuptools with configuration split across:
- `setup.py`: Minimal entry point
- `setup.cfg`: Main package metadata and dependencies
- `pyproject.toml`: Build system configuration with setuptools_scm for versioning

**Key Dependencies:**
- Core scientific stack: numpy==1.26.0 (pinned for FEniCS compatibility), scipy, matplotlib, meshio
- Finite element: petsc4py, gmsh  
- Optimization: nlopt
- Runtime: JAX (GPU-enabled), FEniCS

## Architecture

The codebase follows a scientific computing architecture:
- **Container Base:** NVIDIA JAX image (nvcr.io/nvidia/jax:25.04-py3) with GPU support
- **Development Environment:** Full X11 forwarding for GUI applications, 4GB shared memory for computational workloads
- **Package Structure:** Single `unitcellax/` package with version management via setuptools_scm

### Important Constraints

1. **FEniCS Installation:** Must be installed manually inside container due to MPI setup conflicts (see Dockerfile:24-25)
2. **NumPy Version:** Pinned to 1.26.0 in setup.cfg for FEniCS binary compatibility - do not upgrade
3. **GPU Dependencies:** All computational work assumes NVIDIA GPU availability
4. **Container Isolation:** Host file system mounted at `/workspace/` - all paths should be relative to this
5. **Display Support:** Container configured for WSL2 integration and X11 forwarding
6. **Warnings:** SWIG-related warnings from nlopt are automatically suppressed via warnings_config.py

## Development Workflow

1. Start container and enter development environment
2. Install FEniCS manually if not present
3. Install package in development mode: `pip install -e .`
4. Work within `/workspace/` (mounted from host)
5. Test GPU functionality with computational workloads
6. Consider MPI requirements for parallel computing tasks

### Testing

The test suite enforces environment validation before functional tests to ensure proper setup.

**Standard Test Execution:**
```bash
# Run all tests (environment tests run first automatically)
pytest tests/ -v

# Run only environment validation tests
pytest tests/test_environment.py -v

# Run only basis function tests (requires env validation)
pytest tests/test_basis.py -v

# Quick environment check (run directly)
python tests/test_environment.py
```

**Advanced Test Options:**
```bash
# Run specific test categories
pytest tests/test_environment.py::TestCoreDependencies -v
pytest tests/test_environment.py::TestGPUEnvironment -v
pytest tests/test_environment.py::TestFEniCSEnvironment -v
pytest tests/test_basis.py::TestGetElements -v

# Run tests by marker
pytest -m env_validation -v          # Environment tests only
pytest -m gpu -v                     # GPU-specific tests only  
pytest -m integration -v             # Integration tests only

# Show all warnings (for debugging)
pytest tests/ -v --disable-warnings

# Run with custom verbosity
pytest tests/ -vv --tb=short
```

**Test Execution Order:**
1. **Environment validation** tests run first (automatic)
2. **Dependency verification** follows specific priority order
3. **Integration tests** run only after environment is validated
4. **Test failures** in environment validation may abort remaining tests

**Test Categories:**
- **Environment Setup**: Container configuration, display support, working directory
- **Core Dependencies**: NumPy, JAX, SciPy, matplotlib, meshio, petsc4py, gmsh, nlopt
- **GPU Environment**: NVIDIA driver, JAX GPU support, GPU computations
- **FEniCS Environment**: dolfinx, ufl availability and basic functionality
- **Package Structure**: unitcellax package import and versioning
- **Basis Functions**: Element configuration, shape functions, face quadrature, reordering

### Claude Code Integration

The container can optionally include Claude Code by setting `INSTALL_CLAUDE: "true"` in docker-compose.yml. This installs Node.js and the Claude Code CLI within the container environment.