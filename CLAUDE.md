## Core Development Rules
This document contains critical information about working with this codebase. Follow these guidelines precisely.

1. Package Management
   - use pip

2. Code Quality
   - Type hints required for all code
   - Public APIs must have docstrings
   - Functions must be focused and small
   - Follow existing patterns exactly
   - Line length: 88 chars maximum

3. Testing Requirements
   - Framework: `pytest`
   - New features require tests
   - Bug fixes require regression tests

## git
   ```
   git add .
   git commit -m"<commit massage>"
   git push origin main
   ```


## Python Tools

## Error Resolution

1. CI Failures
   - Fix order:
     1. Formatting
     2. Type errors
     3. Linting
   - Type errors:
     - Get full line context
     - Check Optional types
     - Add type narrowing
     - Verify function signatures

2. Common Issues
   - Line length:
     - Break strings with parentheses
     - Multi-line function calls
     - Split imports
   - Types:
     - Add None checks
     - Narrow string types
     - Match existing patterns

3. Best Practices
   - Run formatters before type checks
   - Keep changes minimal
   - Follow existing patterns
   - Document public APIs
   - Test thoroughly

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

unitcellax is a containerized development environment for computational physics/scientific computing, specifically designed to work with JAX, NumPy, and FEniCSx. The project is currently in its initial setup phase.

## Development Environment

This project is designed to be developed entirely within a Docker container that provides:
- NVIDIA JAX with GPU support
- Scientific computing libraries (NumPy 1.26.0)
- FEniCS ecosystem for finite element analysis
- GUI support through X11 forwarding

### Container Setup

The development environment runs in a Docker container with GPU acceleration. To start the development environment:

```bash
docker compose build --no-cache 
docker compose up
```

The container mounts the project directory to `/workspace/` and provides:
- GPU access via NVIDIA runtime
- X11 forwarding for GUI applications
- WSL2 integration for Windows development
- 4GB shared memory for computational tasks

### Important Notes

- FEniCS must be installed manually inside the container (see Dockerfile:15-16 comment about MPI setup issues)
- The container uses the official NVIDIA JAX image as base (nvcr.io/nvidia/jax:25.04-py3)
- All development should happen inside the container to ensure consistency
- Working directory inside container is `/workspace/` (mounted from host project root)

## Project Structure

Currently minimal structure:
- `Dockerfile`: Container definition with scientific computing stack
- `docker-compose.yml`: Development environment orchestration
- `README.md`: Basic project information
- `LICENSE`: Project license

## Development Workflow

Since this is a container-first project, all commands should be run inside the Docker container:

1. Start the container: `docker compose up -d`
2. Enter the container: `docker exec -it unitcellax /bin/bash`
3. Install FEniCS manually if needed: `apt install -y fenicsx` (inside container)
4. Work in `/workspace/` directory (mounted from host)

### Common Commands

- **Rebuild container after Dockerfile changes**: `docker compose build --no-cache`
- **Stop container**: `docker compose down`
- **View container logs**: `docker compose logs unitcellax`
- **Check GPU access inside container**: `nvidia-smi`

## Container Configuration

The Docker setup includes:
- GPU access with all NVIDIA capabilities
- Display forwarding for GUI applications
- Volume mounting for persistent development
- IPC host mode for inter-process communication
- 4GB shared memory allocation

## Future Development

The project appears to be in early stages. When adding code:
- Consider the scientific computing focus (JAX/NumPy/FEniCS)
- Maintain containerized development approach
- Test GPU functionality for computational workloads
- Consider MPI requirements for parallel computing