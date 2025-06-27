# GitHub Actions Workflows

This directory contains automated workflows for the unitcellax project. All workflows are designed to work with the containerized development environment as specified in the project's CLAUDE.md.

## Workflows Overview

### 1. `test.yml` - Main Test Suite
**Triggers:** Push to main/develop, Pull Requests to main/develop

Comprehensive testing workflow that:
- Builds the Docker development environment
- Installs FEniCS and all dependencies
- Runs environment validation tests
- Executes the full test suite including PBC tests
- Includes separate lint and security jobs
- Provides detailed test reports

**Jobs:**
- `test`: Core functionality testing in containerized environment
- `lint`: Code formatting and style checks (black, isort, flake8)
- `security`: Security scanning (bandit, safety)

### 2. `pr-test.yml` - Fast PR Validation
**Triggers:** Pull Request events (opened, synchronize, reopened)

Lightweight testing for quick PR feedback:
- Quick container build and startup
- Essential dependency installation
- PBC module testing only
- Smoke tests for basic functionality
- Code quality checks

**Purpose:** Provide fast feedback on pull requests without running the full test suite.

### 3. `matrix-test.yml` - Comprehensive Matrix Testing
**Triggers:** Weekly schedule (Sundays 2 AM UTC), Manual dispatch, Main branch changes to build files

Advanced testing across different configurations:
- Multiple Python versions (3.11, 3.12)
- Different test types (unit, integration, pbc)
- Performance benchmarking
- Coverage reporting
- Compatibility testing

**Jobs:**
- `matrix-test`: Tests across Python versions and test types
- `compatibility-test`: Basic import testing without full container
- `performance-test`: Benchmarks for PBC operations

### 4. `docs.yml` - Documentation Validation
**Triggers:** Push/PR to main affecting Python files or docs

Documentation quality assurance:
- Docstring style validation (Google style)
- Docstring coverage analysis
- Example code validation
- API documentation generation

**Jobs:**
- `docstring-check`: Style and coverage validation
- `validate-examples`: Test docstring examples
- `api-docs`: Generate Sphinx documentation

## Container-First Approach

All main testing workflows use the project's Docker environment to ensure:
- Consistent testing environment
- Proper FEniCS installation
- GPU compatibility (where available)
- Exact dependency versions (NumPy 1.26.0 for FEniCS compatibility)

## Workflow Features

### Environment Validation
- Tests core dependencies (JAX, NumPy, SciPy)
- Validates FEniCS installation
- Checks GPU availability
- Verifies package imports

### Test Organization
- **Environment tests**: Basic system validation
- **Unit tests**: Individual component testing
- **Integration tests**: Full workflow testing
- **PBC tests**: Specific periodic boundary condition testing

### Quality Assurance
- **Code formatting**: Black for consistent formatting
- **Import sorting**: isort for organized imports
- **Style checking**: flake8 for PEP 8 compliance
- **Security scanning**: bandit and safety for security issues
- **Documentation**: Docstring style and coverage validation

### Performance Monitoring
- Benchmarks for PBC generation
- Prolongation matrix construction timing
- Memory usage profiling (in matrix tests)

## Usage Guidelines

### For Contributors
1. **PRs**: The `pr-test.yml` workflow provides quick feedback
2. **Main branch**: Full `test.yml` runs on pushes
3. **Weekly**: `matrix-test.yml` provides comprehensive validation
4. **Documentation**: `docs.yml` validates docstring quality

### For Maintainers
- Review security reports from the security job
- Monitor performance benchmarks from matrix tests
- Use workflow dispatch for manual matrix testing
- Check coverage reports for test completeness

## Error Handling

Workflows are designed to be robust:
- GPU unavailability is handled gracefully
- FEniCS installation failures are captured
- Non-critical issues (formatting, docs) are marked as warnings
- Container cleanup always runs

## Artifacts

Workflows generate useful artifacts:
- Test coverage reports (XML and HTML)
- Security scan results (JSON)
- Performance benchmark results
- API documentation (HTML)

These artifacts are uploaded and available for download from the Actions tab.