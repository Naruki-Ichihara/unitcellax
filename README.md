# unitcellax

Containerized development environment for computational physics/scientific computing with JAX, NumPy, and FEniCS.

## Quick Start

### Prerequisites
- Docker with NVIDIA GPU support
- docker-compose

### Setup and Run

1. **Build and start the container:**
   ```bash
   docker compose build --no-cache
   docker compose up
   ```

2. **Enter the container:**
   ```bash
   docker exec -it unitcellax /bin/bash
   ```

3. **Install FEniCS (inside container):**
   ```bash
   apt install -y fenicsx
   ```

### Configuration Options

**Claude Code Installation:**
- Edit `docker-compose.yml` and set `INSTALL_CLAUDE: "true"` to include Claude Code