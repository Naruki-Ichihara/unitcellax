"""
Environment validation tests for unitcellax container setup.

These tests verify that the containerized development environment is properly
configured with all required dependencies and GPU access.
"""
import os
import sys
import subprocess
import platform
import warnings
from typing import Optional, Tuple
import pytest

# Filter out SWIG-related warnings at module level
warnings.filterwarnings("ignore", message="builtin type SwigPyPacked has no __module__ attribute")
warnings.filterwarnings("ignore", message="builtin type SwigPyObject has no __module__ attribute") 
warnings.filterwarnings("ignore", message="builtin type swigvarlink has no __module__ attribute")

# Mark all tests in this module as environment validation
pytestmark = pytest.mark.env_validation


class TestEnvironmentSetup:
    """Test suite for validating the containerized development environment."""

    def test_python_version(self) -> None:
        """Verify Python version meets requirements."""
        version = sys.version_info
        assert version.major == 3, f"Expected Python 3.x, got {version.major}"
        assert version.minor >= 11, f"Expected Python 3.11+, got 3.{version.minor}"

    def test_container_environment(self) -> None:
        """Verify we're running inside the expected container."""
        # Check if we're in the expected working directory
        cwd = os.getcwd()
        assert "/workspace" in cwd, f"Expected to be in /workspace, got {cwd}"
        
        # Check container-specific environment variables
        expected_vars = ["NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES"]
        for var in expected_vars:
            assert os.getenv(var) is not None, f"Missing container env var: {var}"

    def test_display_environment(self) -> None:
        """Verify display environment for GUI applications."""
        display_vars = ["DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR"]
        # At least one display variable should be set
        has_display = any(os.getenv(var) for var in display_vars)
        assert has_display, "No display environment variables found"


class TestCoreDependencies:
    """Test core Python dependencies are available and functional."""

    def test_numpy_available(self) -> None:
        """Test NumPy is available with correct version."""
        try:
            import numpy as np
            # Check version matches FEniCS requirement (pinned in setup.cfg)
            assert np.__version__ == "1.26.0", f"Expected NumPy 1.26.0 (required for FEniCS compatibility), got {np.__version__}"
            
            # Basic functionality test
            arr = np.array([1, 2, 3])
            assert arr.sum() == 6, "NumPy basic operations failed"
        except ImportError as e:
            pytest.fail(f"NumPy not available: {e}")

    def test_jax_available(self) -> None:
        """Test JAX is available and functional."""
        try:
            import jax
            import jax.numpy as jnp
            
            # Basic JAX operation
            x = jnp.array([1.0, 2.0, 3.0])
            result = jnp.sum(x)
            assert float(result) == 6.0, "JAX basic operations failed"
        except ImportError as e:
            pytest.fail(f"JAX not available: {e}")

    def test_scipy_available(self) -> None:
        """Test SciPy is available."""
        try:
            import scipy
            import scipy.optimize
            
            # Basic functionality test
            result = scipy.optimize.minimize_scalar(lambda x: x**2)
            assert abs(result.x) < 1e-6, "SciPy optimization failed"
        except ImportError as e:
            pytest.fail(f"SciPy not available: {e}")

    def test_matplotlib_available(self) -> None:
        """Test Matplotlib is available."""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            
            # Test backend is set (important for headless container)
            backend = matplotlib.get_backend()
            assert backend is not None, "Matplotlib backend not configured"
        except ImportError as e:
            pytest.fail(f"Matplotlib not available: {e}")

    def test_meshio_available(self) -> None:
        """Test meshio is available."""
        try:
            import meshio
            # Basic functionality check
            assert hasattr(meshio, 'read'), "meshio.read not available"
            assert hasattr(meshio, 'write'), "meshio.write not available"
        except ImportError as e:
            pytest.fail(f"meshio not available: {e}")

    def test_petsc4py_available(self) -> None:
        """Test petsc4py is available."""
        try:
            import petsc4py
            from petsc4py import PETSc
            
            # Basic PETSc functionality
            vec = PETSc.Vec().create()
            vec.setSizes(10)
            vec.setUp()
            assert vec.getSize() == 10, "PETSc vector creation failed"
        except ImportError as e:
            pytest.fail(f"petsc4py not available: {e}")

    def test_gmsh_available(self) -> None:
        """Test gmsh is available."""
        try:
            import gmsh
            # Test basic gmsh functionality
            gmsh.initialize()
            gmsh.finalize()
        except ImportError as e:
            pytest.fail(f"gmsh not available: {e}")

    def test_nlopt_available(self) -> None:
        """Test nlopt is available."""
        try:
            import nlopt
            # Basic functionality test
            opt = nlopt.opt(nlopt.LN_COBYLA, 2)
            assert opt.get_dimension() == 2, "nlopt optimization setup failed"
        except ImportError as e:
            pytest.fail(f"nlopt not available: {e}")


@pytest.mark.gpu
class TestGPUEnvironment:
    """Test GPU availability and JAX GPU support."""

    def _check_nvidia_smi(self) -> Tuple[bool, Optional[str]]:
        """Check if nvidia-smi is available and working."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, result.stderr
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False, "nvidia-smi not found or failed"

    def test_nvidia_driver_available(self) -> None:
        """Test NVIDIA driver is available via JAX device detection."""
        try:
            import jax
            
            devices = jax.devices()
            assert len(devices) > 0, "No JAX devices found"
            
            # Check for GPU devices (more reliable than nvidia-smi in container)
            # JAX may return different device_kind formats: 'gpu', 'NVIDIA RTX...', etc.
            gpu_devices = [d for d in devices if 'gpu' in d.device_kind.lower() or 'nvidia' in d.device_kind.lower() or 'cuda' in str(d).lower()]
            if not gpu_devices:
                # Fall back to nvidia-smi check
                available, info = self._check_nvidia_smi()
                if not available:
                    pytest.skip(f"No GPU devices found via JAX or nvidia-smi: {info}")
                else:
                    pytest.skip("nvidia-smi available but JAX not detecting GPU devices")
            
            # If we have GPU devices, driver is working
            assert len(gpu_devices) > 0, f"Expected GPU devices, found: {[str(d) for d in devices]}"
                
        except ImportError:
            pytest.skip("JAX not available for GPU driver testing")

    def test_jax_gpu_support(self) -> None:
        """Test JAX can detect and use GPU devices."""
        try:
            import jax
            
            devices = jax.devices()
            assert len(devices) > 0, "No JAX devices found"
            
            # Check for GPU devices
            # JAX may return different device_kind formats: 'gpu', 'NVIDIA RTX...', etc.
            gpu_devices = [d for d in devices if 'gpu' in d.device_kind.lower() or 'nvidia' in d.device_kind.lower() or 'cuda' in str(d).lower()]
            if not gpu_devices:
                pytest.skip("No GPU devices found in JAX")
            
            # Test basic GPU computation
            import jax.numpy as jnp
            x = jnp.array([1.0, 2.0, 3.0])
            with jax.default_device(gpu_devices[0]):
                result = jnp.sum(x ** 2)
                assert float(result) == 14.0, "GPU computation failed"
            
            # Additional check: verify device placement
            with jax.default_device(gpu_devices[0]):
                test_array = jnp.ones(3)
                # Use device() method to get the device (not callable)
                actual_device = test_array.devices().pop()
                assert actual_device == gpu_devices[0], f"Array not placed on expected GPU device. Expected: {gpu_devices[0]}, Got: {actual_device}"
                
        except ImportError:
            pytest.skip("JAX not available for GPU testing")


class TestFEniCSEnvironment:
    """Test FEniCS installation and basic functionality."""

    def test_fenicsx_available(self) -> None:
        """Test FEniCS-X is available (may need manual installation)."""
        try:
            import dolfinx
            import ufl
            
            # Basic functionality test
            assert hasattr(dolfinx, 'mesh'), "dolfinx.mesh not available"
            assert hasattr(ufl, 'TrialFunction'), "ufl.TrialFunction not available"
            
        except ImportError as e:
            pytest.skip(f"FEniCSx not installed: {e}. Install with 'apt install -y fenicsx'")

    def test_fenicsx_basic_functionality(self) -> None:
        """Test basic FEniCS-X mesh creation."""
        try:
            import dolfinx
            import numpy as np
            from mpi4py import MPI
            
            # Create a simple unit square mesh
            mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
            assert mesh.topology.index_map(0).size_global > 0, "Mesh creation failed"
            
        except ImportError:
            pytest.skip("FEniCSx not available for functionality testing")


class TestPackageStructure:
    """Test the unitcellax package structure and installation."""

    def test_unitcellax_importable(self) -> None:
        """Test unitcellax package can be imported."""
        try:
            import unitcellax
            assert hasattr(unitcellax, '__version__'), "Package version not defined"
        except ImportError as e:
            pytest.fail(f"unitcellax package not importable: {e}")

    def test_package_version(self) -> None:
        """Test package version is defined and valid."""
        import unitcellax
        version = unitcellax.__version__
        assert isinstance(version, str), f"Version should be string, got {type(version)}"
        assert len(version) > 0, "Version string is empty"


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])