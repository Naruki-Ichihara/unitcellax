"""
Global warnings configuration for unitcellax.

This module configures warning filters to suppress known harmless warnings
from dependencies, particularly SWIG-generated bindings.
"""
import warnings


def configure_warnings():
    """Configure global warning filters for the unitcellax package."""
    
    # SWIG-related deprecation warnings (from nlopt, etc.)
    warnings.filterwarnings(
        "ignore", 
        message="builtin type SwigPyPacked has no __module__ attribute",
        category=DeprecationWarning
    )
    warnings.filterwarnings(
        "ignore", 
        message="builtin type SwigPyObject has no __module__ attribute",
        category=DeprecationWarning
    )
    warnings.filterwarnings(
        "ignore", 
        message="builtin type swigvarlink has no __module__ attribute",
        category=DeprecationWarning
    )
    
    # NumPy and scientific computing warnings
    warnings.filterwarnings(
        "ignore", 
        message=".*numpy.dtype size changed.*",
        category=RuntimeWarning
    )
    
    # JAX and TensorFlow warnings (common in GPU environments)
    warnings.filterwarnings(
        "ignore",
        message=".*jax.*",
        category=UserWarning
    )
    

# Auto-configure warnings when this module is imported
configure_warnings()