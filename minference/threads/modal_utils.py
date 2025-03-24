"""
Modal-Specific Utilities and Configurations
==========================================

This module provides utilities and configurations for running the threads package
in Modal's serverless environment. It includes:

1. Volume definitions for persistent storage
2. Path utilities for Modal-compatible file handling
3. Environment setup helpers
4. Preconfigured Modal image with dependencies
5. Helper functions for Modal execution

The utilities in this module ensure that file operations, environment variables,
and other infrastructure elements work correctly in Modal's environment.

Version: 1.0.0
"""

import os
import modal
import logging
from typing import Dict, Optional, Any, List, Union
from pathlib import Path

###########################################
# MODAL VOLUMES & PATHS
###########################################

# Create Modal volumes for persistent storage
# These volumes must be created before running the code with:
# modal volume create inference-cache-vol
# modal volume create results-vol
cache_vol = modal.Volume.from_name("inference-cache-vol")
results_vol = modal.Volume.from_name("results-vol")

def get_modal_cache_dir() -> str:
    """
    Get the Modal-compatible cache directory.
    
    This directory is mounted to the cache volume and persists
    across function invocations.
    
    Returns:
        str: Absolute path to the cache directory in Modal
    """
    return "/cache"

def get_modal_results_dir() -> str:
    """
    Get the Modal-compatible results directory.
    
    This directory is mounted to the results volume and persists
    across function invocations.
    
    Returns:
        str: Absolute path to the results directory in Modal
    """
    return "/results"

def get_modal_file_path(base_dir: str, filename: str) -> str:
    """
    Get Modal-compatible file path by joining base directory and filename.
    
    Ensures paths are properly formatted for Modal's environment.
    
    Args:
        base_dir: Base directory path
        filename: Name of the file
        
    Returns:
        str: Complete file path
    """
    return os.path.join(base_dir, filename)

###########################################
# ENVIRONMENT SETUP
###########################################

def setup_modal_environment(env_vars: Dict[str, str]) -> None:
    """
    Set up environment variables in Modal context.
    
    This function takes a dictionary of environment variables and sets them
    in the Modal execution environment.
    
    Args:
        env_vars: Dictionary of environment variables (key-value pairs)
    """
    # Get a logger for this module
    logger = logging.getLogger("modal_utils")
    
    # Set environment variables
    for key, value in env_vars.items():
        if value is not None:  # Only set if value is provided
            os.environ[key] = value
            # Mask API keys in logs
            if "KEY" in key or "API" in key:
                logger.info(f"Set environment variable: {key}=****")
            else:
                logger.info(f"Set environment variable: {key}={value}")

def ensure_modal_dirs() -> None:
    """
    Ensure Modal directories exist.
    
    Creates cache and results directories if they don't exist.
    This is important for Modal's environment where directories
    might not be automatically created.
    """
    # Create cache directory
    cache_dir = get_modal_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create results directory
    results_dir = get_modal_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    
    # Log directory creation
    logger = logging.getLogger("modal_utils")
    logger.info(f"Ensured Modal directories exist: {cache_dir}, {results_dir}")

###########################################
# MODAL IMAGE CONFIGURATION
###########################################

# Define Modal image with all required dependencies
modal_image = (modal.Image.debian_slim()
    .pip_install(
        # Core ML dependencies
        "openai>=1.0.0",        # OpenAI API client
        "anthropic>=0.10.0",    # Anthropic API client
        
        # Data processing and utilities
        "python-dotenv>=1.0.0", # Environment variable management
        "polars>=0.20.0",       # Fast DataFrame library (used instead of pandas)
        "pydantic>=2.0.0",      # Data validation and settings management
        "aiofiles>=23.1.0",     # Asynchronous file operations
        "jsonschema>=4.17.3",   # JSON Schema validation
        
        # Code analysis and tokenization
        "libcst>=1.0.0",        # Concrete Syntax Tree for Python
        "tiktoken>=0.5.0",      # Fast BPE tokenizer from OpenAI
        
        # Logging and monitoring
        "rich>=13.4.2"          # Rich text and formatting in the terminal
    )
    # NOTE: Future Modal versions will require explicitly adding local Python sources:
    # .add_local_python_source("minference")
    # .add_local_python_source("examples")
    # See deprecation warnings when running examples/lit_agents.py
)

###########################################
# MODAL EXECUTION
###########################################

# Modal stub for common configurations
app = modal.App("threads-inference")

@app.function(
    image=modal_image,
    volumes={
        get_modal_cache_dir(): cache_vol,
        get_modal_results_dir(): results_vol
    },
    timeout=3600,  # 1 hour timeout for long-running jobs
    memory=4096    # 4GB of memory for larger models
)
async def run_modal_inference(
    chat_threads: List[Any],
    env_vars: Dict[str, str],
    request_limits: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[Any]:
    """
    Run inference in Modal environment.
    
    This function:
    1. Sets up the Modal environment
    2. Ensures necessary directories exist
    3. Delegates to the main inference function
    4. Returns processed results
    
    Args:
        chat_threads: List of chat threads to process
        env_vars: Environment variables to set (API keys, etc.)
        request_limits: Optional request limits configuration
        
    Returns:
        List of processed outputs from the model
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("modal_utils")
    logger.info(f"Starting Modal inference with {len(chat_threads)} threads")
    
    # Import here to avoid circular imports
    from minference.threads.modal_inference import run_inference
    
    # Setup environment and directories
    setup_modal_environment(env_vars)
    ensure_modal_dirs()
    
    # Run inference and return results
    try:
        results = await run_inference(chat_threads, env_vars, request_limits)
        logger.info(f"Completed Modal inference with {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error in Modal inference: {str(e)}")
        raise 