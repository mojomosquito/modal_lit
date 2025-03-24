"""
Modal-Compatible Inference Orchestration
========================================

This module provides a Modal-compatible wrapper for the InferenceOrchestrator,
ensuring that inference can run smoothly in Modal's serverless environment.

Key features:
1. Modal volume integration for persistent storage
2. Environment variable management for API keys
3. Modal-specific path handling
4. Optimized request limits configuration
5. Simplified interface for remote execution

The primary entry point is the `run_inference` function, which is decorated as
a Modal function and handles all the necessary setup and configuration.

Version: 1.0.0
"""

import os
import modal
import asyncio
from typing import List, Dict, Any, Optional, Union

# Local imports
from minference.threads.inference import InferenceOrchestrator, RequestLimits
from minference.threads.models import ChatThread, ProcessedOutput
from minference.ecs.entity import EntityRegistry
from minference.threads.modal_utils import app, get_modal_cache_dir, cache_vol

###########################################
# MODAL CONFIGURATION
###########################################

# Create Modal volume for persistent storage
# This volume must be created before running the code with:
# modal volume create inference-cache-vol
cache_vol = modal.Volume.from_name("inference-cache-vol")

def get_modal_cache_dir() -> str:
    """
    Get the Modal-compatible cache directory.
    
    This directory is mounted to the Modal volume and persists
    across function invocations.
    
    Returns:
        str: Absolute path to the cache directory in Modal
    """
    return "/cache"

###########################################
# MODAL INFERENCE ORCHESTRATOR
###########################################

class ModalInferenceOrchestrator(InferenceOrchestrator):
    """
    Modal-compatible version of InferenceOrchestrator.
    
    This class extends the base InferenceOrchestrator to ensure
    compatibility with Modal's serverless environment, particularly
    focusing on file storage, caching, and environment management.
    """
    
    def __init__(self, 
                 oai_request_limits: Optional[RequestLimits] = None,
                 anthropic_request_limits: Optional[RequestLimits] = None,
                 vllm_request_limits: Optional[RequestLimits] = None,
                 litellm_request_limits: Optional[RequestLimits] = None,
                 openrouter_request_limits: Optional[RequestLimits] = None):
        """
        Initialize the Modal-compatible Inference Orchestrator.
        
        Args:
            oai_request_limits: Request limits for OpenAI API
            anthropic_request_limits: Request limits for Anthropic API
            vllm_request_limits: Request limits for VLLM API
            litellm_request_limits: Request limits for LiteLLM API
            openrouter_request_limits: Request limits for OpenRouter API
        """
        # Initialize the parent InferenceOrchestrator with Modal-specific settings
        super().__init__(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=anthropic_request_limits,
            vllm_request_limits=vllm_request_limits,
            litellm_request_limits=litellm_request_limits,
            openrouter_request_limits=openrouter_request_limits,
            local_cache=True,  # Always use cache in Modal
            cache_folder=get_modal_cache_dir()
        )
        
        # Configure logging
        EntityRegistry._logger.info("Initialized ModalInferenceOrchestrator")
    
    def _setup_cache_folder(self, cache_folder: Optional[str]) -> str:
        """
        Override to use Modal volume for cache.
        
        This ensures cache files are stored in the Modal volume
        and persist across function invocations.
        
        Args:
            cache_folder: Path to the cache folder
            
        Returns:
            str: Final cache folder path
        """
        cache_dir = cache_folder or get_modal_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        EntityRegistry._logger.info(f"Using Modal cache directory: {cache_dir}")
        return cache_dir

###########################################
# MODAL FUNCTION
###########################################

@app.function(
    image=modal.Image.debian_slim().pip_install(
        # Core ML dependencies
        "openai>=1.0.0",        # OpenAI API client
        "anthropic>=0.10.0",    # Anthropic API client
        
        # Data processing and utilities
        "aiofiles>=23.1.0",     # Asynchronous file operations
        "python-dotenv>=1.0.0", # Environment variable management
        "pydantic>=2.0.0",      # Data validation
        "jsonschema>=4.17.3"    # JSON schema validation
    ),
    # NOTE: Future Modal versions will require explicitly adding local Python sources:
    # .add_local_python_source("minference")
    # .add_local_python_source("examples")
    # See deprecation warnings when running examples/lit_agents.py
    volumes={get_modal_cache_dir(): cache_vol},
    timeout=3600  # 1 hour timeout for long-running jobs
)
async def run_inference(
    chat_threads: List[ChatThread],
    env_vars: Dict[str, str],
    request_limits: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[ProcessedOutput]:
    """
    Run inference using Modal infrastructure.
    
    This function:
    1. Sets up environment variables (API keys, etc.)
    2. Configures request limits for different providers
    3. Initializes a Modal-compatible orchestrator
    4. Runs inference on all chat threads in parallel
    5. Returns the processed outputs
    
    Args:
        chat_threads: List of chat threads to process
        env_vars: Dictionary of environment variables (API keys, etc.)
        request_limits: Optional dictionary of request limits for different providers
                        Format: {"provider_name": {"max_requests_per_minute": int, 
                                                  "max_tokens_per_minute": int}}
    
    Returns:
        List of processed outputs from the model
    """
    try:
        # Set environment variables
        for key, value in env_vars.items():
            if value is not None:  # Only set if value is provided
                os.environ[key] = value
                # Mask API keys in logs
                if "KEY" in key or "API" in key:
                    EntityRegistry._logger.info(f"Set environment variable: {key}=****")
                else:
                    EntityRegistry._logger.info(f"Set environment variable: {key}={value}")
        
        # Create request limits if provided
        limits = {}
        if request_limits:
            for provider, config in request_limits.items():
                limits[f"{provider}_request_limits"] = RequestLimits(**config)
                EntityRegistry._logger.info(f"Configured request limits for {provider}: {config}")
        
        # Initialize orchestrator with Modal configuration
        orchestrator = ModalInferenceOrchestrator(**limits)
        EntityRegistry.set_inference_orchestrator(orchestrator)
        EntityRegistry.set_tracing_enabled(False)  # Disable tracing for better performance
        
        # Log the processing start
        EntityRegistry._logger.info(f"Starting inference for {len(chat_threads)} chat threads")
        
        # Run inference
        results = await orchestrator.run_parallel_ai_completion(chat_threads)
        
        # Log completion
        EntityRegistry._logger.info(f"Completed inference with {len(results)} results")
        
        return results
    
    except Exception as e:
        # Log any errors at the top level
        EntityRegistry._logger.error(f"Error during inference: {str(e)}")
        EntityRegistry._logger.error(f"Traceback: {e.__traceback__}")
        raise 