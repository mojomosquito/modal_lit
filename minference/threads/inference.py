"""
Refactored InferenceOrchestrator using utility functions for payload handling.
All imports properly managed and type safety enforced.
"""
import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Literal
from pydantic import Field
from dotenv import load_dotenv
import time
from uuid import UUID, uuid4
import aiofiles
import logging
import importlib

# Internal imports - complete set
from minference.threads.models import (
    RawOutput, ProcessedOutput, ChatThread, LLMClient,
    ResponseFormat, Entity, ChatMessage,
    MessageRole
)
from minference.threads.oai_parallel import (
    process_api_requests_from_file,
    OAIApiFromFileConfig
)
from minference.threads.requests import (
    prepare_requests_file,
    convert_chat_thread_to_request,
    create_oai_completion_config,
    create_anthropic_completion_config,
    create_vllm_completion_config,
    create_litellm_completion_config,
    create_openrouter_completion_config
)
from minference.ecs.entity import EntityRegistry, entity_tracer
from minference.threads.modal_utils import get_modal_cache_dir, get_modal_results_dir, get_modal_file_path
from minference.threads.utils import msg_dict_to_oai, msg_dict_to_anthropic, parse_json_string

# Import Groq for API access
try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

class RequestLimits(Entity):
    """
    Configuration for API request limits.
    Inherits from Entity for UUID handling and registry integration.
    """
    max_requests_per_minute: int = Field(
        default=50,
        description="The maximum number of requests per minute for the API"
    )
    max_tokens_per_minute: int = Field(
        default=100000,
        description="The maximum number of tokens per minute for the API"
    )
    provider: Literal["openai", "anthropic", "vllm", "litellm", "openrouter"] = Field(
        default="openai",
        description="The provider of the API"
    )

def create_chat_thread_hashmap(chat_threads: List[ChatThread]) -> Dict[UUID, ChatThread]:
    """Create a hashmap of chat threads by their IDs."""
    return {p.id: p for p in chat_threads if p.id is not None}

async def process_outputs_and_execute_tools(chat_threads: List[ChatThread], llm_outputs: List[ProcessedOutput]) -> List[ProcessedOutput]:
    """Process outputs and execute tools in parallel."""
    # Track thread ID mappings (original -> latest)
    thread_id_mappings = {chat_thread.live_id: chat_thread for chat_thread in chat_threads}
    history_update_tasks = []
    
    EntityRegistry._logger.info("""
=================================================================
                    START PROCESSING OUTPUTS 
=================================================================
""")
    
    for output in llm_outputs:
        if output.chat_thread_live_id:
            
            try:
                # Get the latest version from the thread mapping
                chat_thread = thread_id_mappings.get(output.chat_thread_live_id,None)
                if not chat_thread:
                    EntityRegistry._logger.error(f"ChatThread({output.chat_thread_id}) not found in temporary thread_id_mappings")
                    continue
                    
                EntityRegistry._logger.info(f"""
=== PROCESSING OUTPUT FOR CHAT THREAD ===
Current Thread ID: {chat_thread.id}
Current History Length: {len(chat_thread.history)}
Current New Message: {chat_thread.new_message}
Current Parent ID: {chat_thread.parent_id}
Current Lineage ID: {chat_thread.lineage_id}
=======================================
""")
                
                # Add history and get potentially new version
                history_update_tasks.append(
                    chat_thread.add_chat_turn_history(output)
                )
                EntityRegistry._logger.info(
                    f"Queued history update for ChatThread({chat_thread.id})"
                )
            except Exception as e:
                EntityRegistry._logger.error(f"""
=== ERROR PROCESSING CHAT THREAD ===
Thread ID: {output.chat_thread_id}
Error: {str(e)}
Traceback: {e.__traceback__}
===================================
""")
    
    if history_update_tasks:
        EntityRegistry._logger.info(f"""
=================================================================
            EXECUTING {len(history_update_tasks)} HISTORY UPDATES 
=================================================================
""")
        try:
            results = await asyncio.gather(*history_update_tasks)
            
            
        except Exception as e:
            EntityRegistry._logger.error(f"""
=== ERROR DURING HISTORY UPDATES ===
Error: {str(e)}
Traceback: {e.__traceback__}
==================================
""")
    
    return llm_outputs

async def run_parallel_ai_completion(
    chat_threads: List[ChatThread],
    orchestrator: 'InferenceOrchestrator'
) -> List[ProcessedOutput]:
    """Run parallel AI completion for multiple chat threads."""
    EntityRegistry._logger.info(f"Starting parallel AI completion for {len(chat_threads)} chat threads")
    
    # Track original to forked thread mappings
    original_ids = [chat.id for chat in chat_threads]
    thread_mappings = {}
    
    # First add user messages to all chat threads
    for chat in chat_threads:
        try:
            EntityRegistry._logger.info(f"Adding user message to ChatThread({chat.id})")
            original_id = chat.id
            result = chat.add_user_message()
           
        except Exception as e:
            if chat.llm_config.response_format != ResponseFormat.auto_tools or chat.llm_config.response_format != ResponseFormat.workflow:
                chat_threads.remove(chat)
                EntityRegistry._logger.error(f"Error adding user message to ChatThread({chat.id}): {e}")

    # Run LLM completions in parallel
    tasks = []
    if any(p for p in chat_threads if p.llm_config.client == "openai"):
        tasks.append(orchestrator._run_openai_completion([p for p in chat_threads if p.llm_config.client == "openai"]))
    if any(p for p in chat_threads if p.llm_config.client == "anthropic"):
        tasks.append(orchestrator._run_anthropic_completion([p for p in chat_threads if p.llm_config.client == "anthropic"]))
    if any(p for p in chat_threads if p.llm_config.client == "vllm"):
        tasks.append(orchestrator._run_vllm_completion([p for p in chat_threads if p.llm_config.client == "vllm"]))
    if any(p for p in chat_threads if p.llm_config.client == "litellm"):
        tasks.append(orchestrator._run_litellm_completion([p for p in chat_threads if p.llm_config.client == "litellm"]))
    if any(p for p in chat_threads if p.llm_config.client == "openrouter"):
        tasks.append(orchestrator._run_openrouter_completion([p for p in chat_threads if p.llm_config.client == "openrouter"]))

    results = await asyncio.gather(*tasks)
    llm_outputs = [item for sublist in results for item in sublist]
    
    

    
    EntityRegistry._logger.info(f"Processing {len(llm_outputs)} LLM outputs")
    processed_outputs = await process_outputs_and_execute_tools(chat_threads, llm_outputs)


    
    return processed_outputs

async def parse_results_file(filepath: str, client: LLMClient) -> List[ProcessedOutput]:
    """Parse results file and convert to ProcessedOutput objects asynchronously."""
    results = []
    pending_tasks = []
    EntityRegistry._logger.info(f"Parsing results from {filepath}")
    
    async def process_line(line: str) -> Optional[ProcessedOutput]:
        """Process a single line asynchronously."""
        try:
            result = json.loads(line)
            processed_output = await convert_result_to_llm_output(result, client)
            return processed_output
        except json.JSONDecodeError:
            EntityRegistry._logger.error(f"Error decoding JSON: {line}")
        except Exception as e:
            EntityRegistry._logger.error(f"Error processing result: {e}")
        return None

    async with aiofiles.open(filepath, 'r') as f:
        async for line in f:
            # Create task for each line
            task = asyncio.create_task(process_line(line))
            pending_tasks.append(task)
    
    # Wait for all tasks to complete
    completed_tasks = await asyncio.gather(*pending_tasks)
    
    # Filter out None results and add to results list
    results = [result for result in completed_tasks if result is not None]
    
    EntityRegistry._logger.info(f"Processed {len(results)} results from {filepath}")
    return results

async def convert_result_to_llm_output(result: List[Dict[str, Any]], client: LLMClient) -> ProcessedOutput:
    """Convert raw result directly to ProcessedOutput."""
    metadata, request_data, response_data = result
    # EntityRegistry._logger.info(f"Converting result for chat_thread_id: {metadata['chat_thread_id']}")

    raw_output = RawOutput(
        raw_result=response_data,
        completion_kwargs=request_data,
        start_time=metadata["start_time"],
        end_time=metadata["end_time"] or time.time(),
        chat_thread_id=metadata["chat_thread_id"],
        chat_thread_live_id=metadata["chat_thread_live_id"],
        client=client
    )

    return raw_output.create_processed_output()


class InferenceOrchestrator:
    def __init__(self, 
                 oai_request_limits: Optional[RequestLimits] = None, 
                 anthropic_request_limits: Optional[RequestLimits] = None, 
                 vllm_request_limits: Optional[RequestLimits] = None,
                 litellm_request_limits: Optional[RequestLimits] = None,
                 openrouter_request_limits: Optional[RequestLimits] = None,
                 local_cache: bool = True,
                 cache_folder: Optional[str] = None):
        load_dotenv()
        EntityRegistry._logger.info("Initializing InferenceOrchestrator")
        
        # Use Modal cache directory by default
        self.cache_folder = self._setup_cache_folder(cache_folder or get_modal_cache_dir())
        
        # API Keys and Endpoints
        self.openai_key = os.getenv("OPENAI_KEY", "")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.vllm_key = os.getenv("VLLM_API_KEY", "")
        self.default_vllm_endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
        self.default_litellm_endpoint = os.getenv("LITELLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
        self.litellm_key = os.getenv("LITELLM_API_KEY", "")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        self.default_openrouter_endpoint = os.getenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")
        
        # Request Limits
        self.oai_request_limits = oai_request_limits or RequestLimits(provider="openai")
        self.anthropic_request_limits = anthropic_request_limits or RequestLimits(provider="anthropic")
        self.vllm_request_limits = vllm_request_limits or RequestLimits(provider="vllm")
        self.litellm_request_limits = litellm_request_limits or RequestLimits(provider="litellm")
        self.openrouter_request_limits = openrouter_request_limits or RequestLimits(provider="openrouter")
        
        # Cache settings
        self.local_cache = local_cache
        
    def _setup_cache_folder(self, cache_folder: Optional[str]) -> str:
        """Set up cache folder in Modal environment."""
        cache_dir = cache_folder or get_modal_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _create_chat_thread_hashmap(self, chat_threads: List[ChatThread]) -> Dict[UUID, ChatThread]:
        """Create a hashmap of chat threads by their IDs."""
        return create_chat_thread_hashmap(chat_threads)

    async def run_parallel_ai_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        """Run parallel AI completion for multiple chat threads."""
        return await run_parallel_ai_completion(chat_threads, self)

    async def _process_outputs_and_execute_tools(self, chat_threads: List[ChatThread], llm_outputs: List[ProcessedOutput]) -> List[ProcessedOutput]:
        """Process outputs and execute tools in parallel."""
        return await process_outputs_and_execute_tools(chat_threads, llm_outputs)

    async def _run_openai_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        """Run OpenAI completion with Modal-compatible file handling."""
        if not chat_threads:
            return []
            
        # Use Modal paths for request and results files
        requests_file = get_modal_file_path(get_modal_cache_dir(), f"oai_requests_{uuid4()}.jsonl")
        results_file = get_modal_file_path(get_modal_results_dir(), f"oai_results_{uuid4()}.jsonl")
        
        # Prepare requests
        await prepare_requests_file(
            chat_threads=chat_threads,
            output_file=requests_file,
            completion_config=create_oai_completion_config()
        )
        
        # Process requests
        await process_api_requests_from_file(
            OAIApiFromFileConfig(
                input_file=requests_file,
                output_file=results_file,
                request_limits=self.oai_request_limits
            )
        )
        
        # Parse results
        results = await parse_results_file(results_file, LLMClient.openai)
        
        # Clean up files
        self._delete_files(requests_file, results_file)
        
        return results

    async def _run_anthropic_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        unique_uuid = str(uuid4())
        requests_file = os.path.join(self.cache_folder, f'anthropic_requests_{unique_uuid}_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'anthropic_results_{unique_uuid}_{timestamp}.jsonl')
        

        prepare_requests_file(chat_threads, "anthropic", requests_file)
        config = create_anthropic_completion_config(
            chat_thread=chat_threads[0], 
            requests_file=requests_file, 
            results_file=results_file,
            anthropic_key=self.anthropic_key,
            max_requests_per_minute=self.anthropic_request_limits.max_requests_per_minute,
            max_tokens_per_minute=self.anthropic_request_limits.max_tokens_per_minute
        )
        
        if config:
            try:
                await process_api_requests_from_file(config)
                return await parse_results_file(results_file, client=LLMClient.anthropic)
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    async def _run_vllm_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        unique_uuid = str(uuid4())
        requests_file = os.path.join(self.cache_folder, f'vllm_requests_{unique_uuid}_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'vllm_results_{unique_uuid}_{timestamp}.jsonl')
        

        prepare_requests_file(chat_threads, "vllm", requests_file)
        config = create_vllm_completion_config(
            chat_thread=chat_threads[0], 
            requests_file=requests_file, 
            results_file=results_file,
            vllm_endpoint=self.default_vllm_endpoint,
            vllm_key=self.vllm_key,
            max_requests_per_minute=self.vllm_request_limits.max_requests_per_minute,
            max_tokens_per_minute=self.vllm_request_limits.max_tokens_per_minute
        )
        
        if config:
            try:
                await process_api_requests_from_file(config)
                return await parse_results_file(results_file, client=LLMClient.vllm)
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    async def _run_litellm_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        unique_uuid = str(uuid4())
        requests_file = os.path.join(self.cache_folder, f'litellm_requests_{unique_uuid}_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'litellm_results_{unique_uuid}_{timestamp}.jsonl')
        

        prepare_requests_file(chat_threads, "litellm", requests_file)
        config = create_litellm_completion_config(
            chat_thread=chat_threads[0], 
            requests_file=requests_file, 
            results_file=results_file,
            litellm_endpoint=self.default_litellm_endpoint,
            litellm_key=self.litellm_key,
            max_requests_per_minute=self.litellm_request_limits.max_requests_per_minute,
            max_tokens_per_minute=self.litellm_request_limits.max_tokens_per_minute
        )
        
        if config:
            try:
                await process_api_requests_from_file(config)
                return await parse_results_file(results_file, client=LLMClient.litellm)
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    async def _run_openrouter_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        unique_uuid = str(uuid4())
        requests_file = os.path.join(self.cache_folder, f'openrouter_requests_{unique_uuid}_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'openrouter_results_{unique_uuid}_{timestamp}.jsonl')
        
        prepare_requests_file(chat_threads, "openrouter", requests_file)
        config = create_openrouter_completion_config(
            chat_thread=chat_threads[0], 
            requests_file=requests_file, 
            results_file=results_file,
            openrouter_endpoint=self.default_openrouter_endpoint,
            openrouter_key=self.openrouter_key,
            max_requests_per_minute=self.openrouter_request_limits.max_requests_per_minute,
            max_tokens_per_minute=self.openrouter_request_limits.max_tokens_per_minute
        )
        
        if config:
            try:
                await process_api_requests_from_file(config)
                return await parse_results_file(results_file, client=LLMClient.openrouter)
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    def _delete_files(self, *files):
        """Delete files safely in Modal environment."""
        for file in files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                EntityRegistry._logger.error(f"Error deleting file {file}: {e}")

    async def _run_groq_completion(self, thread: ChatThread) -> ProcessedOutput:
        """
        Run a completion request through the Groq API.
        
        Args:
            thread: ChatThread to process
            
        Returns:
            ProcessedOutput containing the model's response
        """
        if not GROQ_AVAILABLE:
            raise ImportError("The groq package is not installed. Please install it with 'pip install groq'.")
            
        # Get API key from environment
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        # Initialize client
        client = groq.Groq(api_key=api_key)
        
        # Prepare messages
        messages = []
        
        # Add system prompt if present
        if thread.system_prompt:
            messages.append({
                "role": "system",
                "content": thread.system_prompt.content
            })
            
        # Add conversation history
        for msg in thread.messages:
            message = {"role": msg.role}
            
            # Handle content
            if msg.content:
                message["content"] = msg.content
                
            # Handle tool calls and results
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                message["tool_calls"] = msg.tool_calls
                
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                message["tool_call_id"] = msg.tool_call_id
                
            messages.append(message)
            
        # Set parameters
        model = thread.llm_config.model or os.environ.get("GROQ_MODEL", "llama3-8b-8192")
        
        # Track token usage
        start_time = time.time()
        
        try:
            # Make the API call
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=thread.llm_config.temperature or 0.7,
                max_tokens=thread.llm_config.max_tokens or 4096
            )
            
            # Extract generated content
            message = response.choices[0].message
            content = message.content or ""
            
            # Create processed output
            processed_output = ProcessedOutput(
                thread_id=thread.thread_id,
                content=content,
                finish_reason=response.choices[0].finish_reason,
                usage=dict(response.usage) if hasattr(response, 'usage') else {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                },
                json_object=None,  # Groq doesn't support direct JSON mode yet
                model_info={
                    "model": model,
                    "provider": "groq",
                    "temperature": thread.llm_config.temperature or 0.7,
                    "max_tokens": thread.llm_config.max_tokens or 4096,
                    "elapsed_time": time.time() - start_time
                }
            )
            
            # Parse JSON if response format is json
            if (thread.llm_config.response_format and 
                thread.llm_config.response_format.type == "json"):
                try:
                    json_object = parse_json_string(content)
                    processed_output.json_object = json_object
                except Exception as e:
                    self._logger.warning(f"Failed to parse JSON from Groq response: {e}")
                    
            return processed_output
            
        except Exception as e:
            self._logger.error(f"Error during Groq inference: {e}")
            raise

    async def run_ai_completion(self, thread: ChatThread) -> ProcessedOutput:
        """Run an AI completion request based on the thread's LLM configuration."""
        # Get the client from config
        client = thread.llm_config.client if thread.llm_config else None
        
        if client == LLMClient.openai:
            return await self._run_oai_completion(thread)
        elif client == LLMClient.anthropic:
            return await self._run_anthropic_completion(thread)
        elif client == LLMClient.vllm:
            return await self._run_vllm_completion(thread)
        elif client == LLMClient.litellm:
            return await self._run_litellm_completion(thread)
        elif client == LLMClient.openrouter:
            return await self._run_openrouter_completion(thread)
        elif client == LLMClient.groq:
            return await self._run_groq_completion(thread)
        else:
            raise ValueError(f"Unsupported LLM client: {client}")