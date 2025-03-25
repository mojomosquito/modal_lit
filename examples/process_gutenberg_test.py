"""
Process Gutenberg books using the lit_agents application
"""

import os
import json
import random
from datetime import datetime
import modal

# Initialize Modal volumes
gutenberg_data_vol = modal.Volume.from_name("gutenberg-data-vol")
results_vol = modal.Volume.from_name("gutenberg-results-vol", create_if_missing=True)

# Build the image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "openai>=1.0.0",        # OpenAI API client
        "anthropic>=0.10.0",    # Anthropic API client
        "groq>=0.4.0",          # Groq API client
        "modal>=0.56.4272",     # Modal cloud infrastructure
        "typer>=0.9.0",         # CLI framework
        "rich>=12.0.0",         # Rich text formatting
        "python-dotenv>=1.0.0", # Environment variable management
        "polars>=0.10.0",       # Fast DataFrame library
        "pydantic>=2.0.0",      # Data validation
        "instructor>=1.0.0",    # Structured outputs for LLMs
        "aiofiles>=23.1.0",     # Asynchronous file operations
        "jsonschema>=4.17.3",   # JSON Schema validation
        "libcst>=1.0.0",        # Concrete Syntax Tree for Python
        "tiktoken>=0.5.0",      # Fast BPE tokenizer from OpenAI
        "tqdm>=4.66.0",         # Progress bars
    )
    .add_local_python_source("minference")
    .add_local_python_source("examples")
)

# Create a Modal app
app = modal.App("gutenberg-processor")

@app.function(
    volumes={
        "/data": gutenberg_data_vol,
        "/results": results_vol,
    },
    image=image,
    memory=2048,
    timeout=1200  # Increased timeout to 20 minutes
)
def process_book(book_index: int = 0, provider: str = "groq", max_chunk_size: int = 4000, chunk_overlap: int = 200, concurrency: int = 5):
    """
    Process a single book from the Gutenberg dataset using the lit_agents app.
    
    Args:
        book_index: Index of the book to process from the list of available books
        provider: LLM provider to use (e.g., "groq", "anthropic", "openai")
        max_chunk_size: Maximum size of each text chunk to process in parallel
        chunk_overlap: Overlap between consecutive chunks to maintain context
        concurrency: Maximum number of chunks to process in parallel
    """
    # Convert book_index to int in case it's passed as a string
    book_index = int(book_index)
    
    print(f"Starting process_book with book_index={book_index}, provider={provider}")
    
    # Ensure results directory exists
    results_dir = "/results/gutenberg_processed"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get list of available books
    print("Getting list of available books...")
    book_files = sorted(os.listdir("/data/gutenberg"))
    print(f"Found {len(book_files)} books in the dataset")
    
    # If book_index is out of range, choose a random book
    if book_index >= len(book_files) or book_index < 0:
        book_index = random.randint(0, len(book_files) - 1)
        print(f"Selected random book index: {book_index}")
    
    selected_file = book_files[book_index]
    print(f"Processing book file: {selected_file}")
    
    # Load book data
    print(f"Loading book data from: /data/gutenberg/{selected_file}")
    with open(os.path.join("/data/gutenberg", selected_file), "r", encoding="utf-8") as f:
        book_data = json.load(f)
    
    # Extract text content - use "processed_text" field for the Gutenberg dataset
    text = book_data.get("processed_text", "")
    if not text:
        print("No text found in the book data")
        return {"error": "No text found in the book data"}
    
    print(f"Original text length: {len(text)} characters")
    
    # Extract book metadata
    metadata = {
        "title": f"Gutenberg Book {book_data.get('identifier', 'Unknown')}",
        "author": book_data.get("author", "Unknown"),
        "book_id": book_data.get("identifier", "Unknown"),
        "language": book_data.get("language", "Unknown"),
        "word_count": book_data.get("word_count", 0),
        "source_file": selected_file
    }
    
    print(f"Processing book: {metadata['title']} (ID: {metadata['book_id']})")
    
    try:
        # Set up environment for API keys if needed
        os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", "")
        os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")
        os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
        
        print(f"Using provider: {provider}")
        
        # Process the entire book in parallel chunks
        print(f"Splitting book into chunks (max size: {max_chunk_size}, overlap: {chunk_overlap})...")
        actions = process_text_in_parallel.remote(text, provider, max_chunk_size, chunk_overlap, concurrency)
        print(f"Received result with {len(actions)} actions")
        
        # Add metadata and processing info
        result_data = {
            "actions": actions,  # The actions list from the parallel processing
            "book_metadata": metadata,
            "processing_info": {
                "provider": provider,
                "timestamp": datetime.now().isoformat(),
                "max_chunk_size": max_chunk_size,
                "chunk_overlap": chunk_overlap,
                "concurrency": concurrency,
                "total_chunks": max(1, (len(text) // (max_chunk_size - chunk_overlap)) + 1)
            }
        }
        
        # Save result to the results volume
        result_filename = f"{os.path.splitext(selected_file)[0]}_processed_{provider}.json"
        result_path = os.path.join(results_dir, result_filename)
        
        print(f"Saving results to: {result_path}")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"Processing completed. Results saved to {result_path}")
        print(f"Extracted {len(actions)} actions from the text")
        
        return {
            "success": True,
            "book": metadata,
            "result_file": result_path,
            "actions_count": len(actions)
        }
        
    except Exception as e:
        import traceback
        print(f"Error processing book: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        error_result = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "book_metadata": metadata,
            "processing_info": {
                "provider": provider,
                "timestamp": datetime.now().isoformat(),
                "max_chunk_size": max_chunk_size,
                "chunk_overlap": chunk_overlap
            }
        }
        
        # Save error result
        error_filename = f"{os.path.splitext(selected_file)[0]}_error_{provider}.json"
        error_path = os.path.join(results_dir, error_filename)
        
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(error_result, f, ensure_ascii=False, indent=2)
        
        print(f"Error saved to {error_path}")
        return error_result

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("my-openai-secret")],
    memory=2048,
    timeout=600
)
def process_text_in_parallel(text: str, provider: str = "groq", max_chunk_size: int = 4000, chunk_overlap: int = 200, concurrency: int = 5):
    """
    Process a long text by splitting it into overlapping chunks and processing them in parallel.
    
    Args:
        text: The full text to process
        provider: LLM provider to use ("openai", "anthropic", or "groq")
        max_chunk_size: Maximum size of each text chunk
        chunk_overlap: Overlap between consecutive chunks
        concurrency: Maximum number of chunks to process in parallel
        
    Returns:
        list: Combined list of actions from all chunks with duplicate removal
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    from datetime import timedelta
    
    # Split the text into overlapping chunks
    chunks = []
    start_idx = 0
    
    while start_idx < len(text):
        end_idx = min(start_idx + max_chunk_size, len(text))
        
        # Try to find sentence boundaries for cleaner splits
        if end_idx < len(text) and end_idx - start_idx > 500:
            # Look for sentence ending punctuation followed by space within the last 200 characters
            for i in range(end_idx, max(end_idx - 200, start_idx), -1):
                if i < len(text) and text[i-1] in ['.', '!', '?'] and (i == len(text) or text[i].isspace()):
                    end_idx = i
                    break
        
        # If we're at the end of the text, just use what's left
        if end_idx >= len(text):
            chunks.append(text[start_idx:])
            break
            
        chunks.append(text[start_idx:end_idx])
        # Move to the next chunk with overlap
        start_idx = end_idx - chunk_overlap
    
    print(f"Split text into {len(chunks)} chunks (avg size: {sum(len(c) for c in chunks)/len(chunks):.0f} chars)")
    
    # Process chunks in parallel with a thread pool
    all_actions = []
    
    # Function to process a single chunk
    def process_chunk(idx, chunk_text):
        print(f"Processing chunk {idx+1}/{len(chunks)} (length: {len(chunk_text)} chars)")
        start_time = time.time()
        
        # Add a chunk identifier for traceability and to help with ordering
        chunk_prefix = f"Chunk {idx+1} of {len(chunks)}: "
        augmented_text = chunk_prefix + chunk_text
        
        # Process the chunk and get actions
        try:
            chunk_actions = simple_process_story.remote(augmented_text, provider=provider)
            elapsed = time.time() - start_time
            
            # Add chunk metadata to each action
            for action in chunk_actions:
                action["chunk_idx"] = idx
                action["chunk_start_char"] = start_idx + (idx * (max_chunk_size - chunk_overlap))
                
                # Adjust temporal_order_id to account for position in the book
                # This ensures actions from later chunks have higher IDs
                if "temporal_order_id" in action:
                    action["original_temporal_order_id"] = action["temporal_order_id"]
                    action["temporal_order_id"] = action["temporal_order_id"] + (idx * 1000)
            
            print(f"✅ Chunk {idx+1}: Extracted {len(chunk_actions)} actions in {elapsed:.2f}s")
            return chunk_actions
        except Exception as e:
            print(f"❌ Error processing chunk {idx+1}: {str(e)}")
            return []
    
    # Process chunks in parallel with ThreadPoolExecutor
    print(f"Processing chunks in parallel with concurrency={concurrency}...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {executor.submit(process_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
        
        for future in as_completed(future_to_idx):
            chunk_idx = future_to_idx[future]
            try:
                chunk_actions = future.result()
                all_actions.extend(chunk_actions)
                print(f"Chunk {chunk_idx+1} complete. Total actions so far: {len(all_actions)}")
            except Exception as e:
                print(f"Error getting result from chunk {chunk_idx+1}: {str(e)}")
    
    elapsed = time.time() - start_time
    print(f"All chunks processed in {timedelta(seconds=int(elapsed))}. Total actions: {len(all_actions)}")
    
    # Sort actions by temporal_order_id to maintain narrative order
    all_actions.sort(key=lambda x: x.get('temporal_order_id', 0))
    
    # Remove potential duplicates (similar actions from overlapping chunks)
    deduplicated_actions = []
    seen_actions = set()
    
    for action in all_actions:
        # Create a simple signature for the action to identify duplicates
        action_sig = f"{action.get('source', '')}_{action.get('action', '')}_{action.get('consequence', '')}"
        
        # If we haven't seen this action before, add it
        if action_sig not in seen_actions:
            deduplicated_actions.append(action)
            seen_actions.add(action_sig)
    
    print(f"After deduplication: {len(deduplicated_actions)} unique actions")
    
    # Re-number the temporal_order_id to be sequential
    for i, action in enumerate(deduplicated_actions, 1):
        action["temporal_order_id"] = i
    
    return deduplicated_actions

@app.function(
    volumes={"/results": results_vol},
    image=image
)
def list_processed_results():
    """List all processed results from the results volume."""
    print("Starting list_processed_results function")
    results_dir = "/results/gutenberg_processed"
    
    print(f"Checking results directory: {results_dir}")
    if not os.path.exists(results_dir):
        print("Results directory does not exist")
        return {"error": "Results directory does not exist"}
    
    result_files = os.listdir(results_dir)
    print(f"Found {len(result_files)} files in the results directory")
    
    if not result_files:
        print("No processed results found")
        return {"message": "No processed results found"}
    
    results_summary = []
    print("Processing result files:")
    for file in result_files:
        print(f"  - Processing file: {file}")
        try:
            file_path = os.path.join(results_dir, file)
            print(f"    Reading file from: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            
            metadata = result_data.get("book_metadata", {})
            processing_info = result_data.get("processing_info", {})
            
            summary = {
                "file": file,
                "title": metadata.get("title", "Unknown"),
                "book_id": metadata.get("book_id", "Unknown"),
                "provider": processing_info.get("provider", "Unknown"),
                "timestamp": processing_info.get("timestamp", "Unknown"),
                "actions_count": len(result_data.get("actions", [])) if "actions" in result_data else 0,
                "has_error": "error" in result_data
            }
            results_summary.append(summary)
            print(f"    Summary: {summary['title']} - {summary['actions_count']} actions")
        except Exception as e:
            print(f"    Error processing file {file}: {str(e)}")
            results_summary.append({
                "file": file,
                "error": str(e)
            })
    
    print(f"Processed {len(results_summary)} result files")
    return {"results": results_summary}

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("my-openai-secret")]
)
def simple_process_story(story_text: str, provider: str = "groq"):
    """
    A simplified version of the process_story function that processes a story text
    and extracts structured actions using LLMs.
    
    Args:
        story_text (str): The narrative text to process
        provider (str): LLM provider to use ("openai", "anthropic", or "groq")
        
    Returns:
        list: List of extracted actions as dictionaries
    """
    import json
    import os
    import traceback
    
    print(f"Starting simple_process_story with provider: {provider}")
    print(f"Input text length: {len(story_text)} characters")
    
    # Define the action schema to be extracted
    action_schema = {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Who performed the action"},
            "source_type": {"type": "string", "description": "Type of the source (e.g., 'people', 'character', 'creature', 'object')"},
            "source_is_character": {"type": "boolean", "description": "Whether the source is a character"},
            "target": {"type": ["string", "null"], "description": "Who or what was affected by the action"},
            "target_type": {"type": ["string", "null"], "description": "Type of the target (e.g., 'people', 'character', 'creature', 'object')"},
            "target_is_character": {"type": ["boolean", "null"], "description": "Whether the target is a character"},
            "action": {"type": "string", "description": "What action was performed"},
            "consequence": {"type": "string", "description": "What was the result of the action"},
            "text_describing_the_action": {"type": "string", "description": "Text from the story describing the action"},
            "text_describing_the_consequence": {"type": "string", "description": "Text from the story describing the consequence"},
            "location": {"type": ["array", "string", "null"], "description": "Where the action took place (can be a string or array of strings for hierarchical locations)"},
            "temporal_order_id": {"type": "integer", "description": "Order of the action in the narrative"}
        },
        "required": ["source", "source_is_character", "action", "consequence", "text_describing_the_action", 
                   "text_describing_the_consequence", "temporal_order_id"]
    }
    
    # Check which provider to use
    if provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY", "")
        model = os.environ.get("GROQ_MODEL", "qwen-2.5-32b")  # Get model from env or use qwen-2.5-32b
        print(f"Using Groq with model: {model}")
        
        # Verify API key is set
        if not api_key:
            print("WARNING: GROQ_API_KEY is not set")
        else:
            print("GROQ_API_KEY is set")
            
        from groq import Groq
        client = Groq(api_key=api_key)
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        model = "claude-3-haiku-20240307"
        print(f"Using Anthropic with model: {model}")
        
        # Verify API key is set
        if not api_key:
            print("WARNING: ANTHROPIC_API_KEY is not set")
        else:
            print("ANTHROPIC_API_KEY is set")
            
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
    else:  # default to OpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model = "gpt-3.5-turbo"
        print(f"Using OpenAI with model: {model}")
        
        # Verify API key is set
        if not api_key:
            print("WARNING: OPENAI_API_KEY is not set")
        else:
            print("OPENAI_API_KEY is set")
            
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    
    # Create the system prompt
    system_prompt = """
    You are an expert narrative analyst, skilled at extracting structured actions and their consequences from text.
    Your task is to analyze the provided narrative text, identify key actions taken by entities,
    and convert them into a structured output.
    """
    
    # Create the prompt for action extraction
    user_prompt = f"""
    Extract all significant actions from the following text as a list of JSON objects.
    Each action should include:
    - source: Who performed the action
    - source_type: Type of the source (e.g., 'people', 'character', 'creature', 'object')
    - source_is_character: Whether the source is a character (true/false)
    - target: Who or what was affected by the action
    - target_type: Type of the target (e.g., 'people', 'character', 'creature', 'object')
    - target_is_character: Whether the target is a character (true/false)
    - action: What they did
    - consequence: What happened as a result
    - text_describing_the_action: The actual text describing the action
    - text_describing_the_consequence: The actual text describing the consequence
    - location: Where the action took place (can be a string or array of strings for hierarchical locations)
    - temporal_order_id: Sequential order in the narrative, starting from 1
    
    Here's the text to analyze:
    {story_text}
    
    Respond with ONLY a valid JSON array containing the extracted actions.
    """
    
    print("Sending request to LLM API...")
    
    # Call the appropriate API based on the provider
    try:
        if provider == "groq":
            print("Making request to Groq API...")
            
            # Special handling for Qwen models using structured output with Pydantic
            if "qwen" in model.lower():
                try:
                    print("Using Pydantic structured output for Qwen model")
                    
                    # Import required libraries for structured output
                    try:
                        import instructor
                        from pydantic import BaseModel, Field
                        from typing import List, Optional, Union, Literal
                    except ImportError:
                        print("Installing instructor and pydantic...")
                        import subprocess
                        subprocess.check_call(["pip", "install", "instructor", "pydantic"])
                        import instructor
                        from pydantic import BaseModel, Field
                        from typing import List, Optional, Union, Literal
                    
                    # Define Pydantic model for a narrative action
                    class NarrativeAction(BaseModel):
                        source: str = Field(description="Who performed the action")
                        source_type: str = Field(description="Type of the source (e.g., 'people', 'character', 'creature', 'object')")
                        source_is_character: bool = Field(description="Whether the source is a character")
                        target: Optional[str] = Field(None, description="Who or what was affected by the action")
                        target_type: Optional[str] = Field(None, description="Type of the target (e.g., 'people', 'character', 'creature', 'object')")
                        target_is_character: Optional[bool] = Field(None, description="Whether the target is a character")
                        action: str = Field(description="What action was performed")
                        consequence: str = Field(description="What was the result of the action")
                        text_describing_the_action: str = Field(description="Text from the story describing the action")
                        text_describing_the_consequence: str = Field(description="Text from the story describing the consequence")
                        location: Optional[Union[str, List[str]]] = Field(None, description="Where the action took place")
                        temporal_order_id: int = Field(description="Order of the action in the narrative")
                    
                    class NarrativeActions(BaseModel):
                        actions: List[NarrativeAction] = Field(description="List of extracted narrative actions")
                    
                    # Patch the Groq client with instructor
                    patched_client = instructor.from_groq(client)
                    
                    # Create a specialized system prompt for structured extraction
                    structured_system_prompt = """
                    You are an expert narrative analyst, skilled at extracting structured actions and their consequences from text.
                    Your task is to analyze the provided narrative text, identify key actions taken by entities,
                    and convert them into a structured JSON output that exactly matches the required schema.
                    """
                    
                    # Create a specialized user prompt for structured extraction
                    structured_user_prompt = f"""
                    Extract all significant actions from the following text.
                    
                    Text to analyze:
                    {story_text}
                    
                    Identify each action with these details:
                    - Who performed the action (source)
                    - What type of entity they are (character, object, etc.)
                    - Who/what was affected (target)
                    - The action itself
                    - The consequence of the action
                    - Where it took place
                    - The order in the narrative
                    
                    Respond with structured data only.
                    """
                    
                    # Make the request with structured output
                    response = patched_client.chat.completions.create(
                        model=model,
                        response_model=NarrativeActions,
                        messages=[
                            {"role": "system", "content": structured_system_prompt},
                            {"role": "user", "content": structured_user_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=2048
                    )
                    
                    # Extract actions from structured response
                    actions = [action.model_dump() for action in response.actions]
                    print(f"Successfully extracted {len(actions)} actions with Pydantic structured output")
                    return actions
                    
                except Exception as e:
                    print(f"Error using Pydantic structured output: {str(e)}")
                    print("Falling back to standard prompt approach")
                    # Fall back to the standard prompt approach
                    
            # Try using tools API with Qwen
            if "qwen" in model.lower():
                try:
                    print("Using specialized Qwen tools API approach")
                    
                    # Define the tool for extracting actions
                    tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": "extract_narrative_actions",
                                "description": "Extract narrative actions from a text",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "actions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "source": {
                                                        "type": "string",
                                                        "description": "Who performed the action"
                                                    },
                                                    "source_type": {
                                                        "type": "string",
                                                        "description": "Type of the source (e.g., 'people', 'character', 'creature', 'object')"
                                                    },
                                                    "source_is_character": {
                                                        "type": "boolean",
                                                        "description": "Whether the source is a character"
                                                    },
                                                    "target": {
                                                        "type": "string",
                                                        "description": "Who or what was affected by the action"
                                                    },
                                                    "target_type": {
                                                        "type": "string",
                                                        "description": "Type of the target (e.g., 'people', 'character', 'creature', 'object')"
                                                    },
                                                    "target_is_character": {
                                                        "type": "boolean",
                                                        "description": "Whether the target is a character"
                                                    },
                                                    "action": {
                                                        "type": "string",
                                                        "description": "What action was performed"
                                                    },
                                                    "consequence": {
                                                        "type": "string",
                                                        "description": "What was the result of the action"
                                                    },
                                                    "text_describing_the_action": {
                                                        "type": "string",
                                                        "description": "Text from the story describing the action"
                                                    },
                                                    "text_describing_the_consequence": {
                                                        "type": "string",
                                                        "description": "Text from the story describing the consequence"
                                                    },
                                                    "location": {
                                                        "type": ["string", "array"],
                                                        "description": "Where the action took place (can be a string or array of strings for hierarchical locations)"
                                                    },
                                                    "temporal_order_id": {
                                                        "type": "integer",
                                                        "description": "Order of the action in the narrative"
                                                    }
                                                },
                                                "required": ["source", "source_type", "source_is_character", "action", 
                                                           "consequence", "text_describing_the_action", 
                                                           "text_describing_the_consequence", "temporal_order_id"]
                                            }
                                        }
                                    },
                                    "required": ["actions"]
                                }
                            }
                        }
                    ]
                    
                    # Create the tool-specific prompts
                    tool_system_prompt = """
                    You are an expert narrative analyst, skilled at extracting structured actions and their consequences from text.
                    Use the extract_narrative_actions tool to convert the text into structured data.
                    """
                    
                    tool_user_prompt = f"""
                    Extract all important actions from this story text:
                    
                    {story_text}
                    
                    Use the extract_narrative_actions tool to provide a complete and structured analysis.
                    """
                    
                    # Make the request with tool use
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": tool_system_prompt},
                            {"role": "user", "content": tool_user_prompt}
                        ],
                        tools=tools,
                        tool_choice={"type": "function", "function": {"name": "extract_narrative_actions"}},
                        temperature=0.2,
                        max_tokens=2048
                    )
                    
                    # Extract actions from tool response
                    if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                        tool_call = response.choices[0].message.tool_calls[0]
                        if tool_call.function.name == "extract_narrative_actions":
                            tool_args = json.loads(tool_call.function.arguments)
                            if "actions" in tool_args and isinstance(tool_args["actions"], list):
                                actions = tool_args["actions"]
                                print(f"Successfully extracted {len(actions)} actions using tool call")
                                return actions
                    
                    # If we didn't get a proper tool response, fall back to standard approach
                    print("Tool call didn't return proper actions, falling back to standard approach")
                
                except Exception as e:
                    print(f"Error using tool-based approach: {str(e)}")
                    print("Falling back to standard prompt approach")
                    
            # Standard JSON mode approach as fallback
            try:
                # Create a more explicit system prompt for Qwen
                qwen_system_prompt = """
                You are an expert narrative analyst, skilled at extracting structured actions and their consequences from text.
                Your task is to analyze the provided narrative text, identify key actions taken by entities,
                and convert them into a structured JSON output. Please follow the exact format requested and only output valid JSON.
                """
                
                # Create a more explicit prompt for Qwen that specifies the JSON format more clearly
                qwen_user_prompt = f"""
                Please extract all significant actions from the following text as a list of JSON objects.
                
                Here's the exact format I need for each action (with all fields required):
                {{
                    "source": "Who performed the action",
                    "source_type": "Type of the source (e.g., 'people', 'character', 'creature', 'object')",
                    "source_is_character": true or false,
                    "target": "Who or what was affected by the action (or null if none)",
                    "target_type": "Type of the target (e.g., 'people', 'character', 'creature', 'object') (or null if none)",
                    "target_is_character": true or false (or null if none),
                    "action": "What they did",
                    "consequence": "What happened as a result",
                    "text_describing_the_action": "The actual text describing the action",
                    "text_describing_the_consequence": "The actual text describing the consequence",
                    "location": "Where the action took place (can be a string or array of strings)",
                    "temporal_order_id": sequential number starting from 1
                }}
                
                The text to analyze is:
                {story_text}
                
                Wrap the objects in a JSON array like this: [{{action1}}, {{action2}}]
                I need your response to ONLY contain this JSON array with no additional text, comments or explanations.
                """
                
                # Try with JSON mode if available
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": qwen_system_prompt},
                            {"role": "user", "content": qwen_user_prompt}
                        ],
                        response_format={"type": "json_object"}
                    )
                except:
                    # Fall back to regular completion if JSON mode fails
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": qwen_system_prompt},
                            {"role": "user", "content": qwen_user_prompt}
                        ]
                    )
                
                result_text = response.choices[0].message.content
                print("Received response from Groq API")
            except Exception as e:
                print(f"Error in standard Groq API call: {str(e)}")
                # Try to extract the failed_generation from the error if possible
                if hasattr(e, 'response') and hasattr(e.response, 'json'):
                    try:
                        error_json = e.response.json()
                        if 'error' in error_json and 'failed_generation' in error_json['error']:
                            print("Found failed_generation in error response, attempting to extract actions")
                            result_text = error_json['error']['failed_generation']
                        else:
                            print("Error response does not contain failed_generation")
                            raise e
                    except Exception as json_e:
                        print(f"Failed to extract JSON from error: {str(json_e)}")
                        raise e
                else:
                    # If we can't extract the JSON, re-raise the original error
                    raise e
            
        elif provider == "anthropic":
            print("Making request to Anthropic API...")
            response = client.messages.create(
                model=model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000
            )
            result_text = response.content[0].text
            print("Received response from Anthropic API")
            
        else:  # OpenAI
            print("Making request to OpenAI API...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            result_text = response.choices[0].message.content
            print("Received response from OpenAI API")
        
        # If we have a string result (non-structured output), parse it
        if 'result_text' in locals():
            print(f"Response text length: {len(result_text)} characters")
            print("Response text sample (first 100 chars):", result_text[:100])
            
            # Parse the JSON result
            # This might be wrapped in an object with a key like "actions" or just be an array
            try:
                print("Parsing JSON response...")
                # Clean up the JSON string to handle potential formatting issues
                result_text = result_text.strip()
                
                # Try a more robust approach to extract JSON when the response may contain Markdown or other formatting
                if result_text.startswith("```json"):
                    # Extract the JSON part from a code block
                    json_text = result_text.split("```json")[1].split("```")[0].strip()
                    print("Extracted JSON from code block")
                elif result_text.startswith("```"):
                    # Extract the JSON part from a code block without language specifier
                    json_text = result_text.split("```")[1].split("```")[0].strip()
                    print("Extracted JSON from generic code block")
                else:
                    # Use the text as is
                    json_text = result_text
                
                result_json = json.loads(json_text)
                print(f"Parsed JSON. Type: {type(result_json).__name__}")
                
                actions = []
                if isinstance(result_json, dict):
                    print(f"Result is a dictionary with keys: {list(result_json.keys())}")
                    if "actions" in result_json:
                        actions = result_json["actions"]
                        print(f"Found 'actions' key with {len(actions)} actions")
                    else:
                        # Try to find an array in the response
                        for key, value in result_json.items():
                            if isinstance(value, list):
                                actions = value
                                print(f"Found list in key '{key}' with {len(actions)} actions")
                                break
                        else:
                            print("No list found in the response dictionary")
                elif isinstance(result_json, list):
                    actions = result_json
                    print(f"Result is a list with {len(actions)} actions")
                else:
                    print(f"Unexpected result type: {type(result_json).__name__}")
                
                print(f"Final actions count: {len(actions)}")
                return actions
                
            except json.JSONDecodeError as json_error:
                print(f"Failed to parse JSON response: {str(json_error)}")
                print("First 200 chars of response:", result_text[:200])
                
                # Try extracting JSON from the response text using regex (more liberal approach)
                import re
                try:
                    print("Attempting to extract JSON with regex...")
                    json_pattern = r'\{[^}]*\}'
                    matches = re.findall(json_pattern, result_text)
                    if matches:
                        print(f"Found {len(matches)} potential JSON objects")
                        actions = []
                        for match in matches:
                            try:
                                action = json.loads(match)
                                actions.append(action)
                            except:
                                pass
                        
                        if actions:
                            print(f"Successfully extracted {len(actions)} actions with regex")
                            return actions
                except Exception as e:
                    print(f"Regex extraction failed: {str(e)}")
                
                # If all attempts fail, return empty list
                return []
    
    except Exception as e:
        print(f"Error in simple_process_story: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # For Groq errors, try to extract the failed_generation directly from the exception string
        if 'groq' in str(e).lower() and 'failed_generation' in str(e):
            try:
                print("Trying to extract actions from error message...")
                # Extract the JSON part from the error message
                import re
                failed_generation = re.search(r"'failed_generation': '(\{.*?\})'", str(e), re.DOTALL)
                if failed_generation:
                    failed_json = failed_generation.group(1)
                    # Clean up escaped quotes and newlines
                    failed_json = failed_json.replace('\\n', '\n').replace('\\"', '"')
                    
                    # Parse the JSON
                    try:
                        result_json = json.loads(failed_json)
                        if "actions" in result_json and isinstance(result_json["actions"], list):
                            print(f"Successfully extracted {len(result_json['actions'])} actions from error")
                            return result_json["actions"]
                    except json.JSONDecodeError:
                        print("Failed to parse JSON from error message")
            except Exception as extract_e:
                print(f"Failed to extract JSON from error message: {str(extract_e)}")
        
        return []

@app.function(
    volumes={
        "/data": gutenberg_data_vol,
        "/results": results_vol,
    },
    image=image,
    memory=4096,  # Increased memory for batch processing
    timeout=3600  # 60 minutes timeout for batch processing
)
def batch_process_books(
    start_index: int = 0, 
    count: int = 20,
    provider: str = "groq", 
    max_chunk_size: int = 4000,
    chunk_overlap: int = 200,
    skip_existing: bool = True,
    book_concurrency: int = 3,  # Number of books to process in parallel
    chunk_concurrency: int = 5  # Number of chunks per book to process in parallel
):
    """
    Process multiple books from the Gutenberg dataset in batch.
    
    Args:
        start_index: Index of the first book to process
        count: Number of books to process
        provider: LLM provider to use (e.g., "groq", "anthropic", "openai")
        max_chunk_size: Maximum size of each text chunk
        chunk_overlap: Overlap between consecutive chunks
        skip_existing: Whether to skip books that have already been processed
        book_concurrency: Number of books to process in parallel
        chunk_concurrency: Number of chunks per book to process in parallel
        
    Returns:
        dict: Summary of the batch processing
    """
    import time
    from datetime import timedelta
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    print(f"Starting batch_process_books:")
    print(f"  - Start index: {start_index}")
    print(f"  - Count: {count}")
    print(f"  - Provider: {provider}")
    print(f"  - Max chunk size: {max_chunk_size}")
    print(f"  - Chunk overlap: {chunk_overlap}")
    print(f"  - Skip existing: {skip_existing}")
    print(f"  - Book concurrency: {book_concurrency}")
    print(f"  - Chunk concurrency: {chunk_concurrency}")
    
    # Ensure results directory exists
    results_dir = "/results/gutenberg_processed"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get list of available books
    print("Getting list of available books...")
    book_files = sorted(os.listdir("/data/gutenberg"))
    total_books = len(book_files)
    print(f"Found {total_books} books in the dataset")
    
    # Validate indices
    if start_index < 0:
        start_index = 0
    if start_index >= total_books:
        return {"error": f"Start index {start_index} exceeds dataset size {total_books}"}
    
    end_index = min(start_index + count, total_books)
    actual_count = end_index - start_index
    print(f"Will process books from index {start_index} to {end_index - 1} ({actual_count} books)")
    
    # Get list of already processed books if skipping existing
    processed_files = set()
    if skip_existing:
        try:
            for filename in os.listdir(results_dir):
                if f"_processed_{provider}.json" in filename:
                    processed_files.add(filename.split("_processed_")[0])
            print(f"Found {len(processed_files)} already processed files to skip")
        except Exception as e:
            print(f"Error getting list of processed files: {str(e)}")
            # Continue anyway - we'll just process all files
    
    # Helper function to process a single book
    def process_book_wrapper(index):
        book_file = book_files[index]
        book_base = os.path.splitext(book_file)[0]
        
        # Check if already processed
        if skip_existing and book_base in processed_files:
            return {
                "book_index": index,
                "book_file": book_file,
                "success": True,
                "skipped": True,
                "actions_count": 0,
                "error": None,
                "time_taken": 0
            }
        
        print(f"\n--- Processing book {index} of {end_index-1}: {book_file} ---\n")
        start_time = time.time()
        
        try:
            # Process the book using our updated function with parallel chunk processing
            result = process_book.remote(
                index, 
                provider, 
                max_chunk_size=max_chunk_size, 
                chunk_overlap=chunk_overlap, 
                concurrency=chunk_concurrency
            )
            
            time_taken = time.time() - start_time
            
            if "success" in result and result["success"]:
                print(f"Successfully processed {book_file} with {result['actions_count']} actions in {time_taken:.2f}s")
                return {
                    "book_index": index,
                    "book_file": book_file,
                    "success": True,
                    "skipped": False,
                    "actions_count": result.get("actions_count", 0),
                    "error": None,
                    "time_taken": time_taken
                }
            else:
                print(f"Failed to process {book_file}: {result.get('error', 'Unknown error')} in {time_taken:.2f}s")
                return {
                    "book_index": index,
                    "book_file": book_file,
                    "success": False,
                    "skipped": False,
                    "actions_count": 0,
                    "error": result.get("error", "Unknown error"),
                    "time_taken": time_taken
                }
                
        except Exception as e:
            time_taken = time.time() - start_time
            print(f"Error processing book {book_file}: {str(e)} in {time_taken:.2f}s")
            return {
                "book_index": index,
                "book_file": book_file,
                "success": False,
                "skipped": False,
                "error": str(e),
                "time_taken": time_taken
            }
    
    # Process books with parallel execution
    results = []
    successful = 0
    skipped = 0
    failed = 0
    total_actions = 0
    total_time = 0
    
    batch_start_time = time.time()
    
    print(f"\nStarting parallel processing with book_concurrency={book_concurrency}...")
    
    indices_to_process = list(range(start_index, end_index))
    with ThreadPoolExecutor(max_workers=book_concurrency) as executor:
        futures = {executor.submit(process_book_wrapper, i): i for i in indices_to_process}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            # Update counters
            if result.get("skipped", False):
                skipped += 1
            elif result.get("success", False):
                successful += 1
                total_actions += result.get("actions_count", 0)
            else:
                failed += 1
                
            total_time += result.get("time_taken", 0)
            
            # Display progress
            completed += 1
            progress = (completed / actual_count) * 100
            elapsed = time.time() - batch_start_time
            
            # Calculate estimated time remaining
            if completed > 0 and elapsed > 0:
                books_per_second = completed / elapsed
                remaining_books = actual_count - completed
                eta_seconds = remaining_books / books_per_second if books_per_second > 0 else 0
                eta = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta = "unknown"
                
            print(f"\nProgress: {completed}/{actual_count} ({progress:.2f}%)")
            print(f"Success: {successful}, Failed: {failed}, Skipped: {skipped}")
            print(f"Total actions: {total_actions}, Avg actions per book: {total_actions/max(1, successful):.2f}")
            print(f"Elapsed: {str(timedelta(seconds=int(elapsed)))}, ETA: {eta}")
    
    batch_end_time = time.time()
    batch_duration = batch_end_time - batch_start_time
    
    # Save batch results summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "start_index": start_index,
        "end_index": end_index - 1,
        "requested_count": count,
        "processed_count": successful + failed,
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "total_actions": total_actions,
        "avg_actions_per_book": total_actions / max(1, successful),
        "total_time_seconds": batch_duration,
        "avg_time_per_book": batch_duration / max(1, successful + failed),
        "book_concurrency": book_concurrency,
        "chunk_concurrency": chunk_concurrency,
        "max_chunk_size": max_chunk_size,
        "chunk_overlap": chunk_overlap,
        "results": results
    }
    
    # Add timestamp to batch summary filename to avoid overwriting
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_result_path = os.path.join(
        results_dir, 
        f"batch_summary_{start_index}_{end_index-1}_{provider}_{timestamp_str}.json"
    )
    
    with open(batch_result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nBatch processing completed:")
    print(f"  - Processed {successful + failed} books")
    print(f"  - Successfully extracted actions from {successful} books")
    print(f"  - Failed to process {failed} books")
    print(f"  - Skipped {skipped} already processed books")
    print(f"  - Total actions extracted: {total_actions}")
    print(f"  - Average actions per book: {total_actions / max(1, successful):.2f}")
    print(f"  - Total batch processing time: {str(timedelta(seconds=int(batch_duration)))}")
    print(f"  - Average time per book: {batch_duration / max(1, successful + failed):.2f} seconds")
    print(f"  - Saved batch summary to {batch_result_path}")
    
    return summary

@app.function(
    volumes={"/results": results_vol},
    image=image
)
def analyze_results(provider: str = "groq"):
    """
    Analyze all processed results to generate statistics.
    
    Args:
        provider: The LLM provider used for processing
        
    Returns:
        dict: Analysis results with statistics
    """
    print(f"Starting analyze_results for provider: {provider}")
    results_dir = "/results/gutenberg_processed"
    
    if not os.path.exists(results_dir):
        print("Results directory does not exist")
        return {"error": "Results directory does not exist"}
    
    # Find all processed files for the specified provider
    processed_files = [
        f for f in os.listdir(results_dir) 
        if f.endswith(f"_processed_{provider}.json")
    ]
    
    print(f"Found {len(processed_files)} processed files for provider '{provider}'")
    
    if not processed_files:
        return {"message": f"No processed results found for provider '{provider}'"}
    
    # Initialize statistics
    stats = {
        "total_books": len(processed_files),
        "total_actions": 0,
        "actions_per_book": {},
        "books_without_actions": 0,
        "books_with_actions": 0,
        "max_actions": 0,
        "max_actions_book": "",
        "min_actions": float('inf'),
        "min_actions_book": "",
        "sources": {},
        "actions": {},
        "source_types": {},
        "target_types": {},
        "locations": {}
    }
    
    # Process each file
    for filename in processed_files:
        try:
            file_path = os.path.join(results_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            book_id = data.get("book_metadata", {}).get("book_id", "unknown")
            actions = data.get("actions", [])
            action_count = len(actions)
            
            stats["total_actions"] += action_count
            stats["actions_per_book"][book_id] = action_count
            
            if action_count == 0:
                stats["books_without_actions"] += 1
            else:
                stats["books_with_actions"] += 1
            
            if action_count > stats["max_actions"]:
                stats["max_actions"] = action_count
                stats["max_actions_book"] = book_id
                
            if action_count < stats["min_actions"] and action_count > 0:
                stats["min_actions"] = action_count
                stats["min_actions_book"] = book_id
            
            # Analyze action details
            for action in actions:
                # Count sources
                source = action.get("source", "unknown")
                stats["sources"][source] = stats["sources"].get(source, 0) + 1
                
                # Count action types
                action_type = action.get("action", "unknown")
                stats["actions"][action_type] = stats["actions"].get(action_type, 0) + 1
                
                # Count source types
                source_type = action.get("source_type", "unknown")
                stats["source_types"][source_type] = stats["source_types"].get(source_type, 0) + 1
                
                # Count target types
                target_type = action.get("target_type", "unknown")
                if target_type:  # Only count if not None
                    stats["target_types"][target_type] = stats["target_types"].get(target_type, 0) + 1
                
                # Count locations
                location = action.get("location", "unknown")
                if location:
                    if isinstance(location, list):
                        # For list locations, count each one
                        for loc in location:
                            stats["locations"][loc] = stats["locations"].get(loc, 0) + 1
                    else:
                        stats["locations"][location] = stats["locations"].get(location, 0) + 1
                        
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Calculate averages
    if stats["total_books"] > 0:
        stats["avg_actions_per_book"] = stats["total_actions"] / stats["total_books"]
    else:
        stats["avg_actions_per_book"] = 0
        
    if stats["books_with_actions"] > 0:
        stats["avg_actions_per_book_with_actions"] = stats["total_actions"] / stats["books_with_actions"]
    else:
        stats["avg_actions_per_book_with_actions"] = 0
    
    # Get top items
    def get_top_items(items_dict, count=10):
        sorted_items = sorted(items_dict.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:count])
    
    stats["top_sources"] = get_top_items(stats["sources"])
    stats["top_actions"] = get_top_items(stats["actions"])
    stats["top_source_types"] = get_top_items(stats["source_types"])
    stats["top_target_types"] = get_top_items(stats["target_types"])
    stats["top_locations"] = get_top_items(stats["locations"])
    
    # Fix min_actions if no books had actions
    if stats["min_actions"] == float('inf'):
        stats["min_actions"] = 0
        stats["min_actions_book"] = "none"
    
    # Save analysis results
    analysis_path = os.path.join(results_dir, f"analysis_{provider}.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"Analysis complete for provider '{provider}'")
    print(f"  - Total books: {stats['total_books']}")
    print(f"  - Total actions: {stats['total_actions']}")
    print(f"  - Average actions per book: {stats['avg_actions_per_book']:.2f}")
    print(f"  - Books with actions: {stats['books_with_actions']} ({stats['books_with_actions']/stats['total_books']*100:.2f}%)")
    print(f"  - Books without actions: {stats['books_without_actions']} ({stats['books_without_actions']/stats['total_books']*100:.2f}%)")
    print(f"  - Max actions: {stats['max_actions']} (book: {stats['max_actions_book']})")
    print(f"  - Min actions: {stats['min_actions']} (book: {stats['min_actions_book']})")
    print(f"  - Analysis saved to {analysis_path}")
    
    return stats

@app.function(
    volumes={
        "/data": gutenberg_data_vol,
        "/results": results_vol,
    }
)
def view_sample_actions(book_index: int, provider: str = "groq"):
    """Display the extracted actions for a specific book.
    
    Args:
        book_index: Index of the book to view
        provider: The provider used to process the book (default: groq)
    """
    import json
    import os
    
    print(f"Looking for actions for book {book_index} processed with {provider}")
    
    # Check if the processed file exists
    result_file = f"/results/gutenberg_processed/book_{book_index}_processed_{provider}.json"
    
    if not os.path.exists(result_file):
        print(f"Processed file not found: {result_file}")
        return
    
    # Load the processed data
    with open(result_file, "r") as f:
        processed_data = json.load(f)
    
    # Extract original book info
    book_id = processed_data.get("book_id", "Unknown")
    actions = processed_data.get("actions", [])
    
    # Display the actions
    print(f"\nBook ID: {book_id}")
    print(f"Number of actions: {len(actions)}")
    print("\nExtracted Actions:")
    print("="*50)
    
    for i, action in enumerate(actions, 1):
        print(f"Action {i}:")
        print(f"  Source: {action.get('source', 'N/A')}")
        print(f"  Action: {action.get('action', 'N/A')}")
        print(f"  Consequence: {action.get('consequence', 'N/A')}")
        print(f"  Temporal Order ID: {action.get('temporal_order_id', 'N/A')}")
        print("-"*50)
    
    return processed_data

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "batch":
            # Run batch processing
            start_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            count = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            provider = sys.argv[4] if len(sys.argv) > 4 else "groq"
            chunk_size = int(sys.argv[5]) if len(sys.argv) > 5 else 4000
            book_concurrency = int(sys.argv[6]) if len(sys.argv) > 6 else 3
            
            print(f"Running batch processing with parallel chunks:")
            print(f"  - Start index: {start_index}")
            print(f"  - Count: {count}")
            print(f"  - Provider: {provider}")
            print(f"  - Chunk size: {chunk_size}")
            print(f"  - Book concurrency: {book_concurrency}")
            
            modal.run(batch_process_books, 
                      start_index=start_index, 
                      count=count, 
                      provider=provider, 
                      max_chunk_size=chunk_size,
                      book_concurrency=book_concurrency)
            
        elif command == "analyze":
            # Run analysis
            provider = sys.argv[2] if len(sys.argv) > 2 else "groq"
            print(f"Running analysis for provider: {provider}")
            modal.run(analyze_results, provider=provider)
            
        elif command == "list":
            # List processed results
            print("Listing processed results")
            modal.run(list_processed_results)
            
        elif command == "single":
            # Process a single book
            book_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            provider = sys.argv[3] if len(sys.argv) > 3 else "groq"
            chunk_size = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
            chunk_overlap = int(sys.argv[5]) if len(sys.argv) > 5 else 200
            concurrency = int(sys.argv[6]) if len(sys.argv) > 6 else 5
            
            print(f"Processing single book with parallel chunks:")
            print(f"  - Book index: {book_index}")
            print(f"  - Provider: {provider}")
            print(f"  - Chunk size: {chunk_size}")
            print(f"  - Chunk overlap: {chunk_overlap}")
            print(f"  - Concurrency: {concurrency}")
            
            modal.run(process_book, 
                      book_index=book_index, 
                      provider=provider, 
                      max_chunk_size=chunk_size, 
                      chunk_overlap=chunk_overlap, 
                      concurrency=concurrency)
            
        elif command == "view":
            # View sample actions for a specific book
            book_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            provider = sys.argv[3] if len(sys.argv) > 3 else "groq"
            print(f"Viewing sample actions for book {book_index} processed with {provider}")
            modal.run(view_sample_actions, book_index=book_index, provider=provider)
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  single <book_index> <provider> <chunk_size> <chunk_overlap> <concurrency>")
            print("    - Process a single book with parallel chunks")
            print("  batch <start_index> <count> <provider> <chunk_size> <book_concurrency>")
            print("    - Process multiple books in parallel")
            print("  list - List all processed results")
            print("  analyze <provider> - Analyze processed results")
            print("  view <book_index> <provider> - View sample actions for a book")
    else:
        # Default: list available commands
        print("Gutenberg Dataset Processor")
        print("Available commands:")
        print("  single <book_index> <provider> <chunk_size> <chunk_overlap> <concurrency>")
        print("    - Process a single book with parallel chunks")
        print("  batch <start_index> <count> <provider> <chunk_size> <book_concurrency>")
        print("    - Process multiple books in parallel")
        print("  list - List all processed results")
        print("  analyze <provider> - Analyze processed results")
        print("  view <book_index> <provider> - View sample actions for a book")
        print("\nExamples:")
        print("  python process_gutenberg_test.py single 5 groq 4000 200 5")
        print("  python process_gutenberg_test.py batch 0 10 groq 4000 3")
        print("  python process_gutenberg_test.py list")
        print("  python process_gutenberg_test.py analyze groq")
        print("  python process_gutenberg_test.py view 5 groq") 
