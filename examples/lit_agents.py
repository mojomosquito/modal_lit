"""
Narrative Action Extractor using Modal Cloud Infrastructure

This script processes narrative text to extract structured actions and their consequences
using Large Language Models (LLMs) deployed on Modal's cloud infrastructure. It demonstrates:
1. Parallel processing of narrative segments
2. Structured output generation using LLMs
3. Cloud deployment with Modal
4. Environment variable management
5. Asynchronous execution

Date: March 2025
"""

# Standard library imports
import asyncio
import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Literal, Optional
from datetime import datetime

# Third-party imports
import modal
import polars as pl
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Local imports
from minference.ecs.caregistry import CallableRegistry
from minference.ecs.entity import EntityRegistry
from minference.threads.inference import InferenceOrchestrator, RequestLimits
from minference.threads.models import (ChatMessage, ChatThread, LLMConfig, CallableTool,
                          LLMClient, ResponseFormat, SystemPrompt, StructuredTool,
                          Usage, GeneratedJsonObject)
from minference.threads.utils import msg_dict_to_oai, msg_dict_to_anthropic, parse_json_string
from minference.threads.modal_inference import run_inference
from minference.threads.modal_utils import modal_image, get_modal_cache_dir, get_modal_results_dir, cache_vol, results_vol

###########################################
# Modal Configuration and Setup
###########################################

# Configure data paths
def get_data_dir():
    """Get the path to the data directory mounted in Modal."""
    return "/data"

# Create additional volume for dataset
data_vol = modal.Volume.from_name("gutenberg-data-vol")

# Initialize Modal application
app = modal.App("lit-agents")

# Load and prepare environment variables
load_dotenv()
env_vars = {
    "OPENAI_KEY": os.getenv("OPENAI_KEY"),
    "OPENAI_MODEL": os.getenv("OPENAI_MODEL"),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "ANTHROPIC_MODEL": os.getenv("ANTHROPIC_MODEL"),
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
    "GROQ_MODEL": os.getenv("GROQ_MODEL"),
}

# Build the image with all dependencies
image = (modal.Image.debian_slim()
    .pip_install(
        # Core ML dependencies
        "openai>=1.0.0",        # OpenAI API client
        "anthropic>=0.10.0",    # Anthropic API client
        "groq>=0.4.0",          # Groq API client
        "modal>=0.56.4272",     # Modal cloud infrastructure
        
        # CLI and utility libraries
        "typer>=0.9.0",         # CLI framework
        "rich>=12.0.0",         # Rich text formatting
        "python-dotenv>=1.0.0", # Environment variable management
        
        # Data processing and validation
        "polars>=0.10.0",       # Fast DataFrame library
        "pydantic>=2.0.0",      # Data validation
        "aiofiles>=23.1.0",     # Asynchronous file operations
        "jsonschema>=4.17.3",   # JSON Schema validation
        
        # Code analysis and tokenization
        "libcst>=1.0.0",        # Concrete Syntax Tree for Python
        "tiktoken>=0.5.0",      # Fast BPE tokenizer from OpenAI
    )
    # NOTE: The following lines fix deprecation warnings but are retained as warnings
    # to maintain compatibility with current Modal versions:
    # .add_local_python_source("minference")
    # .add_local_python_source("examples")
)

###########################################
# Data Models and Schemas
###########################################

class Action(BaseModel):
    """
    Structured representation of an action extracted from narrative text.
    
    This model captures the key elements of an action:
    - Who performed it (source)
    - Who/what was affected (target)
    - What happened (action)
    - What resulted (consequence)
    - Where it occurred (location)
    - When in the sequence (temporal_order_id)
    """
    # Source information
    source: str = Field(..., description="Identifier of the entity performing the action (e.g., 'Character 1')")
    source_type: Optional[str] = Field(None, description="Type of the source (e.g., 'character', 'creature', 'object')")
    source_is_character: bool = Field(..., description="Whether the source is a character")
    
    # Target information
    target: Optional[str] = Field(None, description="Identifier of the entity affected (e.g., 'Character 2' or 'Object 1')")
    target_type: Optional[str] = Field(None, description="Type of the target (e.g., 'character', 'creature', 'object')")
    target_is_character: Optional[bool] = Field(None, description="Whether the target is a character")
    
    # Action details
    action: str = Field(..., description="The action taken by the source")
    consequence: str = Field(..., description="The immediate outcome of the action")
    text_describing_the_action: str = Field(..., description="Narrative text describing the action")
    text_describing_the_consequence: str = Field(..., description="Narrative text describing the consequence")
    
    # Context information
    location: Optional[str] = Field(None, description="Where the action takes place")
    temporal_order_id: int = Field(..., description="Sequential ID indicating the order of actions in the narrative")

    class Config:
        extra = "allow"

# JSON schema for action extraction
action_schema = {
    "type": "object",
    "title": "Action",
    "description": "Structured representation of a character's action and its context in a narrative",
    "properties": {
        "source": {"type": "string", "description": "Identifier of the entity performing the action (e.g., 'Character 1')"},
        "source_type": {"type": ["string", "null"], "description": "Type of the source"},
        "source_is_character": {"type": "boolean", "description": "Whether the source is a character"},
        "target": {"type": ["string", "null"], "description": "Identifier of the entity affected"},
        "target_type": {"type": ["string", "null"], "description": "Type of the target"},
        "target_is_character": {"type": ["boolean", "null"], "description": "Whether the target is a character"},
        "action": {"type": "string", "description": "The action taken"},
        "consequence": {"type": "string", "description": "The immediate outcome"},
        "text_describing_the_action": {"type": "string", "description": "Narrative text describing the action"},
        "text_describing_the_consequence": {"type": "string", "description": "Narrative text describing the consequence"},
        "location": {"type": ["string", "null"], "description": "Where the action takes place"},
        "temporal_order_id": {"type": "integer", "description": "Sequential ID for narrative order"}
    },
    "required": [
        "source", "source_is_character", "action", "consequence",
        "text_describing_the_action", "text_describing_the_consequence", "temporal_order_id"
    ],
    "additionalProperties": False
}

###########################################
# Modal Processing Function
###########################################

@app.function(image=image, secrets=[modal.Secret.from_dict(env_vars)])
def process_story(story_text: str, provider: str = "openai"):
    """
    Process a story text using Modal cloud infrastructure.
    
    This function:
    1. Sets up the environment with necessary credentials
    2. Initializes the inference orchestrator and registries
    3. Configures the LLM for action extraction
    4. Processes the story in parallel segments
    5. Returns structured actions extracted from the text
    
    Args:
        story_text (str): The narrative text to process
        provider (str): LLM provider to use ("openai", "anthropic", or "groq")
        
    Returns:
        list: List of extracted actions as dictionaries
    """
    # Initialize registries
    EntityRegistry()
    CallableRegistry()

    # Define request limits for different LLM providers
    request_limits = {
        "oai": {
            "max_requests_per_minute": 10000,
            "max_tokens_per_minute": 200000000,
            "provider": "openai"
        },
        "anthropic": {
            "max_requests_per_minute": 1500,
            "max_tokens_per_minute": 2000000,
            "provider": "anthropic"
        },
        "groq": {
            "max_requests_per_minute": 2000,
            "max_tokens_per_minute": 3000000,
            "provider": "groq"
        }
    }

    # Configure action extraction tools
    action_extractor = StructuredTool(name="action_extractor", json_schema=action_schema, post_validate_schema=False)
    system_prompt = SystemPrompt(
        content="""
        You are an expert narrative analyst in an alternate universe, skilled at extracting structured actions and their consequences from text. 
        Your task is to analyze the provided narrative text, identify key actions taken by entities (labeled as 'Character 1', 'Character 2', etc.), 
        and convert them into a structured output. Focus on the source of the action, what they do, the consequence, and the narrative context, 
        while assigning a temporal order to each action.
        """,
        name="action_extractor"
    )

    # Determine LLM client and model based on provider
    if provider == "anthropic":
        llm_client = LLMClient.anthropic
        llm_model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
    elif provider == "groq":
        llm_client = LLMClient.groq
        llm_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    else:  # default to OpenAI
        llm_client = LLMClient.openai
        llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Configure LLM and create base thread
    llm_config = LLMConfig(
        client=llm_client,
        model=llm_model,
        response_format=ResponseFormat.tool,
        max_tokens=4000
    )
    thread = ChatThread(
        system_prompt=system_prompt,
        new_message="",
        llm_config=llm_config,
        forced_output=action_extractor,
        use_schema_instruction=True
    )

    # Process story segments in parallel
    story_segments = story_text.strip().split("\n\n")
    threads = []
    for i, segment in enumerate(story_segments):
        thread.fork(force=True, **{
            "new_message": f"Extract the action and reasoning from this narrative segment: {segment}"
        })
        threads.append(thread)

    # Run inference using Modal
    outputs = run_inference.remote(
        chat_threads=threads,
        env_vars=env_vars,
        request_limits=request_limits
    )
    
    return [output.json_object for output in outputs if output.json_object]

###########################################
# Main Entrypoint
###########################################

@app.local_entrypoint()
def main():
    """
    Main entrypoint for the Modal deployment.
    
    This function demonstrates:
    1. Processing a sample story
    2. Running batch processing on the Gutenberg dataset
    3. Showing how to use the CLI for more advanced operations
    """
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Check for command line args
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "batch":
            # Run batch processing with defaults
            logging.info("Starting batch processing of Gutenberg dataset")
            file_pattern = "gutenberg_chunkingprocessed_en-*.parquet"
            if len(sys.argv) > 2:
                file_pattern = sys.argv[2]
            
            batch_size = 10
            if len(sys.argv) > 3:
                batch_size = int(sys.argv[3])
                
            max_batches = None
            if len(sys.argv) > 4:
                max_batches = int(sys.argv[4])
                
            result = batch_process_gutenberg.remote(file_pattern, batch_size, max_batches)
            
            logging.info(f"Batch processing complete:")
            logging.info(f"  Job ID: {result['job_id']}")
            logging.info(f"  Processed {result['processed_texts']} texts from {result['processed_files']} files")
            logging.info(f"  Found {result['texts_with_actions']} texts with actions")
            logging.info(f"  Extracted {result['total_actions']} total actions")
            logging.info(f"  Status: {result['status']}")
            logging.info("\nTo download the results, run:")
            logging.info(f"  python modal_setup.py download-results --job-id {result['job_id']}")
            logging.info("Or to get just the actions (smaller file):")
            logging.info(f"  python modal_setup.py download-results --job-id {result['job_id']} --dataset-type actions")
            logging.info("Or to get the graph dataset with original text:")
            logging.info(f"  python modal_setup.py download-results --job-id {result['job_id']} --dataset-type graph")
            
            return
        
        if command == "provider":
            # Test with different providers
            provider = "openai"  # default
            if len(sys.argv) > 2:
                provider = sys.argv[2].lower()
                if provider not in ["openai", "anthropic", "groq"]:
                    logging.error(f"Unknown provider: {provider}. Using OpenAI instead.")
                    provider = "openai"
            
            sample_story = """
            Once upon a time, in a dense forest, Character 1 discovered a hidden cave behind a waterfall. 
            Character 1 cautiously entered the dark cave, lighting a torch to see ahead. 
            Inside, Character 1 found a small wooden chest covered in strange symbols.
            Character 1 opened the chest carefully, revealing a glowing blue crystal inside.
            """
            
            logging.info(f"Testing with {provider.upper()} provider...")
            actions = process_story.remote(sample_story, provider=provider)
            
            logging.info(f"Extracted {len(actions)} actions using {provider.upper()} provider:")
            for i, action in enumerate(actions, 1):
                logging.info(f"\nAction {i}:")
                logging.info(f"  Source: {action.get('source', 'Unknown')} ({action.get('source_type', 'Unknown type')})")
                logging.info(f"  Action: {action.get('action', 'Unknown action')}")
                logging.info(f"  Consequence: {action.get('consequence', 'Unknown consequence')}")
            
            return
    
    # Default: Run a sample story for demo purposes
    logging.info("Running sample story processing demo")
    
    # Sample story to process
    sample_story = """
    Once upon a time, in a dense forest, Character 1 discovered a hidden cave behind a waterfall. 
    Character 1 cautiously entered the dark cave, lighting a torch to see ahead. 
    Inside, Character 1 found a small wooden chest covered in strange symbols.
    Character 1 opened the chest carefully, revealing a glowing blue crystal inside.
    As soon as Character 1 touched the crystal, a blinding light filled the cave.
    When Character 1 regained sight, Character 2 was standing before them, dressed in ancient robes.
    "You have awakened me from my slumber," Character 2 said, bowing slightly.
    Character 2 explained the crystal's power to Character 1, warning of its dangers.
    Despite the warning, Character 1 decided to keep the crystal, placing it in their pocket.
    Character 2 frowned but offered to guide Character 1 back to the forest.
    Together, they left the cave and headed toward the nearest village.
    """
    
    # Process the sample story
    actions = process_story.remote(sample_story)
    
    # Display results
    logging.info(f"Extracted {len(actions)} actions from the sample story:")
    for i, action in enumerate(actions, 1):
        logging.info(f"\nAction {i}:")
        logging.info(f"  Source: {action.get('source', 'Unknown')} ({action.get('source_type', 'Unknown type')})")
        logging.info(f"  Action: {action.get('action', 'Unknown action')}")
        logging.info(f"  Consequence: {action.get('consequence', 'Unknown consequence')}")
        if action.get('target'):
            logging.info(f"  Target: {action.get('target')} ({action.get('target_type', 'Unknown type')})")
        if action.get('location'):
            logging.info(f"  Location: {action.get('location')}")
    
    logging.info("\nTo test with different LLM providers:")
    logging.info("  python -m examples.lit_agents provider openai")
    logging.info("  python -m examples.lit_agents provider anthropic")
    logging.info("  python -m examples.lit_agents provider groq")
    
    logging.info("\nTo run batch processing on the Gutenberg dataset:")
    logging.info("  python -m examples.lit_agents batch [file_pattern] [batch_size] [max_batches]")
    logging.info("\nExamples:")
    logging.info("  python -m examples.lit_agents batch gutenberg_sample.parquet 5")  
    logging.info("  python -m examples.lit_agents batch \"gutenberg_chunkingprocessed_en-*.parquet\" 20 5")

###########################################
# Gutenberg Dataset Processing Functions
###########################################

@app.function(
    image=image,
    secrets=[modal.Secret.from_dict(env_vars)],
    volumes={
        get_modal_cache_dir(): cache_vol,
        get_modal_results_dir(): results_vol,
        get_data_dir(): data_vol
    }
)
def process_gutenberg_text(file_path: str, start_idx: int = 0, end_idx: Optional[int] = None):
    """
    Process text from the Gutenberg dataset using polars.
    
    Args:
        file_path: Path to the parquet file
        start_idx: Starting row index
        end_idx: Ending row index (None = process all)
        
    Returns:
        tuple: (List of extracted actions, Original dataframe with rows processed)
    """
    import polars as pl
    
    # Load data
    try:
        # Handle both local and Modal paths
        if file_path.startswith(get_data_dir()):
            actual_path = file_path
        else:
            actual_path = os.path.join(get_data_dir(), os.path.basename(file_path))
        
        # Read parquet file with polars
        df = pl.read_parquet(actual_path)
        
        # Select subset of data if specified
        if end_idx is not None:
            df_subset = df.slice(start_idx, end_idx - start_idx)
        else:
            df_subset = df.slice(start_idx)
        
        # Process each text segment
        all_actions = []
        processed_indices = []
        
        for i, row in enumerate(df_subset.iter_rows(named=True)):
            # Assuming the text column is named 'text'
            text_column = 'text' if 'text' in df_subset.columns else df_subset.columns[0]
            story_text = row[text_column]
            
            # Skip empty texts
            if not story_text or len(story_text.strip()) < 10:
                continue
                
            # Process the text
            actions = process_story.remote(story_text)
            
            # Add source information to each action
            for action in actions:
                # Store the row index for later joining
                action['original_row_idx'] = start_idx + i
                # Include a snippet of the source text (first 100 chars)
                action['text_snippet'] = story_text[:100] + "..." if len(story_text) > 100 else story_text
                # Add source file information
                action['source_file'] = os.path.basename(file_path)
                
            all_actions.extend(actions)
            processed_indices.append(start_idx + i)
            
        return all_actions, df_subset, processed_indices
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return [], None, []

@app.function(
    image=image,
    secrets=[modal.Secret.from_dict(env_vars)],
    volumes={
        get_modal_cache_dir(): cache_vol,
        get_modal_results_dir(): results_vol,
        get_data_dir(): data_vol
    }
)
def batch_process_gutenberg(file_pattern: str = "gutenberg_chunkingprocessed_en-*.parquet", batch_size: int = 10, max_batches: Optional[int] = None):
    """
    Process multiple Gutenberg files in batches using polars.
    
    Args:
        file_pattern: Pattern to match files (e.g. "gutenberg_chunkingprocessed_en-*.parquet")
        batch_size: Number of rows to process per batch
        max_batches: Maximum number of batches to process (None = process all)
        
    Returns:
        dict: Summary of processing results
    """
    import glob
    import polars as pl
    
    # Create progress tracking file
    progress_file = os.path.join(get_data_dir(), "processing_progress.json")
    job_id = f"job_{int(time.time())}"
    
    # Initialize progress data
    progress_data = {
        "job_id": job_id,
        "start_time": datetime.now().isoformat(),
        "status": "running",
        "processed_files": 0,
        "total_files": 0,
        "total_actions": 0,
        "current_file": "",
        "errors": [],
        "last_update": datetime.now().isoformat()
    }
    
    # Write initial progress
    def update_progress(data):
        data["last_update"] = datetime.now().isoformat()
        with open(progress_file, "w") as f:
            json.dump(data, f, indent=2)
        # Also print to logs for real-time monitoring
        print(f"[PROGRESS] {data['processed_files']}/{data['total_files']} files | {data['total_actions']} actions | Status: {data['status']}")
    
    # Find matching files
    data_dir = get_data_dir()
    file_paths = glob.glob(os.path.join(data_dir, file_pattern))
    
    # Update progress with total files
    progress_data["total_files"] = len(file_paths)
    update_progress(progress_data)
    
    print(f"Found {len(file_paths)} files matching pattern: {file_pattern}")
    
    all_actions = []
    all_original_data = []
    
    # Process each file in batches
    for i, file_path in enumerate(file_paths):
        if max_batches and i >= max_batches:
            break
        
        # Update current file in progress
        progress_data["current_file"] = os.path.basename(file_path)
        update_progress(progress_data)
        
        try:
            print(f"Processing file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
            
            # Process the file
            start_time = time.time()
            actions, original_df, processed_indices = process_gutenberg_text.remote(file_path, 0, batch_size)
            processing_time = time.time() - start_time
            
            # Update progress
            progress_data["processed_files"] += 1
            progress_data["total_actions"] += len(actions)
            print(f"✅ Processed {len(actions)} actions in {processing_time:.2f}s from {os.path.basename(file_path)}")
            update_progress(progress_data)
            
            all_actions.extend(actions)
            
            # Store original data if available
            if original_df is not None and not original_df.is_empty():
                if processed_indices:
                    # Add a column to identify which rows were processed
                    row_indices = pl.Series("row_idx", list(range(len(original_df))))
                    is_processed = row_indices.is_in(processed_indices)
                    
                    # Add columns to identify the file and job
                    original_df = original_df.with_columns([
                        pl.lit(os.path.basename(file_path)).alias("source_file"),
                        pl.lit(job_id).alias("job_id"),
                        is_processed.alias("is_processed")
                    ])
                    all_original_data.append(original_df)
            
        except Exception as e:
            error_msg = f"Error processing file {file_path}: {str(e)}"
            progress_data["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            update_progress(progress_data)
    
    # Combine all original data
    combined_original_data = None
    if all_original_data:
        try:
            combined_original_data = pl.concat(all_original_data)
            print(f"✅ Combined {len(all_original_data)} original data frames with {len(combined_original_data)} total rows")
        except Exception as e:
            print(f"❌ Error combining original data: {str(e)}")
    
    # Convert actions to DataFrame
    actions_df = None
    if all_actions:
        try:
            actions_df = pl.DataFrame(all_actions)
            print(f"✅ Created actions DataFrame with {len(actions_df)} rows")
        except Exception as e:
            print(f"❌ Error creating actions DataFrame: {str(e)}")
    
    # Create combined results file
    results = {
        "job_id": job_id,
        "total_files": len(file_paths),
        "processed_files": progress_data["processed_files"],
        "total_actions": progress_data["total_actions"],
        "errors": progress_data["errors"]
    }
    
    # Save as integrated dataset
    output_file = os.path.join(get_data_dir(), f"gutenberg_actions_{job_id}.parquet")
    
    if actions_df is not None and combined_original_data is not None:
        try:
            # Create a mapping table between actions and original data
            action_mapping = actions_df.select(
                ["original_row_idx", "source_file"]
            ).unique()
            
            # Add a column to mark rows that have associated actions
            has_actions = combined_original_data.select(
                pl.col("row_idx").is_in(action_mapping.select("original_row_idx")).alias("has_actions")
            )
            
            # Add this column to the combined data
            enriched_data = combined_original_data.with_column(
                has_actions
            )
            
            # Save integrated dataset
            enriched_data.write_parquet(output_file)
            results["output_file"] = output_file
            results["output_rows"] = len(enriched_data)
            print(f"✅ Saved integrated dataset with {len(enriched_data)} rows to {output_file}")
            
            # Also save just the actions as a separate file for easier analysis
            actions_file = os.path.join(get_data_dir(), f"actions_only_{job_id}.parquet")
            actions_df.write_parquet(actions_file)
            results["actions_file"] = actions_file
            print(f"✅ Saved {len(actions_df)} actions to {actions_file}")
            
            # Create a graph dataset that contains both actions and text in a denormalized format
            try:
                # Create a mapping dictionary for original texts
                text_dict = {}
                for row in combined_original_data.filter(pl.col("has_actions")).iter_rows(named=True):
                    # Create a key from source_file and row_idx
                    key = (row["source_file"], row["row_idx"])
                    # Store text and metadata
                    text_dict[key] = {
                        "text": row.get("text", ""),
                        "id": row.get("id", ""),
                        "title": row.get("title", ""),
                        "author": row.get("author", ""),
                        "source_file": row["source_file"],
                        "job_id": job_id
                    }
                
                # Add the original text to each action
                graph_records = []
                for action in actions_df.iter_rows(named=True):
                    key = (action["source_file"], action["original_row_idx"])
                    if key in text_dict:
                        # Combine action with its original text
                        graph_record = {**action}  # Copy action data
                        # Add text data 
                        graph_record["full_text"] = text_dict[key]["text"]
                        graph_record["text_id"] = text_dict[key].get("id", "")
                        graph_record["text_title"] = text_dict[key].get("title", "")
                        graph_record["text_author"] = text_dict[key].get("author", "")
                        graph_records.append(graph_record)
                
                # Create and save graph dataset
                if graph_records:
                    graph_df = pl.DataFrame(graph_records)
                    graph_file = os.path.join(get_data_dir(), f"graph_dataset_{job_id}.parquet")
                    graph_df.write_parquet(graph_file)
                    results["graph_file"] = graph_file
                    print(f"✅ Saved graph dataset with {len(graph_df)} rows to {graph_file}")
                else:
                    print("⚠️ No graph records created - could not link actions to texts")
            except Exception as e:
                error_msg = f"Error creating graph dataset: {str(e)}"
                progress_data["errors"].append(error_msg)
                print(f"❌ {error_msg}")
            
        except Exception as e:
            error_msg = f"Error saving integrated dataset: {str(e)}"
            progress_data["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            
            # Fall back to saving just the actions
            if actions_df is not None:
                actions_df.write_parquet(output_file)
                results["output_file"] = output_file
                print(f"✅ Fallback: Saved {len(actions_df)} actions to {output_file}")
    elif actions_df is not None:
        # If we have only actions but no original data
        actions_df.write_parquet(output_file)
        results["output_file"] = output_file
        print(f"✅ Saved {len(actions_df)} actions to {output_file}")
    elif combined_original_data is not None:
        # If we have only original data but no actions
        combined_original_data.write_parquet(output_file)
        results["output_file"] = output_file
        print(f"✅ Saved {len(combined_original_data)} original rows to {output_file}")
    
    # Mark job as completed
    progress_data["status"] = "completed"
    progress_data["end_time"] = datetime.now().isoformat()
    update_progress(progress_data)
    
    return results