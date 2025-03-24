"""
Modal Setup for Gutenberg Dataset Processing
============================================

This script provides a comprehensive interface for processing the Gutenberg dataset 
using Modal's cloud infrastructure to extract structured actions from narrative text.

Features:
1. Modal configuration and environment setup
2. Volume management (create, list, upload, download)
3. Dataset preparation and processing
4. Real-time progress monitoring
5. Results management and analysis helpers

Date: March 2025
Version: 1.0.0
"""

import os
import glob
import json
import time
import modal
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

###########################################
# CONFIGURATION
###########################################

# Volume names for persistent storage
CACHE_VOL_NAME = "inference-cache-vol"
RESULTS_VOL_NAME = "results-vol" 
DATA_VOL_NAME = "gutenberg-data-vol"

# Mount paths in Modal environment
CACHE_DIR = "/cache"
RESULTS_DIR = "/results"
DATA_DIR = "/data"

# Create Modal application
app = modal.App("gutenberg-processor")

# Define Modal volumes
cache_vol = modal.Volume.from_name(CACHE_VOL_NAME)
results_vol = modal.Volume.from_name(RESULTS_VOL_NAME)
data_vol = modal.Volume.from_name(DATA_VOL_NAME)

# Define Modal image with all required dependencies
image = (modal.Image.debian_slim()
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
        
        # CLI utilities
        "typer>=0.9.0",         # Command-line interface builder
        "rich>=13.4.2"          # Rich text and formatting in the terminal
    )
    # Add local source code to the image
    .add_local_python_source("ecs", "threads")
)

###########################################
# VOLUME MANAGEMENT
###########################################

def create_volumes() -> None:
    """
    Create all required Modal volumes if they don't exist.
    
    This function attempts to create each volume and gracefully handles
    the case where volumes already exist.
    """
    for vol_name in [CACHE_VOL_NAME, RESULTS_VOL_NAME, DATA_VOL_NAME]:
        try:
            subprocess.run(["modal", "volume", "create", vol_name], check=True)
            print(f"✅ Created volume: {vol_name}")
        except subprocess.CalledProcessError:
            print(f"ℹ️ Volume already exists: {vol_name}")

def list_volume_contents(volume_name: str, path: str = "/") -> List[str]:
    """
    List contents of a Modal volume.
    
    Args:
        volume_name: Name of the Modal volume
        path: Path within the volume to list
        
    Returns:
        List of file/directory names in the specified path
    """
    result = subprocess.run(
        ["modal", "volume", "ls", volume_name, path],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip().split("\n")

def upload_to_volume(volume_name: str, local_path: str, remote_path: str) -> None:
    """
    Upload data to a Modal volume.
    
    Args:
        volume_name: Name of the Modal volume
        local_path: Local file or directory to upload
        remote_path: Destination path within the volume
    """
    subprocess.run(
        ["modal", "volume", "put", volume_name, local_path, remote_path],
        check=True
    )
    print(f"✅ Uploaded {local_path} to {volume_name}:{remote_path}")

def download_from_volume(volume_name: str, remote_path: str, local_path: str) -> None:
    """
    Download data from a Modal volume.
    
    Args:
        volume_name: Name of the Modal volume
        remote_path: Source path within the volume
        local_path: Local destination path
    """
    subprocess.run(
        ["modal", "volume", "get", volume_name, remote_path, local_path],
        check=True
    )
    print(f"✅ Downloaded {volume_name}:{remote_path} to {local_path}")

###########################################
# DATASET PREPARATION
###########################################

def download_gutenberg_sample(output_dir: str = "gutenberg_data") -> str:
    """
    Download a sample Gutenberg dataset file for testing.
    
    Args:
        output_dir: Directory to save the downloaded file
        
    Returns:
        Path to the downloaded sample file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    sample_file = os.path.join(output_dir, "gutenberg_sample.parquet")
    
    # Download a sample file from Hugging Face
    sample_url = "https://huggingface.co/datasets/Pclanglais/gutenberg_set/resolve/main/gutenberg_chunkingprocessed_en-00000-of-00052-7cda8f63c262acf8.parquet"
    
    subprocess.run(["curl", "-L", sample_url, "-o", sample_file], check=True)
    print(f"✅ Downloaded sample file to {sample_file}")
    
    return sample_file

def upload_gutenberg_dataset(local_dir: str = "gutenberg_data") -> None:
    """
    Upload Gutenberg dataset to Modal volume.
    
    Args:
        local_dir: Local directory containing dataset files
    
    Raises:
        FileNotFoundError: If the local directory doesn't exist
    """
    # Ensure local directory exists
    if not os.path.exists(local_dir):
        raise FileNotFoundError(f"Local directory {local_dir} not found")
    
    # Upload to Modal volume
    upload_to_volume(DATA_VOL_NAME, local_dir, DATA_DIR)

###########################################
# DEPLOYMENT FUNCTIONS
###########################################

def deploy_app() -> None:
    """
    Deploy the main application to Modal.
    
    This makes the application available for remote execution.
    """
    subprocess.run(["modal", "deploy", "examples/lit_agents.py"], check=True)
    print("✅ Deployed application to Modal")

def run_test_batch(file_pattern: str = "gutenberg_sample.parquet", batch_size: int = 5) -> None:
    """
    Run a test batch to verify the setup.
    
    Args:
        file_pattern: Pattern to match dataset files
        batch_size: Number of rows to process per batch
    """
    subprocess.run([
        "modal", "run", "examples/lit_agents.py::batch_process_gutenberg",
        "--file-pattern", file_pattern,
        "--batch-size", str(batch_size),
        "--max-batches", "1"
    ], check=True)
    print("✅ Test batch completed")

def start_full_processing(
    file_pattern: str = "gutenberg_chunkingprocessed_en-*.parquet",
    batch_size: int = 20,
    max_batches: Optional[int] = None
) -> None:
    """
    Start full processing of the Gutenberg dataset.
    
    Args:
        file_pattern: Pattern to match dataset files
        batch_size: Number of rows to process per batch
        max_batches: Maximum number of batches to process (None = process all)
    """
    cmd = [
        "modal", "run", "examples/lit_agents.py::batch_process_gutenberg",
        "--file-pattern", file_pattern,
        "--batch-size", str(batch_size)
    ]
    
    if max_batches is not None:
        cmd.extend(["--max-batches", str(max_batches)])
    
    subprocess.run(cmd, check=True)
    print(f"✅ Started processing with pattern: {file_pattern}")

def check_app_status() -> None:
    """
    Check the status of the Modal application.
    
    Displays a list of all Modal apps and their status.
    """
    subprocess.run(["modal", "app", "list"], check=True)

def view_app_logs() -> None:
    """
    View logs from the Modal application.
    
    Shows real-time logs from the lit-agents application.
    """
    subprocess.run(["modal", "app", "logs", "lit-agents"], check=True)

###########################################
# PROGRESS TRACKING
###########################################

def get_progress_data() -> Dict[str, Any]:
    """
    Get the latest progress data from the Modal volume.
    
    Returns:
        Dictionary containing progress information:
        - status: Current status (running, completed, etc.)
        - processed_files: Number of files processed
        - total_files: Total number of files to process
        - total_actions: Number of actions extracted
        - errors: List of error messages
    """
    try:
        # Create a temporary file to store the progress data
        temp_file = "temp_progress.json"
        
        # Download the progress file from Modal
        download_from_volume(DATA_VOL_NAME, f"{DATA_DIR}/processing_progress.json", temp_file)
        
        # Read and parse the progress data
        with open(temp_file, "r") as f:
            progress_data = json.load(f)
        
        # Clean up
        os.remove(temp_file)
        
        return progress_data
    except Exception as e:
        print(f"Error getting progress data: {e}")
        return {
            "status": "unknown",
            "processed_files": 0,
            "total_files": 0,
            "total_actions": 0,
            "errors": []
        }

def monitor_progress(interval: int = 10, live_view: bool = True):
    """
    Monitor processing progress in real-time.
    
    Args:
        interval: Seconds between progress checks
        live_view: Whether to display a continuously updating view
    """
    from rich.live import Live
    from rich.table import Table
    from rich.console import Console
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
    
    console = Console()
    stop_monitoring = threading.Event()
    
    def generate_table():
        """Generate a rich table with the latest progress information."""
        # Get latest progress data
        progress_data = get_progress_data()
        
        # Create status table
        table = Table(title="Gutenberg Processing Status")
        table.add_column("Metric")
        table.add_column("Value")
        
        # Add rows
        table.add_row("Status", progress_data.get("status", "Unknown"))
        table.add_row("Job ID", progress_data.get("job_id", "N/A"))
        table.add_row("Files Processed", f"{progress_data.get('processed_files', 0)}/{progress_data.get('total_files', 0)}")
        table.add_row("Actions Extracted", f"{progress_data.get('total_actions', 0)}")
        table.add_row("Current File", progress_data.get("current_file", "N/A"))
        table.add_row("Started", progress_data.get("start_time", "N/A"))
        table.add_row("Last Update", progress_data.get("last_update", "N/A"))
        
        # Add progress bar
        if progress_data.get("total_files", 0) > 0:
            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
            )
            task = progress.add_task("Processing", total=progress_data.get("total_files", 100))
            progress.update(task, completed=progress_data.get("processed_files", 0))
            table.add_row("Progress", progress)
        
        # Add errors if any
        errors = progress_data.get("errors", [])
        if errors:
            table.add_row("Errors", f"{len(errors)} errors")
            for i, error in enumerate(errors[:5]):  # Show only the first 5 errors
                table.add_row(f"Error {i+1}", error)
            if len(errors) > 5:
                table.add_row("", f"... {len(errors) - 5} more errors")
        
        return table
    
    # Live view mode with continuous updates
    if live_view:
        try:
            with Live(generate_table(), refresh_per_second=1) as live:
                while not stop_monitoring.is_set():
                    try:
                        live.update(generate_table())
                        
                        # Check if processing is completed
                        progress_data = get_progress_data()
                        if progress_data.get("status") == "completed":
                            console.print("\n[bold green]Processing completed![/]")
                            break
                            
                        time.sleep(interval)
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        console.print(f"[bold red]Error updating progress: {e}[/]")
                        time.sleep(interval)
        except KeyboardInterrupt:
            console.print("[yellow]Monitoring stopped.[/]")
    
    # Simple periodic check mode
    else:
        try:
            while not stop_monitoring.is_set():
                console.print(generate_table())
                
                # Check if processing is completed
                progress_data = get_progress_data()
                if progress_data.get("status") == "completed":
                    console.print("\n[bold green]Processing completed![/]")
                    break
                    
                time.sleep(interval)
        except KeyboardInterrupt:
            console.print("[yellow]Monitoring stopped.[/]")

###########################################
# RESULTS MANAGEMENT
###########################################

def download_job_results(job_id: str = None, output_dir: str = "results", dataset_type: str = "integrated"):
    """
    Download results from a specific job or the latest job.
    
    Args:
        job_id: Specific job ID to download (None = latest job)
        output_dir: Directory to save results
        dataset_type: Type of dataset to download ("integrated", "actions", or "graph")
    """
    from rich.console import Console
    console = Console()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # If no job ID provided, get the latest from progress file
    if not job_id:
        progress_data = get_progress_data()
        job_id = progress_data.get("job_id")
        if not job_id:
            console.print("[bold red]No job ID found in progress data[/]")
            return
    
    # Find the results files for this job
    integrated_file = f"gutenberg_actions_{job_id}.parquet"
    actions_file = f"actions_only_{job_id}.parquet"
    graph_file = f"graph_dataset_{job_id}.parquet"
    
    # Check which files exist
    try:
        contents = list_volume_contents(DATA_VOL_NAME, DATA_DIR)
        actions_exist = actions_file in contents
        integrated_exists = integrated_file in contents
        graph_exists = graph_file in contents
        
        if not actions_exist and not integrated_exists and not graph_exists:
            console.print(f"[bold red]No result files found for job {job_id}[/]")
            return
            
    except Exception as e:
        console.print(f"[bold red]Error checking for results: {e}[/]")
        # Continue anyway, we'll try to download the file
    
    # Determine which file to download based on dataset_type
    if dataset_type == "actions" and actions_exist:
        target_file = actions_file
    elif dataset_type == "graph" and graph_exists:
        target_file = graph_file
    elif dataset_type == "integrated" and integrated_exists:
        target_file = integrated_file
    else:
        # Fall back to whatever is available
        if integrated_exists:
            target_file = integrated_file
            dataset_type = "integrated"
        elif actions_exist:
            target_file = actions_file
            dataset_type = "actions"
        elif graph_exists:
            target_file = graph_file
            dataset_type = "graph"
        else:
            console.print(f"[bold red]No requested dataset type available for job {job_id}[/]")
            return
    
    local_file = os.path.join(output_dir, target_file)
    
    # Download the results
    try:
        download_from_volume(DATA_VOL_NAME, f"{DATA_DIR}/{target_file}", local_file)
        
        console.print(f"[bold green]Downloaded {dataset_type} dataset to {local_file}[/]")
        
        # Provide appropriate guidance based on dataset type
        console.print(f"\n[bold blue]Quick tips for working with the {dataset_type} dataset:[/]")
        console.print(f"  • Load in Python: [yellow]import polars as pl; df = pl.read_parquet('{local_file}')[/]")
        
        if dataset_type == "actions":
            console.print(f"  • View action stats: [yellow]print(f'Actions: {{len(df)}}, Sources: {{df['source_file'].n_unique()}}')[/]")
            console.print(f"  • View action types: [yellow]df.group_by('action_type').agg(pl.count().alias('count')).sort('count', descending=True)[/]")
            console.print(f"  • Get text snippets: [yellow]df.select(['text_snippet', 'action_type'])[/]")
        
        elif dataset_type == "integrated":
            console.print(f"  • View dataset stats: [yellow]print(f'Rows: {{len(df)}}, With actions: {{df.filter(pl.col(\"has_actions\")).height}}')[/]")
            console.print(f"  • Get original text: [yellow]text_data = df.select(['text', 'has_actions'])[/]")
            console.print(f"  • Filter to rows with actions: [yellow]df_with_actions = df.filter(pl.col('has_actions'))[/]")
        
        elif dataset_type == "graph":
            console.print(f"  • View entity relationships: [yellow]print(f'Actions: {{len(df)}}, Unique texts: {{df['text_id'].n_unique()}}')[/]")
            console.print(f"  • Analyze action contexts: [yellow]df.select(['source', 'target', 'action', 'full_text', 'text_title'])[/]")
            console.print(f"  • Filter by action type: [yellow]df.filter(pl.col('action_type') == 'physical')[/]")
        
        console.print(f"  • Export CSV: [yellow]df.write_csv('{os.path.splitext(local_file)[0]}.csv')[/]")
        
        # Inform about other available formats
        available_formats = []
        if integrated_exists and dataset_type != "integrated":
            available_formats.append(("integrated", "integrated dataset with original text"))
        if actions_exist and dataset_type != "actions":
            available_formats.append(("actions", "actions-only dataset"))
        if graph_exists and dataset_type != "graph":
            available_formats.append(("graph", "graph dataset with relationships"))
        
        if available_formats:
            console.print(f"\n[bold yellow]Note:[/] Other dataset formats are also available for this job:")
            for format_type, format_desc in available_formats:
                console.print(f"  • Download the {format_desc}: [yellow]python modal_setup.py download-results --job-id {job_id} --dataset-type {format_type}[/]")
        
    except Exception as e:
        console.print(f"[bold red]Error downloading results: {e}[/]")

###########################################
# COMPLETE SETUP PROCESS
###########################################

def setup_everything() -> None:
    """
    Run the complete setup process.
    
    This function:
    1. Creates all necessary Modal volumes
    2. Downloads and uploads a sample dataset
    3. Deploys the application
    4. Runs a test batch to verify everything works
    """
    # 1. Create volumes
    create_volumes()
    
    # 2. Download and upload sample data
    sample_file = download_gutenberg_sample()
    upload_gutenberg_dataset("gutenberg_data")
    
    # 3. Deploy the application
    deploy_app()
    
    # 4. Run a test batch
    run_test_batch()
    
    print("\n✅ Setup completed! You can now start processing with start_full_processing()")
    print("\n📋 Next steps:")
    print("  1. Start processing: python modal_setup.py process")
    print("  2. Monitor progress: python modal_setup.py monitor")
    print("  3. Download results: python modal_setup.py download-results")
    print("  4. Download specific job: python modal_setup.py download-results --job-id <job_id>")

###########################################
# CLI INTERFACE
###########################################

if __name__ == "__main__":
    import typer
    from rich.console import Console
    from typing_extensions import Annotated
    
    console = Console()
    app = typer.Typer(help="Modal setup and management for Gutenberg processing")
    
    @app.command()
    def setup():
        """Run the complete setup process."""
        console.print("[bold green]Starting complete setup process...[/]")
        setup_everything()
    
    @app.command()
    def create_vols():
        """Create Modal volumes."""
        console.print("[bold green]Creating Modal volumes...[/]")
        create_volumes()
    
    @app.command()
    def download_sample():
        """Download a sample dataset file."""
        console.print("[bold green]Downloading sample dataset...[/]")
        download_gutenberg_sample()
    
    @app.command()
    def upload_data(local_dir: str = "gutenberg_data"):
        """Upload dataset to Modal volume."""
        console.print(f"[bold green]Uploading data from {local_dir}...[/]")
        upload_gutenberg_dataset(local_dir)
    
    @app.command()
    def deploy():
        """Deploy the application to Modal."""
        console.print("[bold green]Deploying application...[/]")
        deploy_app()
    
    @app.command()
    def test(file_pattern: str = "gutenberg_sample.parquet", batch_size: int = 5):
        """Run a test batch."""
        console.print(f"[bold green]Running test batch with {file_pattern}...[/]")
        run_test_batch(file_pattern, batch_size)
    
    @app.command()
    def process(
        file_pattern: str = "gutenberg_chunkingprocessed_en-*.parquet",
        batch_size: int = 20,
        max_batches: Optional[int] = None
    ):
        """Start full processing."""
        console.print(f"[bold green]Starting full processing with {file_pattern}...[/]")
        start_full_processing(file_pattern, batch_size, max_batches)
    
    @app.command()
    def status():
        """Check application status."""
        console.print("[bold green]Checking application status...[/]")
        check_app_status()
    
    @app.command()
    def logs():
        """View application logs."""
        console.print("[bold green]Viewing application logs...[/]")
        view_app_logs()
    
    @app.command()
    def monitor(interval: int = 10, live: bool = True):
        """Monitor processing progress in real-time."""
        console.print("[bold green]Monitoring processing progress...[/]")
        monitor_progress(interval, live)
    
    @app.command()
    def download_results(
        job_id: Annotated[Optional[str], typer.Option(help="Specific job ID to download")] = None, 
        output_dir: Annotated[str, typer.Option(help="Directory to save results")] = "results",
        dataset_type: Annotated[str, typer.Option(help="Type of dataset to download (integrated, actions, or graph)")] = "integrated"
    ):
        """
        Download processing results from Modal.
        
        This downloads one of three dataset types:
        - integrated: Original text with flags indicating rows that have actions (default)
        - actions: Just the extracted actions (smaller file)
        - graph: Denormalized format with actions linked to their original text (higher dimensional)
        
        If no job_id is specified, downloads results from the latest job.
        Results are saved as parquet files that can be loaded with polars.
        """
        if job_id:
            console.print(f"[bold green]Downloading results for job {job_id}...[/]")
        else:
            console.print("[bold green]Downloading results for latest job...[/]")
        
        console.print(f"[bold green]Downloading {dataset_type} dataset to {output_dir}[/]")
        
        download_job_results(job_id, output_dir, dataset_type)
    
    @app.command(name="list-jobs")
    def list_jobs():
        """List all available processing jobs with their IDs."""
        console.print("[bold green]Listing available processing jobs...[/]")
        try:
            progress_data = get_progress_data()
            job_id = progress_data.get("job_id")
            if job_id:
                console.print(f"[bold blue]Latest job:[/] {job_id}")
                console.print(f"  Status: {progress_data.get('status', 'unknown')}")
                console.print(f"  Files processed: {progress_data.get('processed_files', 0)}/{progress_data.get('total_files', 0)}")
                console.print(f"  Actions extracted: {progress_data.get('total_actions', 0)}")
                console.print(f"  Started: {progress_data.get('start_time', 'N/A')}")
                
                # List other job files
                console.print("\n[bold blue]Searching for other job results...[/]")
                try:
                    # Create a temporary file to list volume contents
                    result = subprocess.run(
                        ["modal", "volume", "ls", DATA_VOL_NAME, DATA_DIR],
                        capture_output=True,
                        text=True,
                    )
                    files = result.stdout.strip().split("\n")
                    job_files = [f for f in files if f.startswith("gutenberg_actions_job_")]
                    
                    if job_files:
                        console.print("[bold blue]Available job results:[/]")
                        for file in job_files:
                            job_id = file.replace("gutenberg_actions_", "").replace(".parquet", "")
                            console.print(f"  • {job_id}")
                        console.print("\n[bold green]To download a specific job:[/]")
                        console.print(f"  python modal_setup.py download-results --job-id JOB_ID")
                    else:
                        console.print("[yellow]No job result files found[/]")
                except Exception as e:
                    console.print(f"[bold red]Error listing jobs: {e}[/]")
            else:
                console.print("[yellow]No jobs found in progress data[/]")
        except Exception as e:
            console.print(f"[bold red]Error getting jobs: {e}[/]")
    
    app() 