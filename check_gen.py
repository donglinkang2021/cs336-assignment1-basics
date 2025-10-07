# import debugpy; debugpy.connect(('0.0.0.0', 5678))
import torch
import time
import tracemalloc
from pathlib import Path
from tokenizers import Tokenizer
from omegaconf import OmegaConf
from contextlib import contextmanager
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.columns import Columns
from rich.text import Text
import gc

from cs336_basics.model import TransformerLM
from cs336_basics.generate import generate, install_kv_cache
from cs336_basics.config import TrainConfig

console = Console()

@contextmanager
def measure_performance():
    """Context manager for measuring performance metrics"""
    # Clear GPU memory cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        gpu_memory_start = torch.cuda.memory_allocated()
    else:
        gpu_memory_start = 0
    
    # Start CPU memory tracking
    tracemalloc.start()
    
    # Record start time
    start_time = time.time()
    
    # Create results dictionary to store metrics
    results = {}
    
    try:
        yield results
    finally:
        # Record end time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Get CPU peak memory
        current_cpu, peak_cpu = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Get GPU peak memory
        peak_gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory_max = torch.cuda.max_memory_allocated()
            peak_gpu_memory = gpu_memory_max - gpu_memory_start
        
        # Store results in the yielded dictionary
        results.update({
            'execution_time': execution_time,
            'peak_cpu_memory_mb': peak_cpu / 1024 / 1024,
            'gpu_memory_start_mb': gpu_memory_start / 1024 / 1024 if torch.cuda.is_available() else 0,
            'gpu_memory_max_mb': gpu_memory_max / 1024 / 1024 if torch.cuda.is_available() else 0,
            'peak_gpu_memory_mb': peak_gpu_memory / 1024 / 1024 if torch.cuda.is_available() else 0,
        })

def create_metrics_table(title, metrics, max_tokens, batch_size=1):
    """Create performance metrics table"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_column("Unit", style="yellow")
    
    table.add_row("Execution Time", f"{metrics['execution_time']:.3f}", "seconds")
    table.add_row("Tokens/Second", f"{(max_tokens * batch_size)/metrics['execution_time']:.2f}", "tokens/s")
    table.add_row("Peak CPU Memory", f"{metrics['peak_cpu_memory_mb']:.2f}", "MB")
    table.add_row("GPU Memory Start", f"{metrics['gpu_memory_start_mb']:.2f}", "MB")
    table.add_row("GPU Memory Max", f"{metrics['gpu_memory_max_mb']:.2f}", "MB")
    table.add_row("Peak GPU Memory", f"{metrics['peak_gpu_memory_mb']:.2f}", "MB")
    
    return table

def create_comparison_table(metrics_std, metrics_cached, max_tokens, batch_size=1):
    """Create comparison table"""
    table = Table(title="📊 Performance Comparison", show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Standard", style="red")
    table.add_column("KV Cache", style="green")
    table.add_column("Improvement", style="yellow")
    
    # Time comparison
    speedup = metrics_std['execution_time'] / metrics_cached['execution_time']
    time_reduction = ((metrics_std['execution_time'] - metrics_cached['execution_time']) / metrics_std['execution_time']) * 100
    table.add_row(
        "Execution Time", 
        f"{metrics_std['execution_time']:.3f}s", 
        f"{metrics_cached['execution_time']:.3f}s",
        f"{speedup:.2f}x ({time_reduction:.1f}% faster)"
    )
    
    # tokens/s comparison
    std_tps = (max_tokens * batch_size) / metrics_std['execution_time']
    cached_tps = (max_tokens * batch_size) / metrics_cached['execution_time']
    table.add_row(
        "Tokens/Second",
        f"{std_tps:.2f}",
        f"{cached_tps:.2f}",
        f"{cached_tps/std_tps:.2f}x"
    )
    
    # GPU memory comparison
    mem_diff = metrics_cached['peak_gpu_memory_mb'] - metrics_std['peak_gpu_memory_mb']
    mem_percent = (mem_diff / metrics_std['peak_gpu_memory_mb']) * 100 if metrics_std['peak_gpu_memory_mb'] > 0 else 0
    table.add_row(
        "Peak GPU Memory",
        f"{metrics_std['peak_gpu_memory_mb']:.2f}MB",
        f"{metrics_cached['peak_gpu_memory_mb']:.2f}MB",
        f"{'+' if mem_diff > 0 else ''}{mem_diff:.2f}MB ({mem_percent:+.1f}%)"
    )
    
    return table

# Load model
console.print(Panel.fit("🚀 Loading Model", style="bold blue"))

# Load model configuration
# the best model of openwebtext
# ckpt_path = "outputs/multiruns/2025-10-01_18-19-21/2/ckpt_4999.pt"
# the best model of tinystories
ckpt_path = "outputs/runs/2025-10-02_05-14-38/ckpt_19999.pt"
cfg_path = Path(ckpt_path).parent / ".hydra" / "config.yaml"
cfg: TrainConfig = OmegaConf.load(cfg_path)

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
) as progress:
    task = progress.add_task("Loading model...", total=None)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerLM(**cfg.model).to(device)
    if cfg.training.is_compile:
        model = torch.compile(model)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model._orig_mod
    
    progress.update(task, description="Model loaded successfully!")

total_params = sum(p.numel() for p in model.parameters())
model.eval()
tokenizer = Tokenizer.from_file(cfg.data.tokenizer_path)

# Test parameters
batch_size = 128
max_new_tokens = 1000
temperature = 0.6
top_p = 0.95

# Create context with batch_size
context = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
install_kv_cache(model, batch_size=batch_size, total_len=cfg.model.context_length + max_new_tokens)

# Display model and test information

model_info = Table(title="📋 Model Information", show_header=False)
model_info.add_column("Property", style="cyan")
model_info.add_column("Value", style="green")
model_info.add_row("Checkpoint Path", ckpt_path)
model_info.add_row("Total Parameters", f"{total_params/1e6:.2f}M")
model_info.add_row("Device", device)
model_info.add_row("Context Length", str(cfg.model.context_length))

para_info = Table(title="⚙️ Test Parameters", show_header=False)
para_info.add_column("Parameter", style="cyan")
para_info.add_column("Value", style="green")
para_info.add_row("Batch Size", str(batch_size))
para_info.add_row("Max New Tokens", str(max_new_tokens))
para_info.add_row("Temperature", str(temperature))
para_info.add_row("Top P", str(top_p)),
        

info_tables = Columns([
    # Model Information Table
    Table.grid(model_info, padding=(0, 2)),
    # Test Parameters Table
    Table.grid(para_info, padding=(0, 2))
])

# Display information tables side by side
model_info = Table(title="📋 Model Information", show_header=False)
model_info.add_column("Property", style="cyan")
model_info.add_column("Value", style="green")
model_info.add_row("Checkpoint Path", ckpt_path)
model_info.add_row("Total Parameters", f"{total_params/1e6:.2f}M")
model_info.add_row("Device", device)
model_info.add_row("Context Length", str(cfg.model.context_length))

test_params = Table(title="⚙️ Test Parameters", show_header=False)
test_params.add_column("Parameter", style="cyan")
test_params.add_column("Value", style="green")
test_params.add_row("Batch Size", str(batch_size))
test_params.add_row("Max New Tokens", str(max_new_tokens))
test_params.add_row("Temperature", str(temperature))
test_params.add_row("Top P", str(top_p))

console.print(Columns([model_info, test_params]))

console.print("\n" + "="*70)
console.print(Panel.fit("🔥 Performance Benchmarking", style="bold red"))

# Test 1: Standard generation
console.print("\n🎯 Running Standard Generation...")
with measure_performance() as metrics_standard:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating tokens...", total=None)
        generated_standard = generate(
            model, 
            context.clone(), 
            max_new_tokens=max_new_tokens, 
            block_size=cfg.model.context_length,
            temperature=temperature,
            top_p=top_p,
        )
        progress.update(task, description="Standard generation completed!")
generated_text_standard = tokenizer.decode(generated_standard[0].tolist())

# Test 2: KV cache generation
console.print("\n⚡ Running KV Cache Generation...")
with measure_performance() as metrics_cached:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating tokens with KV cache...", total=None)
        generated_cached = generate(
            model,
            context.clone(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            use_kv_cache=True
        )
        progress.update(task, description="KV cache generation completed!")
generated_text_cached = tokenizer.decode(generated_cached[0].tolist())

# Display results
console.print("\n")
tables = Columns([
    create_metrics_table("🐌 Standard Generation", metrics_standard, max_new_tokens, batch_size),
    create_metrics_table("⚡ KV Cache Generation", metrics_cached, max_new_tokens, batch_size)
])
console.print(tables)

console.print("\n")
console.print(create_comparison_table(metrics_standard, metrics_cached, max_new_tokens, batch_size))

# Display generated text preview
console.print("\n")
console.print(Panel(
    f"Standard: {generated_text_standard[:100]}...\n\n"
    f"KV Cache: {generated_text_cached[:100]}...",
    title="📝 Generated Text Preview (First Batch)",
    style="blue"
))

# Display full KV Cache generated text
console.print("\n")
console.print(Panel(
    generated_text_cached,
    title="📖 Full Generated Text (KV Cache - First Batch)",
    style="green"
))

console.print(f"\n✅ Output shapes - Standard: {generated_standard.shape}, KV Cache: {generated_cached.shape}")
console.print(f"✅ Total tokens generated - Standard: {generated_standard.numel()}, KV Cache: {generated_cached.numel()}")