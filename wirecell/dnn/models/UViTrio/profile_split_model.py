#!/usr/bin/env python
"""
Profiling script for ViTUNetCrossView with splitting policy.

Tests on input shape (1, 1, 2560, 1500) and uses torch profiler to analyze performance.
Supports both CPU and GPU execution with arbitrary input dimensions.

Architecture based on U-Net paper (Ronneberger et al.):
- 5-level network with max pooling downsampling
- Bilinear upsampling
- Batch normalization
- Automatic decoder padding for non-power-of-2 dimensions

Usage:
    python profile_split_model.py                    # Run on CPU (default shape)
    python profile_split_model.py --gpu              # Run on GPU
    python profile_split_model.py --gpu --device 0   # Run on specific GPU
    python profile_split_model.py --input-shape 1,2560,1536  # Custom shape (any W works!)
"""

import argparse
import sys
import torch
from torch.profiler import profile, record_function, ProfilerActivity

sys.path.insert(0, '.')

from uvcgan2.models.generator.vitunet_crossview import ViTUNetCrossView


def create_model(device, input_shape=(1, 2560, 1500), split_sizes=[800, 800, 960]):
    """
    Create ViTUNetCrossView model with paper-based architecture.

    Architecture based on O. Ronneberger et al. U-Net paper:
    - 5-level network
    - 2x2 max pooling for downsampling
    - Bilinear upsampling
    - Batch normalization
    - Two 3x3 convolutions per level
    """
    print("Creating model...")
    print(f"Input shape: {input_shape}")
    print(f"Split sizes: {split_sizes}")

    _, h, w = input_shape
    if w % 32 != 0:
        print(f"Note: W dimension ({w}) is not divisible by 32.")
        print(f"  Decoder will automatically pad mismatches from max pooling.")

    model = ViTUNetCrossView(
        features=64,  # Reduced from 128
        n_heads=4,
        n_blocks=2,
        ffn_features=128,  # Reduced from 256
        embed_features=64,
        activ='gelu',
        norm='layer',
        input_shape=input_shape,
        output_shape=input_shape,
        unet_features_list=[32, 64, 128, 256, 512],  # Reduced from [64, 128, 256, 512, 1024]
        unet_activ='relu',
        unet_norm='batch',  # Batch normalization per paper
        unet_downsample='maxpool',  # 2x2 max pooling per paper
        unet_upsample={'name': 'upsample', 'mode': 'bilinear'},  # Bilinear upsample per paper
        split_sizes=split_sizes,
        split_dim=2,  # H dimension
        activ_output='tanh'
    )

    model = model.to(device)
    model.eval()

    print(f"Model created on device: {device}")
    print(f"Number of views: {model.num_views}")
    print(f"Split sizes: {model.split_sizes}")
    print(f"Split dimension: {model.split_dim}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def profile_model(model, device, use_cuda, input_shape):
    """Profile the model using torch profiler"""
    print("\nPreparing input...")
    # input_shape is (C, H, W), we need (N, C, H, W)
    x = torch.randn(1, *input_shape, device=device)
    print(f"Input shape: {x.shape}")
    print(f"Input device: {x.device}")
    print(f"Input size in memory: {x.element_size() * x.nelement() / 1024**2:.2f} MB")

    # Warmup runs
    print("\nWarming up (3 iterations)...")
    with torch.no_grad():
        for i in range(3):
            _ = model(x)
            if use_cuda:
                torch.cuda.synchronize()
            print(f"  Warmup {i+1}/3 complete")

    # Profile
    print("\nStarting profiling...")
    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                y = model(x)
                if use_cuda:
                    torch.cuda.synchronize()

    print(f"Output shape: {y.shape}")
    print(f"Output device: {y.device}")

    return prof, y


def print_profiler_stats(prof, use_cuda):
    """Print profiler statistics"""
    print("\n" + "="*80)
    print("PROFILER RESULTS")
    print("="*80)

    # CPU time stats
    print("\nTop 10 operations by CPU time:")
    print("-" * 80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=10,
        max_src_column_width=50
    ))

    if use_cuda:
        print("\nTop 10 operations by CUDA time:")
        print("-" * 80)
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=10,
            max_src_column_width=50
        ))

    # Memory stats
    print("\nTop 10 operations by memory usage:")
    print("-" * 80)
    print(prof.key_averages().table(
        sort_by="self_cpu_memory_usage",
        row_limit=10,
        max_src_column_width=50
    ))

    # Overall summary
    print("\nOverall summary:")
    print("-" * 80)
    events = prof.key_averages()
    total_cpu_time = sum([evt.cpu_time_total for evt in events])
    if use_cuda:
        total_cuda_time = sum([evt.device_time_total for evt in events if hasattr(evt, 'device_time_total')])
        total_cuda_memory = sum([evt.cuda_memory_usage for evt in events if hasattr(evt, 'cuda_memory_usage')])

    print(f"Total CPU time: {total_cpu_time / 1000:.2f} ms")
    if use_cuda:
        print(f"Total CUDA time: {total_cuda_time / 1000:.2f} ms")
        print(f"Total CUDA memory: {total_cuda_memory / 1024**2:.2f} MB")


def save_profiler_trace(prof, filename):
    """Save profiler trace for Chrome tracing"""
    print(f"\nSaving trace to {filename}...")
    prof.export_chrome_trace(filename)
    print(f"Trace saved!")


def benchmark_throughput(model, device, use_cuda, input_shape, num_iterations=10):
    """Benchmark throughput"""
    print("\n" + "="*80)
    print(f"THROUGHPUT BENCHMARK ({num_iterations} iterations)")
    print("="*80)

    x = torch.randn(1, *input_shape, device=device)

    if use_cuda:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
    else:
        import time
        with torch.no_grad():
            start = time.time()
            for _ in range(num_iterations):
                _ = model(x)
            elapsed_time = time.time() - start

    throughput = num_iterations / elapsed_time
    latency = elapsed_time / num_iterations * 1000  # Convert to ms

    print(f"Total time: {elapsed_time:.3f} seconds")
    print(f"Average latency: {latency:.2f} ms per image")
    print(f"Throughput: {throughput:.2f} images/second")

    if use_cuda:
        max_memory = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"Peak GPU memory: {max_memory:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Profile ViTUNetCrossView with splitting policy'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Run on GPU instead of CPU'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='GPU device ID (default: 0)'
    )
    parser.add_argument(
        '--trace-file',
        type=str,
        default='profile_trace.json',
        help='Output file for Chrome trace (default: profile_trace.json)'
    )
    parser.add_argument(
        '--no-benchmark',
        action='store_true',
        help='Skip throughput benchmark'
    )
    parser.add_argument(
        '--input-shape',
        type=str,
        default='1,2560,1500',
        help='Input shape as C,H,W (default: 1,2560,1500). Can be arbitrary - decoder handles size mismatches.'
    )
    parser.add_argument(
        '--split-sizes',
        type=str,
        default='800,800,960',
        help='Split sizes as comma-separated values (default: 800,800,960)'
    )

    args = parser.parse_args()

    # Setup device
    if args.gpu:
        if not torch.cuda.is_available():
            print("ERROR: CUDA is not available. Falling back to CPU.")
            device = torch.device('cpu')
            use_cuda = False
        else:
            device = torch.device(f'cuda:{args.device}')
            use_cuda = True
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Total GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        use_cuda = False
        print("Using CPU")

    print("\n" + "="*80)
    print("MODEL CONFIGURATION")
    print("="*80)

    # Parse input shape and split sizes from arguments
    input_shape = tuple(int(x) for x in args.input_shape.split(','))
    split_sizes = [int(x) for x in args.split_sizes.split(',')]

    # Validate
    if len(input_shape) != 3:
        print(f"ERROR: input_shape must have 3 values (C,H,W), got {len(input_shape)}")
        return
    if sum(split_sizes) != input_shape[1]:
        print(f"ERROR: sum of split_sizes ({sum(split_sizes)}) must equal H dimension ({input_shape[1]})")
        return

    # Create model
    model = create_model(device, input_shape, split_sizes)

    # Profile
    prof, output = profile_model(model, device, use_cuda, input_shape)

    # Print stats
    print_profiler_stats(prof, use_cuda)

    # Save trace
    save_profiler_trace(prof, args.trace_file)

    # Benchmark
    if not args.no_benchmark:
        benchmark_throughput(model, device, use_cuda, input_shape)

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)

    if use_cuda:
        print(f"\nFinal GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")


if __name__ == '__main__':
    main()
