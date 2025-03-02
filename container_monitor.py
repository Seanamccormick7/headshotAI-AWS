#!/usr/bin/env python3
"""
Container Resource Monitor - For monitoring inside a Docker container
Save this file in your Docker image and run it alongside your training process
"""

import os
import time
import json
import argparse
import subprocess
import datetime
from collections import defaultdict

try:
    import torch
except ImportError:
    print("Warning: PyTorch not found. GPU memory monitoring will be limited.")
    HAS_TORCH = False
else:
    HAS_TORCH = True

def get_gpu_info():
    """Get GPU information using nvidia-smi"""
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", 
                                         "--format=csv,noheader,nounits"], 
                                        universal_newlines=True)
        
        gpu_info = []
        for line in output.strip().split("\n"):
            index, memory_used, memory_total, gpu_util = line.split(", ")
            memory_used = float(memory_used)
            memory_total = float(memory_total)
            memory_percent = (memory_used / memory_total) * 100.0
            
            gpu_info.append({
                "index": int(index),
                "memory_used_mb": memory_used,
                "memory_total_mb": memory_total,
                "utilization_percent": float(gpu_util),
                "memory_percent": memory_percent
            })
        return gpu_info
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

def get_torch_gpu_info():
    """Get GPU memory info using PyTorch (more accurate for container's actual usage)"""
    if not HAS_TORCH or not torch.cuda.is_available():
        return []
    
    gpu_info = []
    try:
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)  # MiB
            memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)    # MiB
            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)  # MiB
            memory_percent = (memory_allocated / memory_total) * 100.0
            
            gpu_info.append({
                "index": i,
                "memory_allocated_mb": memory_allocated,
                "memory_reserved_mb": memory_reserved,
                "memory_total_mb": memory_total,
                "memory_percent": memory_percent
            })
        return gpu_info
    except Exception as e:
        print(f"Error getting PyTorch GPU info: {e}")
        return []

def get_cpu_memory_info():
    """Get CPU and memory usage"""
    try:
        # Memory info
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        total_mem = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) / 1024  # MiB
        free_mem = int([line for line in meminfo.split('\n') if 'MemFree' in line][0].split()[1]) / 1024  # MiB
        available_mem = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1]) / 1024  # MiB
        
        used_mem = total_mem - available_mem
        mem_percent = (used_mem / total_mem) * 100.0
        
        # CPU info
        cpu_percent = 0.0
        try:
            cpu_info = subprocess.check_output(['top', '-bn1'], universal_newlines=True)
            cpu_lines = [line for line in cpu_info.split('\n') if '%Cpu(s)' in line]
            if cpu_lines:
                cpu_percent = 100.0 - float(cpu_lines[0].split(',')[3].strip().split()[0])
        except:
            # Fallback
            loadavg = os.getloadavg()[0]
            cpu_count = os.cpu_count()
            cpu_percent = (loadavg / cpu_count) * 100.0
            
        return {
            "cpu_percent": cpu_percent,
            "memory_total_mb": total_mem,
            "memory_used_mb": used_mem,
            "memory_available_mb": available_mem,
            "memory_percent": mem_percent
        }
    except Exception as e:
        print(f"Error getting CPU/memory info: {e}")
        return {
            "cpu_percent": 0.0,
            "memory_total_mb": 0.0,
            "memory_used_mb": 0.0,
            "memory_available_mb": 0.0,
            "memory_percent": 0.0
        }

def monitor_resources(output_file, interval=10, duration=None):
    """Monitor system resources at regular intervals"""
    print(f"Starting container resource monitoring every {interval} seconds...")
    print(f"Results will be saved to {output_file}")
    print("Press Ctrl+C to stop monitoring")
    
    # Get container info
    container_id = subprocess.check_output(['cat', '/etc/hostname'], universal_newlines=True).strip()
    
    # Initialize data structure
    data = {
        "container_id": container_id,
        "start_time": datetime.datetime.now().isoformat(),
        "samples": [],
        "summary": {}
    }
    
    try:
        start_time = time.time()
        elapsed_time = 0
        
        # Tracking metrics
        gpu_memory_percent = defaultdict(list)
        gpu_util_percent = defaultdict(list)
        cpu_percent = []
        memory_percent = []
        
        sample_count = 0
        
        while duration is None or elapsed_time < duration:
            # Get system information
            cpu_mem_info = get_cpu_memory_info()
            nvidia_gpu_info = get_gpu_info()
            torch_gpu_info = get_torch_gpu_info()
            
            # Combine GPU info (prefer torch info when available)
            gpu_info = {}
            for gpu in nvidia_gpu_info:
                gpu_info[gpu["index"]] = gpu
                
            # Override with torch information when available
            for gpu in torch_gpu_info:
                idx = gpu["index"]
                if idx in gpu_info:
                    gpu_info[idx]["memory_used_mb"] = gpu["memory_allocated_mb"]
                    gpu_info[idx]["memory_percent"] = gpu["memory_percent"]
                    gpu_info[idx]["memory_reserved_mb"] = gpu["memory_reserved_mb"]
            
            # Create a sample
            sample = {
                "timestamp": datetime.datetime.now().isoformat(),
                "elapsed_seconds": elapsed_time,
                "cpu_memory": cpu_mem_info,
                "gpus": list(gpu_info.values())
            }
            
            # Store the sample
            data["samples"].append(sample)
            sample_count += 1
            
            # Update tracking metrics
            cpu_percent.append(sample["cpu_memory"]["cpu_percent"])
            memory_percent.append(sample["cpu_memory"]["memory_percent"])
            
            for gpu in sample["gpus"]:
                idx = gpu["index"]
                gpu_memory_percent[idx].append(gpu["memory_percent"])
                if "utilization_percent" in gpu:
                    gpu_util_percent[idx].append(gpu["utilization_percent"])
            
            # Print current status
            print(f"\nSample #{sample_count} - Elapsed: {elapsed_time:.1f}s")
            print(f"CPU: {sample['cpu_memory']['cpu_percent']:.1f}% | "
                  f"Memory: {sample['cpu_memory']['memory_percent']:.1f}%")
            
            for gpu in sample["gpus"]:
                print(f"GPU {gpu['index']}: "
                      f"Util: {gpu.get('utilization_percent', 0):.1f}% | "
                      f"Mem: {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB "
                      f"({gpu['memory_percent']:.1f}%)")
            
            # Save intermediate results
            if sample_count % 5 == 0:
                # Calculate summary so far
                summary = {
                    "cpu": {
                        "peak_percent": max(cpu_percent) if cpu_percent else 0,
                        "average_percent": sum(cpu_percent) / len(cpu_percent) if cpu_percent else 0,
                    },
                    "memory": {
                        "peak_percent": max(memory_percent) if memory_percent else 0,
                        "average_percent": sum(memory_percent) / len(memory_percent) if memory_percent else 0,
                    },
                    "gpus": {}
                }
                
                # Process GPU metrics
                for gpu_idx, values in gpu_memory_percent.items():
                    if not values:
                        continue
                        
                    if gpu_idx not in summary["gpus"]:
                        summary["gpus"][gpu_idx] = {}
                        
                    summary["gpus"][gpu_idx]["peak_memory_percent"] = max(values)
                    summary["gpus"][gpu_idx]["average_memory_percent"] = sum(values) / len(values)
                
                for gpu_idx, values in gpu_util_percent.items():
                    if not values:
                        continue
                        
                    if gpu_idx not in summary["gpus"]:
                        summary["gpus"][gpu_idx] = {}
                        
                    summary["gpus"][gpu_idx]["peak_utilization_percent"] = max(values)
                    summary["gpus"][gpu_idx]["average_utilization_percent"] = sum(values) / len(values)
                
                data["summary"] = summary
                
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                print(f"Saved intermediate results to {output_file}")
            
            # Wait for next interval
            time.sleep(interval)
            elapsed_time = time.time() - start_time
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    # Calculate final summary statistics
    summary = {
        "cpu": {
            "peak_percent": max(cpu_percent) if cpu_percent else 0,
            "average_percent": sum(cpu_percent) / len(cpu_percent) if cpu_percent else 0,
        },
        "memory": {
            "peak_percent": max(memory_percent) if memory_percent else 0,
            "average_percent": sum(memory_percent) / len(memory_percent) if memory_percent else 0,
        },
        "gpus": {}
    }
    
    # Process GPU metrics
    for gpu_idx, values in gpu_memory_percent.items():
        if not values:
            continue
            
        if gpu_idx not in summary["gpus"]:
            summary["gpus"][gpu_idx] = {}
            
        summary["gpus"][gpu_idx]["peak_memory_percent"] = max(values)
        summary["gpus"][gpu_idx]["average_memory_percent"] = sum(values) / len(values)
    
    for gpu_idx, values in gpu_util_percent.items():
        if not values:
            continue
            
        if gpu_idx not in summary["gpus"]:
            summary["gpus"][gpu_idx] = {}
            
        summary["gpus"][gpu_idx]["peak_utilization_percent"] = max(values)
        summary["gpus"][gpu_idx]["average_utilization_percent"] = sum(values) / len(values)
    
    data["summary"] = summary
    data["end_time"] = datetime.datetime.now().isoformat()
    data["duration_seconds"] = elapsed_time
    
    # Save final results
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("\n" + "="*50)
    print("Monitoring Complete - Summary:")
    print("="*50)
    print(f"Duration: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Samples collected: {sample_count}")
    print("\nCPU Usage:")
    print(f"  Peak: {summary['cpu']['peak_percent']:.1f}%")
    print(f"  Average: {summary['cpu']['average_percent']:.1f}%")
    
    print("\nMemory Usage:")
    print(f"  Peak: {summary['memory']['peak_percent']:.1f}%")
    print(f"  Average: {summary['memory']['average_percent']:.1f}%")
    
    print("\nGPU Usage:")
    for gpu_idx, stats in summary["gpus"].items():
        print(f"\nGPU {gpu_idx}:")
        print(f"  Peak Memory: {stats.get('peak_memory_percent', 0):.1f}%")
        print(f"  Average Memory: {stats.get('average_memory_percent', 0):.1f}%")
        if "peak_utilization_percent" in stats:
            print(f"  Peak Utilization: {stats.get('peak_utilization_percent', 0):.1f}%")
            print(f"  Average Utilization: {stats.get('average_utilization_percent', 0):.1f}%")
    
    print(f"\nResults saved to {output_file}")
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor container resources")
    parser.add_argument("--output", type=str, default="/tmp/container_monitoring.json",
                        help="Output file for monitoring data (JSON)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=int, default=None,
                        help="Monitoring duration in seconds (default: run until interrupted)")
    
    args = parser.parse_args()
    
    monitor_resources(args.output, args.interval, args.duration)