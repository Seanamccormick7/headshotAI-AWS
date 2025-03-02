#!/bin/bash
# docker_monitor.sh - Monitor Docker container resource usage
# Usage: ./docker_monitor.sh [container_name] [interval_seconds] [duration_seconds]

CONTAINER=${1:-dreambooth-worker}
INTERVAL=${2:-30}
DURATION=${3:-0}  # 0 means run indefinitely
OUTPUT_FILE="docker_${CONTAINER}_stats.json"

echo "Starting Docker container monitoring for: $CONTAINER"
echo "Stats will be collected every $INTERVAL seconds"
echo "Results will be saved to $OUTPUT_FILE"
[ $DURATION -gt 0 ] && echo "Monitoring will run for $DURATION seconds" || echo "Monitoring will run until Ctrl+C is pressed"

# Initialize JSON output
echo "{" > $OUTPUT_FILE
echo "  \"container\": \"$CONTAINER\"," >> $OUTPUT_FILE
echo "  \"start_time\": \"$(date -Iseconds)\"," >> $OUTPUT_FILE
echo "  \"instance_type\": \"$(curl -s http://169.254.169.254/latest/meta-data/instance-type)\"," >> $OUTPUT_FILE
echo "  \"samples\": [" >> $OUTPUT_FILE

# Track max values
MAX_CPU=0
MAX_MEM=0
MAX_GPU_UTIL=0
MAX_GPU_MEM=0

# Counter for samples
COUNT=0
START_TIME=$(date +%s)

# Function to get GPU stats for the container
function get_gpu_stats() {
    # Get the PIDs running in the container
    container_pids=$(docker top $CONTAINER -eo pid | tail -n +2)
    
    # Get GPU stats
    gpu_stats=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
    
    # Initialize result array
    gpu_result=()
    
    # For each GPU
    while IFS=, read -r gpu_id utilization mem_used mem_total; do
        # Remove leading/trailing spaces
        gpu_id=$(echo $gpu_id | xargs)
        utilization=$(echo $utilization | xargs)
        mem_used=$(echo $mem_used | xargs)
        mem_total=$(echo $mem_total | xargs)
        
        # Check processes on this GPU to see if any belong to our container
        processes=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits)
        container_gpu_mem=0
        
        # For each process running on GPUs
        while IFS=, read -r pid gpu_uuid; do
            pid=$(echo $pid | xargs)
            for container_pid in $container_pids; do
                # If this is a process from our container
                if [[ "$pid" == "$container_pid" ]] || [[ "$pid" == $(pgrep -P $container_pid) ]]; then
                    # This GPU is being used by our container
                    gpu_result+=("    {\"index\": $gpu_id, \"utilization\": $utilization, \"memory_used\": $mem_used, \"memory_total\": $mem_total, \"memory_percent\": $(echo "$mem_used/$mem_total*100" | bc -l)}")
                    
                    # Update max values
                    if (( $(echo "$utilization > $MAX_GPU_UTIL" | bc -l) )); then
                        MAX_GPU_UTIL=$utilization
                    fi
                    
                    # Calculate memory percentage
                    MEM_PERCENT=$(echo "$mem_used/$mem_total*100" | bc -l)
                    if (( $(echo "$MEM_PERCENT > $MAX_GPU_MEM" | bc -l) )); then
                        MAX_GPU_MEM=$MEM_PERCENT
                    fi
                    
                    break 2
                fi
            done
        done <<< "$processes"
    done <<< "$gpu_stats"
    
    # Format as JSON array
    if [ ${#gpu_result[@]} -eq 0 ]; then
        echo "[]"
    else
        echo "[$(IFS=,; echo "${gpu_result[*]}")]"
    fi
}

# Trap Ctrl+C to gracefully finish the JSON file
trap 'echo ""; echo "Monitoring stopped. Processing results..."; break;' INT

# Main monitoring loop
while true; do
    # Get Docker stats for the container
    STATS=$(docker stats --no-stream --format "{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" $CONTAINER)
    
    # Parse the stats
    if [ -z "$STATS" ]; then
        echo "Error: Container $CONTAINER not found or not running."
        exit 1
    fi
    
    # Extract values
    CPU_PERC=$(echo $STATS | awk '{print $1}' | sed 's/%//')
    MEM_USAGE=$(echo $STATS | awk '{print $2}' | cut -d '/' -f1)
    MEM_PERC=$(echo $STATS | awk '{print $3}' | sed 's/%//')
    
    # Convert memory to MiB for consistency
    if [[ $MEM_USAGE == *GiB ]]; then
        MEM_USAGE_MB=$(echo $MEM_USAGE | sed 's/GiB//' | awk '{print $1 * 1024}')
    elif [[ $MEM_USAGE == *MiB ]]; then
        MEM_USAGE_MB=$(echo $MEM_USAGE | sed 's/MiB//')
    else
        MEM_USAGE_MB=0
    fi
    
    # Get GPU stats for this container
    GPU_STATS=$(get_gpu_stats)
    
    # Get current time and calculate elapsed seconds
    CURRENT_TIME=$(date +%s)
    ELAPSED_SECONDS=$((CURRENT_TIME - START_TIME))
    
    # Update max values
    if (( $(echo "$CPU_PERC > $MAX_CPU" | bc -l) )); then
        MAX_CPU=$CPU_PERC
    fi
    
    if (( $(echo "$MEM_PERC > $MAX_MEM" | bc -l) )); then
        MAX_MEM=$MEM_PERC
    fi
    
    # Append to JSON with comma if not the first entry
    if [ $COUNT -gt 0 ]; then
        echo "," >> $OUTPUT_FILE
    fi
    
    # Add this sample
    cat <<EOT >> $OUTPUT_FILE
    {
      "timestamp": "$(date -Iseconds)",
      "elapsed_seconds": $ELAPSED_SECONDS,
      "cpu_percent": $CPU_PERC,
      "memory_usage_mb": $MEM_USAGE_MB,
      "memory_percent": $MEM_PERC,
      "gpus": $GPU_STATS
    }
EOT
    
    COUNT=$((COUNT + 1))
    
    # Print progress
    echo -ne "\rSamples: $COUNT | CPU: ${CPU_PERC}% | Memory: ${MEM_PERC}% | GPU Util: ${MAX_GPU_UTIL}% | GPU Mem: ${MAX_GPU_MEM}%"
    
    # Check if we've reached the duration limit
    if [ $DURATION -gt 0 ] && [ $ELAPSED_SECONDS -ge $DURATION ]; then
        echo -e "\nReached specified duration of $DURATION seconds."
        break
    fi
    
    # Wait for the next interval
    sleep $INTERVAL
done

# Complete the JSON file
echo "" >> $OUTPUT_FILE
echo "  ]," >> $OUTPUT_FILE
echo "  \"summary\": {" >> $OUTPUT_FILE
echo "    \"samples\": $COUNT," >> $OUTPUT_FILE
echo "    \"duration_seconds\": $ELAPSED_SECONDS," >> $OUTPUT_FILE
echo "    \"peak_cpu_percent\": $MAX_CPU," >> $OUTPUT_FILE
echo "    \"peak_memory_percent\": $MAX_MEM," >> $OUTPUT_FILE
echo "    \"peak_gpu_util_percent\": $MAX_GPU_UTIL," >> $OUTPUT_FILE
echo "    \"peak_gpu_memory_percent\": $MAX_GPU_MEM" >> $OUTPUT_FILE
echo "  }," >> $OUTPUT_FILE
echo "  \"end_time\": \"$(date -Iseconds)\"" >> $OUTPUT_FILE
echo "}" >> $OUTPUT_FILE

# Print summary
echo -e "\n\n======= Resource Usage Summary ======="
echo "Container: $CONTAINER"
echo "Duration: $ELAPSED_SECONDS seconds ($(echo "$ELAPSED_SECONDS / 60" | bc -l) minutes)"
echo "Samples: $COUNT"
echo ""
echo "Peak CPU Usage: ${MAX_CPU}%"
echo "Peak Memory Usage: ${MAX_MEM}%"
echo "Peak GPU Utilization: ${MAX_GPU_UTIL}%"
echo "Peak GPU Memory: ${MAX_GPU_MEM}%"
echo ""
echo "Results saved to $OUTPUT_FILE"

# Generate recommendations
echo -e "\n======= EC2 Instance Recommendations ======="
echo "Current Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)"
echo ""

if (( $(echo "$MAX_GPU_MEM < 50" | bc -l) )); then
    echo "GPU Memory: Peak usage ${MAX_GPU_MEM}% - UNDERUTILIZED"
    echo "  - Consider downgrading to a smaller GPU instance"
    if (( $(echo "$MAX_GPU_MEM < 30" | bc -l) )); then
        echo "  - g4dn.xlarge might be sufficient (16GB GPU memory)"
    else
        echo "  - g5.xlarge might be appropriate (24GB GPU memory)"
    fi
elif (( $(echo "$MAX_GPU_MEM > 85" | bc -l) )); then
    echo "GPU Memory: Peak usage ${MAX_GPU_MEM}% - RISK OF OOM"
    echo "  - Consider upgrading to a larger GPU instance"
else
    echo "GPU Memory: Peak usage ${MAX_GPU_MEM}% - WELL UTILIZED"
    echo "  - Current instance size seems appropriate"
fi

echo ""
if (( $(echo "$MAX_CPU < 40" | bc -l) )); then
    echo "CPU: Peak usage ${MAX_CPU}% - UNDERUTILIZED"
    echo "  - Consider an instance with fewer vCPUs"
else
    echo "CPU: Peak usage ${MAX_CPU}% - WELL UTILIZED"
fi

echo ""
if (( $(echo "$MAX_MEM < 50" | bc -l) )); then
    echo "Memory: Peak usage ${MAX_MEM}% - UNDERUTILIZED"
    echo "  - Consider an instance with less system memory"
else
    echo "Memory: Peak usage ${MAX_MEM}% - WELL UTILIZED"
fi