"""
Memory monitoring utilities for timelyLLM system
"""
import psutil
import os
import time
import threading
from typing import Optional
from util.log_config import logger


class MemoryMonitor:
    """Monitor memory usage of the current process."""
    
    def __init__(self, log_interval: float = 5.0):
        """
        Initialize memory monitor.
        
        Args:
            log_interval: Interval in seconds between memory logs
        """
        self.process = psutil.Process(os.getpid())
        self.log_interval = log_interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.peak_memory_mb = 0.0
        
    def get_memory_usage(self) -> dict:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory statistics in MB
        """
        mem_info = self.process.memory_info()
        memory_mb = mem_info.rss / 1024 / 1024  # Convert to MB
        
        # Update peak memory
        if memory_mb > self.peak_memory_mb:
            self.peak_memory_mb = memory_mb
        
        # Get system memory
        system_mem = psutil.virtual_memory()
        
        return {
            'process_memory_mb': round(memory_mb, 2),
            'peak_memory_mb': round(self.peak_memory_mb, 2),
            'system_total_mb': round(system_mem.total / 1024 / 1024, 2),
            'system_available_mb': round(system_mem.available / 1024 / 1024, 2),
            'system_percent': round(system_mem.percent, 2)
        }
    
    def log_memory(self):
        """Log current memory usage."""
        stats = self.get_memory_usage()
        logger.info(
            f"Memory: Process={stats['process_memory_mb']}MB, "
            f"Peak={stats['peak_memory_mb']}MB, "
            f"System={stats['system_percent']}% "
            f"({stats['system_available_mb']}MB available)"
        )
        return stats
    
    def _monitor_loop(self):
        """Internal monitoring loop."""
        while self.monitoring:
            self.log_memory()
            time.sleep(self.log_interval)
    
    def start_monitoring(self):
        """Start continuous memory monitoring in background thread."""
        if self.monitoring:
            logger.warning("Memory monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Memory monitoring started (interval={self.log_interval}s)")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.log_interval + 1)
        
        # Log final stats
        final_stats = self.log_memory()
        logger.info(f"Memory monitoring stopped. Peak usage: {final_stats['peak_memory_mb']}MB")


def check_memory_threshold(threshold_mb: float = 1000.0) -> bool:
    """
    Check if current process memory exceeds threshold.
    
    Args:
        threshold_mb: Memory threshold in MB
        
    Returns:
        True if memory exceeds threshold
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_mb = mem_info.rss / 1024 / 1024
    
    if memory_mb > threshold_mb:
        logger.warning(f"Memory usage ({memory_mb:.2f}MB) exceeds threshold ({threshold_mb}MB)")
        return True
    return False


if __name__ == "__main__":
    # Test the memory monitor
    monitor = MemoryMonitor(log_interval=2.0)
    monitor.start_monitoring()
    
    # Simulate some work
    time.sleep(10)
    
    monitor.stop_monitoring()


