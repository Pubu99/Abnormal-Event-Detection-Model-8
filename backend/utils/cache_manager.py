"""
Professional Cache Manager for Backend
Handles cache lifecycle management with smart clearing on startup
Author: Professional Backend Engineer
Date: 2025-11-13
"""

import os
import shutil
import glob
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Professional cache management system
    - Clears stale caches on fresh startup
    - Preserves active runtime caches
    - Provides cache statistics and monitoring
    """
    
    def __init__(self, backend_root: Path):
        """
        Initialize cache manager
        
        Args:
            backend_root: Root directory of backend application
        """
        self.backend_root = Path(backend_root)
        self.data_dir = self.backend_root / "data"
        self.cache_dirs = [
            self.backend_root / "api" / "__pycache__",
            self.backend_root / "core" / "__pycache__",
            self.backend_root / "services" / "__pycache__",
            self.backend_root / "tools" / "__pycache__",
        ]
        
        # Runtime state file to track if backend is running
        self.runtime_state_file = self.data_dir / ".backend_running"
        self.startup_time = datetime.now()
        
    def is_fresh_startup(self) -> bool:
        """
        Determine if this is a fresh startup or a reload
        
        Returns:
            True if fresh startup (backend was stopped), False if hot reload
        """
        # If runtime state file doesn't exist, it's a fresh startup
        if not self.runtime_state_file.exists():
            return True
        
        # If file exists but is old (>5 minutes), consider it stale
        try:
            mtime = self.runtime_state_file.stat().st_mtime
            age_seconds = datetime.now().timestamp() - mtime
            
            # If state file is older than 5 minutes, it's a fresh startup
            if age_seconds > 300:
                return True
            
            return False
        except Exception as e:
            logger.warning(f"Error checking runtime state: {e}")
            return True
    
    def mark_backend_running(self):
        """Mark backend as running by creating/updating state file"""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.runtime_state_file.touch()
            logger.info(f"✅ Backend marked as running")
        except Exception as e:
            logger.error(f"Failed to mark backend as running: {e}")
    
    def mark_backend_stopped(self):
        """Mark backend as stopped by removing state file"""
        try:
            if self.runtime_state_file.exists():
                self.runtime_state_file.unlink()
                logger.info("🛑 Backend marked as stopped")
        except Exception as e:
            logger.error(f"Failed to mark backend as stopped: {e}")
    
    def clear_python_cache(self) -> Dict[str, Any]:
        """
        Clear Python bytecode cache (__pycache__ directories)
        
        Returns:
            Statistics about cleared cache
        """
        stats = {
            "dirs_cleared": 0,
            "files_removed": 0,
            "bytes_freed": 0
        }
        
        for cache_dir in self.cache_dirs:
            if cache_dir.exists():
                try:
                    # Calculate size before deletion
                    size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
                    file_count = len(list(cache_dir.rglob("*.pyc")))
                    
                    # Remove directory
                    shutil.rmtree(cache_dir)
                    
                    stats["dirs_cleared"] += 1
                    stats["files_removed"] += file_count
                    stats["bytes_freed"] += size
                    
                    logger.info(f"🗑️  Cleared {cache_dir}: {file_count} files, {size/1024:.2f} KB")
                except Exception as e:
                    logger.error(f"Failed to clear {cache_dir}: {e}")
        
        return stats
    
    def clear_temp_files(self) -> Dict[str, Any]:
        """
        Clear temporary files (RL training temp files, etc.)
        
        Returns:
            Statistics about cleared temp files
        """
        stats = {
            "files_removed": 0,
            "bytes_freed": 0
        }
        
        if not self.data_dir.exists():
            return stats
        
        # Patterns for temporary files
        temp_patterns = [
            "*.tmp",
            "*.tmp.*",
            "rl_policy.retrain.*.tmp.metrics.json",
            ".*.swp",
            ".*.swo",
        ]
        
        for pattern in temp_patterns:
            for temp_file in self.data_dir.glob(pattern):
                try:
                    size = temp_file.stat().st_size
                    temp_file.unlink()
                    
                    stats["files_removed"] += 1
                    stats["bytes_freed"] += size
                    
                    logger.debug(f"🗑️  Removed temp file: {temp_file.name}")
                except Exception as e:
                    logger.error(f"Failed to remove {temp_file}: {e}")
        
        if stats["files_removed"] > 0:
            logger.info(
                f"🗑️  Cleared {stats['files_removed']} temp files, "
                f"{stats['bytes_freed']/1024:.2f} KB freed"
            )
        
        return stats
    
    def clear_old_backups(self, keep_recent: int = 3) -> Dict[str, Any]:
        """
        Clear old backup files, keeping only recent ones
        
        Args:
            keep_recent: Number of recent backups to keep
            
        Returns:
            Statistics about cleared backups
        """
        stats = {
            "files_removed": 0,
            "bytes_freed": 0
        }
        
        if not self.data_dir.exists():
            return stats
        
        # Find backup files (*.bak.*, *.pt.bak.*)
        backup_patterns = [
            "*.bak.*",
            "*.pt.bak.*",
        ]
        
        for pattern in backup_patterns:
            backup_files = sorted(
                self.data_dir.glob(pattern),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            
            # Keep only recent backups
            for old_backup in backup_files[keep_recent:]:
                try:
                    size = old_backup.stat().st_size
                    old_backup.unlink()
                    
                    stats["files_removed"] += 1
                    stats["bytes_freed"] += size
                    
                    logger.debug(f"🗑️  Removed old backup: {old_backup.name}")
                except Exception as e:
                    logger.error(f"Failed to remove {old_backup}: {e}")
        
        if stats["files_removed"] > 0:
            logger.info(
                f"🗑️  Cleared {stats['files_removed']} old backups, "
                f"{stats['bytes_freed']/1024:.2f} KB freed"
            )
        
        return stats
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get current cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "python_cache": {"dirs": 0, "files": 0, "size_kb": 0},
            "temp_files": {"count": 0, "size_kb": 0},
            "backup_files": {"count": 0, "size_kb": 0},
            "data_dir_size_mb": 0,
        }
        
        # Python cache
        for cache_dir in self.cache_dirs:
            if cache_dir.exists():
                stats["python_cache"]["dirs"] += 1
                pyc_files = list(cache_dir.rglob("*.pyc"))
                stats["python_cache"]["files"] += len(pyc_files)
                stats["python_cache"]["size_kb"] += sum(
                    f.stat().st_size for f in pyc_files
                ) / 1024
        
        # Temp files
        if self.data_dir.exists():
            temp_files = list(self.data_dir.glob("*.tmp*"))
            stats["temp_files"]["count"] = len(temp_files)
            stats["temp_files"]["size_kb"] = sum(
                f.stat().st_size for f in temp_files if f.is_file()
            ) / 1024
            
            # Backup files
            backup_files = list(self.data_dir.glob("*.bak.*"))
            stats["backup_files"]["count"] = len(backup_files)
            stats["backup_files"]["size_kb"] = sum(
                f.stat().st_size for f in backup_files if f.is_file()
            ) / 1024
            
            # Total data dir size
            stats["data_dir_size_mb"] = sum(
                f.stat().st_size for f in self.data_dir.rglob("*") if f.is_file()
            ) / (1024 * 1024)
        
        return stats
    
    def clear_all_caches(self, keep_backups: int = 3) -> Dict[str, Any]:
        """
        Clear all caches (Python bytecode + temp files + old backups)
        
        Args:
            keep_backups: Number of recent backups to keep
            
        Returns:
            Combined statistics
        """
        logger.info("🧹 Starting comprehensive cache cleanup...")
        
        combined_stats = {
            "python_cache": self.clear_python_cache(),
            "temp_files": self.clear_temp_files(),
            "old_backups": self.clear_old_backups(keep_backups),
            "total_bytes_freed": 0,
        }
        
        # Calculate total
        combined_stats["total_bytes_freed"] = (
            combined_stats["python_cache"]["bytes_freed"] +
            combined_stats["temp_files"]["bytes_freed"] +
            combined_stats["old_backups"]["bytes_freed"]
        )
        
        total_mb = combined_stats["total_bytes_freed"] / (1024 * 1024)
        logger.info(f"✅ Cache cleanup complete! Total freed: {total_mb:.2f} MB")
        
        return combined_stats
    
    def startup_cache_cleanup(self):
        """
        Perform cache cleanup on startup based on runtime state
        - Fresh startup: Clear all caches
        - Hot reload: Preserve caches
        """
        is_fresh = self.is_fresh_startup()
        
        if is_fresh:
            logger.info("🔄 Detected FRESH STARTUP - Clearing all caches...")
            stats = self.clear_all_caches()
            logger.info(f"   Python cache: {stats['python_cache']['files_removed']} files removed")
            logger.info(f"   Temp files: {stats['temp_files']['files_removed']} files removed")
            logger.info(f"   Old backups: {stats['old_backups']['files_removed']} files removed")
            logger.info(f"   Total freed: {stats['total_bytes_freed']/(1024*1024):.2f} MB")
        else:
            logger.info("♻️  Detected HOT RELOAD - Preserving runtime caches")
            logger.info("   Caches will be maintained for optimal performance")
        
        # Mark backend as running
        self.mark_backend_running()
        
        return is_fresh


# Global cache manager instance
_cache_manager = None


def get_cache_manager(backend_root: Path = None) -> CacheManager:
    """
    Get or create global cache manager instance
    
    Args:
        backend_root: Backend root directory (required on first call)
        
    Returns:
        CacheManager instance
    """
    global _cache_manager
    
    if _cache_manager is None:
        if backend_root is None:
            # Try to auto-detect backend root
            backend_root = Path(__file__).parent.parent
        _cache_manager = CacheManager(backend_root)
    
    return _cache_manager
