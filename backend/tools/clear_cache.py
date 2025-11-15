#!/usr/bin/env python3
"""
Backend Cache Management CLI Tool
Professional cache management for development and maintenance

Usage:
    python clear_cache.py              # Show cache statistics
    python clear_cache.py --clear      # Clear all caches
    python clear_cache.py --status     # Show cache status
    python clear_cache.py --force      # Force clear even if backend is running
"""

import sys
import argparse
from pathlib import Path

# Add backend to path
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from utils.cache_manager import get_cache_manager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_size(bytes_val: float) -> str:
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def show_statistics():
    """Show current cache statistics"""
    cache_manager = get_cache_manager(backend_root)
    stats = cache_manager.get_cache_statistics()
    
    print("\n" + "="*70)
    print("📊 BACKEND CACHE STATISTICS")
    print("="*70)
    
    print(f"\n🐍 Python Bytecode Cache:")
    print(f"   Directories: {stats['python_cache']['dirs']}")
    print(f"   Files: {stats['python_cache']['files']}")
    print(f"   Size: {format_size(stats['python_cache']['size_kb'] * 1024)}")
    
    print(f"\n📄 Temporary Files:")
    print(f"   Files: {stats['temp_files']['count']}")
    print(f"   Size: {format_size(stats['temp_files']['size_kb'] * 1024)}")
    
    print(f"\n💾 Backup Files:")
    print(f"   Files: {stats['backup_files']['count']}")
    print(f"   Size: {format_size(stats['backup_files']['size_kb'] * 1024)}")
    
    print(f"\n📁 Total Data Directory:")
    print(f"   Size: {format_size(stats['data_dir_size_mb'] * 1024 * 1024)}")
    
    # Check if backend is running
    is_running = not cache_manager.is_fresh_startup()
    status = "🟢 RUNNING" if is_running else "🔴 STOPPED"
    print(f"\n🖥️  Backend Status: {status}")
    
    print("\n" + "="*70 + "\n")


def show_status():
    """Show backend runtime status"""
    cache_manager = get_cache_manager(backend_root)
    
    print("\n" + "="*70)
    print("🖥️  BACKEND RUNTIME STATUS")
    print("="*70)
    
    is_fresh = cache_manager.is_fresh_startup()
    
    if is_fresh:
        print("\n🔴 Backend is STOPPED")
        print("   Next startup will clear all caches")
        print("   State file: Not found or stale")
    else:
        print("\n🟢 Backend is RUNNING")
        print("   Caches are being preserved")
        print(f"   State file: {cache_manager.runtime_state_file}")
        
        if cache_manager.runtime_state_file.exists():
            import datetime
            mtime = cache_manager.runtime_state_file.stat().st_mtime
            last_touch = datetime.datetime.fromtimestamp(mtime)
            print(f"   Last activity: {last_touch.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*70 + "\n")


def clear_caches(force: bool = False):
    """Clear all caches"""
    cache_manager = get_cache_manager(backend_root)
    
    # Check if backend is running
    if not force and not cache_manager.is_fresh_startup():
        print("\n⚠️  WARNING: Backend appears to be running!")
        print("   Clearing caches while running may cause issues.")
        response = input("   Continue anyway? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("   Aborted.")
            return
    
    print("\n" + "="*70)
    print("🧹 CLEARING ALL BACKEND CACHES")
    print("="*70 + "\n")
    
    stats = cache_manager.clear_all_caches(keep_backups=3)
    
    print("\n" + "-"*70)
    print("📊 CLEANUP SUMMARY")
    print("-"*70)
    
    print(f"\n🐍 Python Bytecode Cache:")
    print(f"   Directories: {stats['python_cache']['dirs_cleared']}")
    print(f"   Files: {stats['python_cache']['files_removed']}")
    print(f"   Freed: {format_size(stats['python_cache']['bytes_freed'])}")
    
    print(f"\n📄 Temporary Files:")
    print(f"   Files: {stats['temp_files']['files_removed']}")
    print(f"   Freed: {format_size(stats['temp_files']['bytes_freed'])}")
    
    print(f"\n💾 Old Backups:")
    print(f"   Files: {stats['old_backups']['files_removed']}")
    print(f"   Freed: {format_size(stats['old_backups']['bytes_freed'])}")
    
    print(f"\n✅ TOTAL FREED: {format_size(stats['total_bytes_freed'])}")
    print("\n" + "="*70 + "\n")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Backend Cache Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clear_cache.py              # Show cache statistics
  python clear_cache.py --clear      # Clear all caches
  python clear_cache.py --status     # Show runtime status
  python clear_cache.py --force      # Force clear (ignore running state)
        """
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear all caches (Python bytecode, temp files, old backups)'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show backend runtime status'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force cache clearing even if backend is running'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show cache statistics (default action)'
    )
    
    args = parser.parse_args()
    
    # If no arguments, show statistics
    if not any([args.clear, args.status, args.stats]):
        show_statistics()
        return
    
    # Execute requested action
    if args.status:
        show_status()
    
    if args.clear:
        clear_caches(force=args.force)
    
    if args.stats:
        show_statistics()


if __name__ == "__main__":
    main()
