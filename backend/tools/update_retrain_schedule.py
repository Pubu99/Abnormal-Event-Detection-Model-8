#!/usr/bin/env python3
"""
CLI helper to update the retrain schedule JSON file used by the running server.

Usage:
  python3 update_retrain_schedule.py --start 22 --end 04 --days 0,1,2,3,4 --min_new 20

This writes `backend/data/retrain_schedule.json`. The running FastAPI app polls the file and
will pick up changes automatically.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=int, help='start hour UTC (0-23)')
    p.add_argument('--end', type=int, help='end hour UTC (0-23)')
    p.add_argument('--days', type=str, help='comma-separated weekdays 0=Mon..6=Sun')
    p.add_argument('--min_new', type=int, help='minimum new experiences to trigger retrain')
    p.add_argument('--cpu_threshold', type=int, help='CPU percent threshold to pause training')
    p.add_argument('--gpu_util_threshold', type=int, help='GPU util percent threshold to pause training')
    p.add_argument('--gpu_mem_threshold', type=int, help='GPU mem percent threshold to pause training')
    p.add_argument('--cooldown', type=int, help='cooldown seconds required before resuming')
    p.add_argument('--enable', type=int, choices=[0,1], help='enable (1) or disable (0) scheduled retrains')

    args = p.parse_args()
    cfg_path = Path(__file__).parent.parent / 'data' / 'retrain_schedule.json'
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            cfg = {}

    if args.start is not None:
        cfg['start_hour'] = args.start
    if args.end is not None:
        cfg['end_hour'] = args.end
    if args.days is not None:
        cfg['days'] = [int(x) for x in args.days.split(',') if x.strip().isdigit()]
    if args.min_new is not None:
        cfg['min_new'] = args.min_new
    if args.cpu_threshold is not None:
        cfg['cpu_threshold_percent'] = args.cpu_threshold
    if args.gpu_util_threshold is not None:
        cfg['gpu_util_threshold_percent'] = args.gpu_util_threshold
    if args.gpu_mem_threshold is not None:
        cfg['gpu_mem_threshold_percent'] = args.gpu_mem_threshold
    if args.cooldown is not None:
        cfg['cooldown_seconds'] = args.cooldown
    if args.enable is not None:
        cfg['enabled'] = bool(args.enable)

    cfg_path.write_text(json.dumps(cfg, indent=2))
    print('Wrote schedule to', cfg_path)


if __name__ == '__main__':
    main()
