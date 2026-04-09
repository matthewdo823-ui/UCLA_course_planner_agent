#!/usr/bin/env python3
"""Launch all 8 course-planner agents as separate processes.

Usage:
    cd ~/Agents\ 2
    python run_all.py

Each agent binds to its own port (8001–8008).  Press Ctrl+C to stop all.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import subprocess
import sys
import time

PORTS = [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008]
MAX_RESTARTS = 3  # per agent


def _kill_ports():
    """Kill any existing processes on the agent ports."""
    for port in PORTS:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True,
            )
            pids = result.stdout.strip().split()
            for pid in pids:
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except (ProcessLookupError, ValueError):
                        pass
        except Exception:
            pass
    if any(
        subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True,
        ).stdout.strip()
        for port in PORTS
    ):
        time.sleep(1)  # give processes a moment to release


def _run_input():
    from course_planner.agents.input_agent import agent
    agent.run()


def _run_available_classes():
    from course_planner.agents.available_classes_agent import agent
    agent.run()


def _run_enrollment():
    from course_planner.agents.enrollment_agent import agent
    agent.run()


def _run_bruinwalk():
    from course_planner.agents.bruinwalk_agent import agent
    agent.run()


def _run_grade_dist():
    from course_planner.agents.grade_dist_agent import agent
    agent.run()


def _run_schedule():
    from course_planner.agents.schedule_agent import agent
    agent.run()


def _run_ranking():
    from course_planner.agents.ranking_agent import agent
    agent.run()


def _run_report():
    from course_planner.agents.report_agent import agent
    agent.run()


AGENTS: list[tuple[str, int, callable]] = [
    ("input",             8001, _run_input),
    ("available_classes", 8002, _run_available_classes),
    ("enrollment",        8003, _run_enrollment),
    ("bruinwalk",         8004, _run_bruinwalk),
    ("grade_dist",        8005, _run_grade_dist),
    ("schedule",          8006, _run_schedule),
    ("ranking",           8007, _run_ranking),
    ("report",            8008, _run_report),
]


def main():
    # Free ports from any prior run
    print("Cleaning up stale processes on ports 8001–8008 …")
    _kill_ports()

    processes: list[multiprocessing.Process] = []
    restart_counts: list[int] = [0] * len(AGENTS)

    for name, port, target in AGENTS:
        p = multiprocessing.Process(target=target, name=name, daemon=True)
        processes.append(p)

    print(f"Starting {len(processes)} agents …")
    for p in processes:
        p.start()
        print(f"  started {p.name} (pid {p.pid})")
        # Stagger starts so the grade_dist agent (slow CSV download)
        # doesn't block others, and ports don't race
        time.sleep(1)

    print("\nAll agents launched.  Press Ctrl+C to stop.\n")

    def _shutdown(sig, frame):
        print("\nShutting down …")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Monitor loop — restart crashed agents up to MAX_RESTARTS times
    while True:
        for i, p in enumerate(processes):
            if not p.is_alive() and p.exitcode is not None and p.exitcode != 0:
                name, port, target = AGENTS[i]
                if restart_counts[i] >= MAX_RESTARTS:
                    if restart_counts[i] == MAX_RESTARTS:
                        print(f"  {name} (port {port}) failed {MAX_RESTARTS} times — giving up")
                        restart_counts[i] += 1  # only print once
                    continue
                restart_counts[i] += 1
                wait = restart_counts[i] * 3  # backoff: 3s, 6s, 9s
                print(f"  {name} (port {port}) exited ({p.exitcode}) "
                      f"— restart {restart_counts[i]}/{MAX_RESTARTS} in {wait}s")
                time.sleep(wait)
                new_p = multiprocessing.Process(
                    target=target, name=name, daemon=True
                )
                new_p.start()
                processes[i] = new_p
        time.sleep(3)


if __name__ == "__main__":
    main()
