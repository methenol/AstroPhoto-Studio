"""The machine's CPU cores and memory, for sizing worker pools (macOS, Linux, Windows).
In a container, the cgroup limits (``docker run --cpus / --memory``, compose ``cpus`` /
``mem_limit``) count instead of the host's: a pool sized for the host would be OOM-killed."""
from __future__ import annotations

import os


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _cgroup_cpus() -> float | None:
    v2 = _read("/sys/fs/cgroup/cpu.max")                       # "max 100000" or "400000 100000"
    if v2:
        quota, period = (v2.split() + ["100000"])[:2]
        if quota != "max":
            return int(quota) / int(period)
    q, p = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if q and p and int(q) > 0:
        return int(q) / int(p)
    return None


def _cgroup_ram() -> float | None:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        v = _read(path)
        if v and v != "max" and int(v) < 2 ** 60:                # v1 reports "no limit" as ~2^63
            return float(v)
    return None


def cpu_cores() -> int:
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    q = _cgroup_cpus()
    if q:
        n = min(n, max(1, int(q)))
    return max(1, n)


def total_ram_bytes() -> float:
    """Physical memory (unified memory on Apple silicon)."""
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return float(st.ullTotalPhys)
        phys = float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        lim = _cgroup_ram()
        return min(phys, lim) if lim else phys
    except Exception:
        return 8 * 2 ** 30


def ram_budget_bytes(env: str, fraction: float = 0.5) -> float:
    """Memory a stage may use for its workers: ``fraction`` of physical RAM, or the
    environment variable ``env`` (GB) when set."""
    v = os.environ.get(env)
    if v:
        try:
            return float(v) * 2 ** 30
        except ValueError:
            pass
    return fraction * total_ram_bytes()


def workers_for(per_worker_bytes: float, env: str, fraction: float = 0.5, reserve_cores: int = 1) -> int:
    """Parallel workers: every core but ``reserve_cores`` (the UI and the main loop), as many as
    fit in the memory budget."""
    by_ram = int(ram_budget_bytes(env, fraction) // max(per_worker_bytes, 1))
    return max(1, min(cpu_cores() - reserve_cores, by_ram))


def exit_with_parent():
    """In a worker process: exit as soon as the parent does.  A pool whose parent is killed (the
    out-of-memory killer) otherwise lives on: its workers block for ever writing results to a pipe
    nobody reads (their siblings hold its other end), keeping every byte they hold."""
    import multiprocessing as mp
    import threading
    from multiprocessing.connection import wait
    parent = mp.parent_process()
    if parent is None:
        return

    def watch():
        wait([parent.sentinel])
        os._exit(1)
    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()
