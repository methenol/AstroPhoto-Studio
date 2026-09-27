"""The machine's CPU cores and memory, for sizing worker pools (macOS, Linux, Windows)."""
from __future__ import annotations

import os


def cpu_cores() -> int:
    return max(1, os.cpu_count() or 1)


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
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
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
