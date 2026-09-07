import json
import os
import platform
import threading
import time
from datetime import datetime


# 获取当前进程 RSS 内存，单位 MB。
def get_process_rss_mb():
    system = platform.system().lower()
    if system == 'windows':
        return get_windows_rss_mb()
    return get_unix_rss_mb()


# 通过 Windows API 获取当前进程工作集大小。
def get_windows_rss_mb():
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD),
                ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
                ('PrivateUsage', ctypes.c_size_t),
            ]

        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(ProcessMemoryCountersEx)
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCountersEx),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return None
        return counters.WorkingSetSize / (1024 ** 2)
    except Exception:
        return None


# 在 Unix-like 系统上获取当前进程最大 RSS。
def get_unix_rss_mb():
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system().lower() == 'darwin':
            return rss / (1024 ** 2)
        return rss / 1024
    except Exception:
        return None


# 将秒数格式化为紧凑文本。
def format_seconds(seconds):
    minutes, sec = divmod(float(seconds), 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes}m {sec:.1f}s"
    if minutes:
        return f"{minutes}m {sec:.1f}s"
    return f"{sec:.1f}s"


class ResourceTracker:
    def __init__(self, name, device=None, sample_interval=0.5):
        self.name = name
        self.device = device
        self.sample_interval = sample_interval
        self.start_time = None
        self.end_time = None
        self.start_rss_mb = None
        self.end_rss_mb = None
        self.peak_rss_mb = None
        self._stop_event = threading.Event()
        self._thread = None

    # 开始记录时间、CPU 内存和 CUDA 峰值。
    def start(self):
        self.start_time = time.perf_counter()
        self.start_rss_mb = get_process_rss_mb()
        self.peak_rss_mb = self.start_rss_mb
        self._reset_cuda_peak()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    # 结束记录并返回汇总字典。
    def stop(self):
        self.end_time = time.perf_counter()
        self.end_rss_mb = get_process_rss_mb()
        self._update_peak(self.end_rss_mb)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.summary()

    # 周期性采样 RSS 峰值。
    def _sample_loop(self):
        while not self._stop_event.wait(self.sample_interval):
            self._update_peak(get_process_rss_mb())

    # 更新 RSS 峰值。
    def _update_peak(self, value):
        if value is None:
            return
        if self.peak_rss_mb is None or value > self.peak_rss_mb:
            self.peak_rss_mb = value

    # 重置 CUDA 峰值统计。
    def _reset_cuda_peak(self):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    # 获取 CUDA 峰值内存。
    def _cuda_summary(self):
        try:
            import torch

            if not torch.cuda.is_available():
                return {}
            return {
                'cuda_device': str(self.device),
                'cuda_max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
                'cuda_max_reserved_mb': torch.cuda.max_memory_reserved() / (1024 ** 2),
            }
        except Exception:
            return {}

    # 汇总资源记录。
    def summary(self):
        elapsed = None
        if self.start_time is not None:
            elapsed = (self.end_time or time.perf_counter()) - self.start_time
        result = {
            'name': self.name,
            'elapsed_seconds': elapsed,
            'elapsed': format_seconds(elapsed) if elapsed is not None else None,
            'rss_start_mb': self.start_rss_mb,
            'rss_end_mb': self.end_rss_mb,
            'rss_peak_mb': self.peak_rss_mb,
        }
        result.update(self._cuda_summary())
        return result


# 保存资源记录 JSON。
def save_resource_report(run_dir, report):
    path = os.path.join(run_dir, 'resource_usage.json')
    payload = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        **report,
    }
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return path


# 打印一行资源摘要。
def format_resource_summary(summary):
    parts = [f"time={summary.get('elapsed')}"]
    if summary.get('rss_peak_mb') is not None:
        parts.append(f"rss_peak={summary['rss_peak_mb']:.1f}MB")
    if summary.get('cuda_max_allocated_mb') is not None:
        parts.append(f"cuda_alloc_peak={summary['cuda_max_allocated_mb']:.1f}MB")
    if summary.get('cuda_max_reserved_mb') is not None:
        parts.append(f"cuda_reserved_peak={summary['cuda_max_reserved_mb']:.1f}MB")
    return " | ".join(parts)
