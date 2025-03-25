from time import perf_counter

import psutil
from fastapi import Request
from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "http_request_total", "Total HTTP Requests", ["method", "status", "path"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP Request Duration",
    ["method", "status", "path"],
)
REQUEST_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "HTTP Requests in progress",
    ["method", "path"],
)

# System metrics
CPU_USAGE = Gauge("process_cpu_usage", "Current CPU usage in percent")
MEMORY_USAGE = Gauge(
    "process_memory_usage_bytes", "Current memory usage in bytes"
)


async def monitor_requests(request: Request, call_next):
    method = request.method
    path = request.url.path

    REQUEST_IN_PROGRESS.labels(method=method, path=path).inc()
    start_time = perf_counter()
    response = await call_next(request)
    duration = perf_counter() - start_time
    status = response.status_code
    REQUEST_COUNT.labels(method=method, status=status, path=path).inc()
    REQUEST_LATENCY.labels(method=method, status=status, path=path).observe(
        duration
    )
    REQUEST_IN_PROGRESS.labels(method=method, path=path).dec()

    return response


def update_system_metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.Process().memory_info().rss)
