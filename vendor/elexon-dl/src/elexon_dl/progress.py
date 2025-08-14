import asyncio, datetime as dt

class ProgressReporter:
    def __init__(self, http, interval_s: int = 15):
        self.http = http
        self.interval_s = interval_s
        self._task = None

    async def _run(self):
        print(f"{'Time':>8} | {'Reqs':>8} | {'Avg Lat':>8} | {'Max Lat':>8} | {'Thr/sec':>8} | {'Thr/min':>8}")
        print("-"*62)
        t0 = dt.datetime.now()
        while True:
            await asyncio.sleep(self.interval_s)
            snap = self.http.metrics.snapshot()
            elapsed = (dt.datetime.now() - t0).total_seconds() or 1.0
            rps = snap['count']/elapsed
            print(f"{dt.datetime.now().strftime('%H:%M:%S'):>8} | "
                  f"{snap['count']:8d} | "
                  f"{snap['avg_latency']:8.3f} | "
                  f"{snap['max_latency']:8.3f} | "
                  f"{rps:8.2f} | "
                  f"{rps*60:8.0f}")
    def start(self):
        self._task = asyncio.create_task(self._run())
        return self._task
    def stop(self):
        if self._task:
            self._task.cancel()
