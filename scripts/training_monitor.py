"""Observe CPU memory and optimizer-step times during a local training run."""

from threading import Event, Thread
from time import perf_counter

import psutil
from transformers import TrainerCallback


class TrainingMonitor(TrainerCallback):
    def __init__(self):
        self.process = psutil.Process()
        self.stop_event = Event()
        self.peak_rss_bytes = 0
        self.peak_private_bytes = 0
        self.minimum_available_bytes = psutil.virtual_memory().available
        self.step_times = []
        self.save_times = []
        self.step_started = None
        self.step_finished = None

    def sample(self):
        memory = self.process.memory_info()
        self.peak_rss_bytes = max(self.peak_rss_bytes, memory.rss)
        self.peak_private_bytes = max(self.peak_private_bytes, getattr(memory, "private", memory.vms))
        self.minimum_available_bytes = min(self.minimum_available_bytes, psutil.virtual_memory().available)

    def watch(self):
        while not self.stop_event.wait(0.25):
            self.sample()

    def start(self):
        self.sample()
        self.thread = Thread(target=self.watch, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_started = perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        self.step_finished = perf_counter()
        elapsed = self.step_finished - self.step_started
        self.step_times.append(elapsed)
        self.sample()
        print(f"Optimizer step {state.global_step}/{state.max_steps}: {elapsed:.2f}s", flush=True)

    def on_save(self, args, state, control, **kwargs):
        if self.step_finished is not None:
            self.save_times.append(perf_counter() - self.step_finished)

    def summary(self):
        return {
            "sampling_interval_seconds": 0.25,
            "peak_process_rss_bytes": self.peak_rss_bytes,
            "peak_process_private_bytes": self.peak_private_bytes,
            "minimum_system_available_bytes": self.minimum_available_bytes,
            "optimizer_step_seconds": self.step_times,
            "seconds_from_step_end_to_checkpoint_save": self.save_times,
        }
