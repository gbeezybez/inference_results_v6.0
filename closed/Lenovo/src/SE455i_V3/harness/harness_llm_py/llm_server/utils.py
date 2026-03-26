# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from functools import wraps
import inspect
import json
import os
from pathlib import Path
from statistics import mean, stdev
import threading
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from tqdm import tqdm

from code.common import logging
from code.common.utils import FastPercentileHeap


def add_prefix_logger(prefix_attr: str = None, verbose_attr: str = 'verbose'):
    """
    A decorator to add a prefix to all logging.info and logging.verbose calls within a class.

    Args:
        prefix_attr (str): The attribute name to use for the prefix. If None, the class name will be used.
        verbose_attr (str): The attribute name to determine if verbose logging is enabled.

    Returns:
        type: The decorated class.
    """

    class PrefixLogger:
        """
        A wrapper for the logger to add a prefix to log messages.

        Attributes:
            logger (logging.Logger): The original logger.
            prefix (str): The prefix to add to log messages.
            enable_verbose (bool): Flag to enable verbose logging.
        """

        def __init__(self, logger: logging.Logger, prefix: str, enable_verbose: bool):
            """
            Initialize the PrefixLogger.

            Args:
                logger (logging.Logger): The original logger.
                prefix (str): The prefix to add to log messages.
                enable_verbose (bool): Flag to enable verbose logging.
            """
            self.logger = logger
            self.prefix = prefix
            self.enable_verbose = enable_verbose

        def _get_callsite_str(self):
            frame = inspect.currentframe().f_back.f_back
            filename = Path(frame.f_code.co_filename).name
            line_no = frame.f_lineno
            func_name = frame.f_code.co_name
            return f' {filename}:{line_no} {func_name} '

        def info(self, msg: str, *args, **kwargs):
            callsite_str = self._get_callsite_str()
            self.logger.info(f" [ {self.prefix} ] [{callsite_str}] {msg}", *args, **kwargs)

        def verbose(self, msg: str, *args, **kwargs):
            if self.enable_verbose:
                callsite_str = self._get_callsite_str()
                self.logger.info(f" [ {self.prefix} ] [{callsite_str}] {msg}", *args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self.logger, name)

    def decorator(cls: type) -> type:
        original_init = cls.__init__

        @wraps(original_init)
        def __init__(self, *args, **kwargs):
            prefix = self.__class__.__name__ if prefix_attr is None else getattr(self, prefix_attr, kwargs.get(prefix_attr, None))
            assert prefix is not None, f"{cls.__name__} must have non-None (str) attribute / __init__ param: {prefix_attr}"

            verbose = getattr(self, verbose_attr, kwargs.get(verbose_attr, None))
            assert verbose is not None, f"{cls.__name__} must have non-None (bool) attribute / __init__ param: {verbose_attr}"

            # Setup logger before original __init__ is invoked
            self.logger = PrefixLogger(logging.getLogger(), prefix, verbose)

            # for self.logger, verbose logging is conditional on value of `verbose_attr` for wrapped object.
            # eg: this is only visible when object.verbose == True
            # self.logger.verbose("Enabled verbose logging.")

            original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return decorator


def track_latencies(func):
    """
    A decorator to track the latencies of the invoked function across threads.
    This tracks times at which each thread starts and ends the function.
    It will additionally log out latency metrics.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapped function with added tracking and logging.
    """
    state = {
        'lock': threading.Lock(),
        'start_times': {},
        'end_times': {}
    }

    @wraps(func)
    def wrapper(*args, **kwargs):
        thread_id = threading.get_ident()

        start_time = time.time()
        with state['lock']:
            state['start_times'][thread_id] = start_time

        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.time()
            with state['lock']:
                state['end_times'][thread_id] = end_time

                # NOTE(vir): ideally we pass in total threads as decorator parameter
                if len(state['start_times']) > 1 and len(state['end_times']) == len(state['start_times']):
                    start_times_list = list(state['start_times'].values())
                    end_times_list = list(state['end_times'].values())
                    latencies = [end - start for start, end in zip(start_times_list, end_times_list)]
                    tail_latency = max(end_times_list) - min(end_times_list)
                    stdev_val = 0 if len(latencies) <= 1 else stdev(latencies)

                    log_prefix = func.__name__
                    if hasattr(args[0], '__class__'):
                        log_prefix = f" [ {args[0].__class__.__name__}.{log_prefix} ] "

                    logging.info(f"{log_prefix} - {len(latencies)} Thread(s) Summary : "
                                 f"Mean Duration = {mean(latencies):.4f}s, "
                                 f"Tail Latency = {tail_latency:.4f}s, "
                                 f"Std Dev = {stdev_val:.4f}s")
        return result

    return wrapper


class LLMServerProgressDisplay:
    """
    A tqdm wrapper to display llm-server progress.

    Methods:
        update(completed=1):
            Updates the progress bar by a specified number of completed units.

        update_total(total):
            Updates the total number of units to process and refreshes the progress bar.

        finish():
            Closes the progress bar.
    """

    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        unit: str = "samples",
        enable: bool = True,
        additional_units: Dict[str, str] = {},
        log_dir: os.Pathlike = None
    ):
        """
        Initializes the LLMServerProgressDisplay instance.

        Args:
            total (int): The total number of units to process.
            desc (str): The prefix-description to display with the progress bar.
            unit (str): The unit of measurement for the progress bar.
            enable (bool): Flag to enable or disable the progress bar.
            additional_units Dict[str, str]: Additional units to track, name: type (where type=mean|value).
            log_dir (Optional[os.Pathlike]): Directory to store log files.
        """
        self.enabled = enable
        self.total = total
        self.desc = desc
        self.unit = unit
        self.additional_units_specs = additional_units
        self.log_dir = log_dir
        assert self.log_dir is not None

        self.progress_bar_args = {
            'total': self.total,
            'desc': self.desc,
            'unit': self.unit,
            'smoothing': 1,
            'mininterval': 0.20,
            'leave': True,
            'disable': not self.enabled,
            'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        }

        self.completed = 0
        self.additional_units_values = {}
        self.state_history = {}
        self.iteration_stats = []
        self.progress_bar = None  # lazy init in update_total
        self.lock = threading.Lock()

        for unit, _type in additional_units.items():
            match _type:
                case 'value' | 'mean':
                    self.additional_units_values[unit] = 0
                case '99%':
                    self.additional_units_values[unit] = FastPercentileHeap()
                case _:
                    raise ValueError(f"Unknown type({_type}) of additional unit({unit})")

        if self.enabled:
            self.stop_event = threading.Event()
            self.stats_flush_thread = threading.Thread(target=self.stats_periodic_flush)
            self.stats_flush_thread.start()

    def update(self, completed: int = 1, additional_unit_updates: Dict[str, int] = {}):
        """
        Updates the progress bar by a specified number of completed units.

        Args:
            completed (int): The number of units completed since the last update. Default is 1.
            additional Dict[str, int]: A dictionary of additional units completed since the last update.
        """
        if not self.enabled:
            return

        with self.lock:
            for unit, update in additional_unit_updates.items():
                match self.additional_units_specs[unit]:
                    case 'mean':
                        self.additional_units_values[unit] += update
                    case '99%':
                        self.additional_units_values[unit].extend(update)
                    case 'value':
                        self.additional_units_values[unit] = update

            displayed = completed > 0 and (self.progress_bar is not None and self.progress_bar.update(completed))
            self.completed += completed

        # NOTE(vir):
        # we do a non-blocking attempt to grab the lock and update stats display
        # whichever thread gets the lock will render a cumulative update
        if displayed and self.lock.acquire(blocking=False):
            displayed = False
            duration = max(self.progress_bar.format_dict['elapsed'], 1)
            n = self.progress_bar.format_dict['n']
            rate = 0 if self.progress_bar.format_dict['rate'] is None else float(self.progress_bar.format_dict['rate'])
            mean_rate = float(n / duration)

            additional_values = {}
            for unit, _type in self.additional_units_specs.items():
                match _type:
                    case 'value':
                        additional_values[unit] = float(self.additional_units_values[unit])
                    case 'mean':
                        additional_values[unit] = float(self.additional_units_values[unit] / duration)
                    case '99%':
                        if (top_p := self.additional_units_values[unit].p()) is not None:
                            additional_values[unit] = top_p

            samples_fmt = f'{rate:.2f}{self.unit}/s] Stats=[{mean_rate:.2f}{self.unit}/s'
            postfix_fmt = ''.join([f', {value:.2f}{unit}' for unit, value in additional_values.items()])

            self.progress_bar.set_postfix_str(f'{samples_fmt}{postfix_fmt}', refresh=True)
            self.state_history[n] = additional_values
            self.lock.release()

    def update_total(self, total: int):
        """
        Updates the total number of units to process and refreshes the progress bar.

        Args:
            total (int): The new total number of units to process.
        """
        if not self.enabled:
            return

        with self.lock:
            if self.progress_bar is None:
                self.progress_bar = tqdm(**self.progress_bar_args)
                self.progress_bar.n = self.completed

            self.total = self.progress_bar.total = total

    def record_iteration_stats(self, stats: List[Dict]):
        """
        Records the iteration statistics.
        Saved to self.log_dir on finish.

        Args:
            stats (List[Dict]): A list of dictionaries containing iteration statistics.
        """
        if not self.enabled:
            return

        with self.lock:
            self.iteration_stats.extend(stats)

    def finish(self):
        """
        Completes and freezes the progress bar.
        """
        if not self.enabled:
            return

        if not self.progress_bar.disable:
            self.progress_bar.close()

        self.stop_event.set()
        self.stats_flush_thread.join()
        self.enabled = False

        if len(self.state_history) > 0:
            stats_dump = Path(self.log_dir) / "harness_stats.log"
            stats_plot = Path(self.log_dir) / "harness_stats_timeline.png"

            # dump harness stats log
            stats_dump.write_text(json.dumps(self.state_history, separators=(', ', ':')))
            logging.info(f"Harness stats saved to: {stats_dump}")

            completed = list(self.state_history.keys())
            additional_stats_keys = self.additional_units_specs.keys()

            fig, axs = plt.subplots(len(additional_stats_keys), 1, figsize=(10, 5 * len(additional_stats_keys)))
            for i, key in enumerate(additional_stats_keys):
                type_str = self.additional_units_specs[key]
                values = [self.state_history[n][key] for n in completed]
                axs[i].plot(completed, values, label=key)
                axs[i].set_title(f'[{type_str}] {key}')
                axs[i].set_xlabel('Completed')
                axs[i].set_ylabel(key)
                axs[i].legend()

            # plot and dump harness stats over-time
            plt.tight_layout()
            plt.savefig(stats_plot)
            logging.info(f"Harness timeline plot generated to: {stats_plot}")

            plt.close()

    def stats_periodic_flush(self):
        """
        Periodically flushes accumulated iteration stats to log file.
        """
        stats_file = Path(self.log_dir) / 'harness_iteration_stats.log'
        logging.info(f"Harness iteration stats will be dumped to: {stats_file}")

        with open(stats_file, 'w') as f:
            while not self.stop_event.is_set():
                with self.lock:
                    stats = [json.dumps(stats, separators=(', ', ':')) + '\n' for stats in self.iteration_stats]
                    self.iteration_stats.clear()

                f.writelines(stats)
                f.flush()
                time.sleep(1)

        logging.info(f"Harness iteration stats dumped to: {stats_file}")
