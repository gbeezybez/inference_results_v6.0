# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
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

import glob
import os
from datetime import datetime
from pathlib import Path

from code.common import logging, run_command
from code.common.log_parser import from_loadgen_by_keys
from code.internal.tegrastats_parser import TegraStatsParser


class PowerMeasurements:
    def __init__(self, log_file, interval=1, duration=3600):
        self.log_file = log_file
        self.interval = interval
        self.duration = duration
        self.platform = self.get_platform()

    def start(self):
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        start_commands = self.get_start_commands()
        self.run(start_commands)

    def stop(self):
        process_names = self.get_process_names()
        self.kill_process(process_names)

    def report_stats(self, log_dir):
        if self.platform == "aarch64":
            print("======================= Tegrastats Stats =======================")
            log_paths = glob.glob(os.path.join(log_dir, "**", "mlperf_log_detail.txt"), recursive=True)
            if len(log_paths) > 1:
                print(f"More than one result found under {log_dir}, Skipping the tegrastats stats...")
            elif len(log_paths) == 0:
                print(f"Cannot find any result found under {log_dir}, Skipping the tegrastats stats...")
            else:
                log_path = log_paths[0]
                window_timestamps = from_loadgen_by_keys(os.path.dirname(log_path), ["power_begin", "power_end"])
                start_timestamp = datetime.strptime(window_timestamps["power_begin"].split(".")[0], "%m-%d-%Y %H:%M:%S")
                end_timestamp = datetime.strptime(window_timestamps["power_end"].split(".")[0], "%m-%d-%Y %H:%M:%S")

                tegrastats_parser = TegraStatsParser()
                tegrastats_parser.parse_with_time(
                    "{}_tegrastats.log".format(self.log_file),
                    "{}_tegrastats.csv".format(self.log_file),
                    [start_timestamp, end_timestamp]
                )

    def get_process_names(self):
        if self.platform == "x86_64":
            return []
        elif self.platform == "aarch64":
            return ["powersig", "tegrastats"]
        else:
            return self.handle_unknown_platform()

    def handle_unknown_platform(self):
        logging.error("Unknown platform {}.".format(self.platform))
        raise ("Unknown platform {}.".format(self.platform))

    def get_start_commands(self):
        if self.platform == "x86_64":
            return self.get_desktop_start_commands()
        elif self.platform == "aarch64":
            return self.get_mobile_start_commands()
        else:
            return self.handle_unknown_platform()

    def get_desktop_start_commands(self):
        return ""

    def get_mobile_start_commands(self):
        return [
            # explicitly use UTC0 to set time zone to align with loadgen
            "sudo tegrastats --interval 100 | TZ='UTC0' ts '%F %T' > {}_tegrastats.log &".format(
                self.log_file
            ),
            "sudo /opt/tools/powersig/bin/powersig --csv --time {} --enable_stats > {}_powersig.log &".format(
                self.duration, self.log_file
            )
        ]

    def get_platform(self):
        platform = run_command("uname -p", get_output=True, tee=False)
        platform = "".join(platform)
        return platform

    def run(self, commands):
        if isinstance(commands, str):
            commands = [commands]
        for command in commands:
            run_command(command, tee=False)

    def kill_process(self, process_names):
        ps_output = run_command("ps -ax", get_output=True, tee=False)
        for line in ps_output:
            for process_name in process_names:
                if process_name in line:
                    pid = int(line.split(None, 1)[0])
                    # os.kill(pid, signal.SIGUSR1)
                    os.system("sudo kill --signal SIGUSR1 {}".format(pid))
