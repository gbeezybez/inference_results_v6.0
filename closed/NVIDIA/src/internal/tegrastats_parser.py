#!/home/tools/continuum/Anaconda3-2020.07/bin/python
import re
from datetime import datetime
import numpy as np


class TegraStatsParser():
    # For detail, please look at https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/AppendixTegraStats.html#
    # It is still for Xavier, but some of the voltage rails are similar.
    GPU_SOC_RE = re.compile("VDD_GPU_SOC (\d+)")  # GPU_SOC power
    CPU_CV_RE = re.compile("VDD_CPU_CV (\d+)")  # CPU_CV (DLA) power
    VIN_SYS_RE = re.compile("VIN_SYS_5V0 (\d+)")  # VIN_SYS is power for 5V rail feed to DRAM and other voltage regulators on chip.
    VDDQ_RE = re.compile("VDDQ_VDD2_1V8AO (\d+)")  # it is power for 1.8V rail, but these power are already captured by CPU/SOC/GPU/CV so we will not add this to AP_DRAM power
    GR3D_RE = re.compile("GR3D_FREQ (\d+)%@(\d+)")  # GPU freq and util
    CPU_FREQ_RE = re.compile("CPU \[.*] ")  # GPU freq and util per core
    GR3D2_RE = re.compile("GR3D2_FREQ (\d+)%@(\d+)")  # GPU freq and util
    NVDLA0_RE = re.compile("NVDLA0_FREQ @(\d+)")  # NVDLA freq
    NVDLA1_RE = re.compile("NVDLA1_FREQ @(\d+)")  # NVDLA freq
    EMC_FREQ_RE = re.compile("EMC_FREQ (\d+)%@(\d+)")  # Memory freq and util

    def mul_perc(self, pattern, line):
        match = pattern.search(line)
        f = int(match.group(1)) / float(100) * int(match.group(2))
        return int(f)

    def dla_pat(self, pattern, line):
        match = pattern.search(line)
        return match.group(1) if match else 0

    def cpu_pat(self, line):
        match = self.CPU_FREQ_RE.search(line)
        cpu_part = match.group(0).split()[1]
        num_core = 0
        core_util = 0
        core_freq = 0
        for core in cpu_part.split(","):
            if "off" in core:
                pass
            else:
                num_core += 1
                core_util += int(core.split("%@")[0].replace("[", ""))
                core_freq += int(core.split("%@")[1].replace("]", ""))

        core_util = core_util / 12
        core_freq = core_freq / num_core

        return core_util, core_freq

    def get_start_point(self, arr):
        '''
        from the arr, find the index where the main harness operation starts (ignore warmup)
        '''
        window_size = max(int(len(arr) / 10), 1)
        SMA = np.convolve(arr, np.ones(window_size), 'valid') / window_size
        max_slope = 0
        max_i = 0
        for i in range(len(SMA) - 1):
            if SMA[i + 1] - SMA[i] > max_slope:
                max_slope = SMA[i + 1] - SMA[i]
                max_i = i
        return int(max_i + window_size / 2)

    def parse_with_time(self, fin_name, fout_name, time_region):
        import matplotlib.pyplot as plt
        gpu_freq = []
        dla_freq = []
        cpu_util = []
        cpu_freq = []
        gpu_soc_pwr = []
        cpu_cv_pwr = []
        vin_sys_pwr = []
        vddq_pwr = []

        with open(fin_name, 'r') as fin, open(fout_name, 'w') as fout:
            fout.write("Time, Pwr_Module(mW), Pwr_GPU_SOC (mW), Pwr_CPU_CV (mW), Pwr_SYS_5VO (mW), Pwr_VDDQ_VDD2_1V8AO (mW), GR3D_FREQ (MHz), GR3D2_FREQ (MHz),EMC_FREQ (MHz), NVDLA0_FREQ (MHz), NVDLA1_FREQ (MHz)\n")
            lines = fin.readlines()
            for line in lines:
                time = datetime.strptime(" ".join(line.split(" ")[0:2]), "%Y-%m-%d %H:%M:%S")
                if (time >= time_region[0] and time <= time_region[1]):
                    gpu_mW = int(self.GPU_SOC_RE.search(line).group(1))
                    cpu_mW = int(self.CPU_CV_RE.search(line).group(1))
                    vin_mW = int(self.VIN_SYS_RE.search(line).group(1))
                    vddq_mW = int(self.VDDQ_RE.search(line).group(1))
                    gr3d_freq = int(self.mul_perc(self.GR3D_RE, line))
                    gr3d2_freq = int(self.mul_perc(self.GR3D2_RE, line))
                    emc_freq = int(self.mul_perc(self.EMC_FREQ_RE, line))
                    dla0_freq = int(self.dla_pat(self.NVDLA0_RE, line))
                    dla1_freq = int(self.dla_pat(self.NVDLA1_RE, line))
                    cpu_core_util, cpu_core_freq = self.cpu_pat(line)
                    fout.write(f"{time}, {gpu_mW+cpu_mW+vin_mW}, {gpu_mW},{cpu_mW},{vin_mW},{vddq_mW},{gr3d_freq},{gr3d2_freq},{emc_freq},{dla0_freq},{dla1_freq}\n")
                    gpu_freq.append(gr3d_freq)
                    dla_freq.append(dla0_freq)
                    cpu_util.append(cpu_core_util)
                    cpu_freq.append(cpu_core_freq)
                    gpu_soc_pwr.append(gpu_mW)
                    cpu_cv_pwr.append(cpu_mW)
                    vin_sys_pwr.append(vin_mW)
                    vddq_pwr.append(vddq_mW)

                if time > time_region[1]:
                    break

            ap_pwr = np.array(gpu_soc_pwr) + np.array(cpu_cv_pwr) + np.array(vin_sys_pwr)
            runtime_end_point = len(ap_pwr)
            test_config = fout_name.replace(".csv", "")
            # Find first shoot up
            start_point = self.get_start_point(ap_pwr)
            end_point = runtime_end_point

            plt.figure(figsize=(10, 6))
            plt.plot(ap_pwr)
            plt.axvspan(start_point, end_point, color='pink', alpha=0.5)
            png_fname = f"{test_config}_power.png"
            print(png_fname)
            plt.savefig(png_fname)
            plt.close()

            ap_pwr_window = ap_pwr[start_point:end_point]
            gpu_soc_window = np.array(gpu_soc_pwr)[start_point:end_point]
            cpu_cv_window = np.array(cpu_cv_pwr)[start_point:end_point]
            vin_sys_window = np.array(vin_sys_pwr)[start_point:end_point]
            dla_freq_window = np.array(dla_freq)[start_point:end_point]
            cpu_core_util_window = np.array(cpu_util)[start_point:end_point]
            cpu_freq_window = np.array(cpu_freq)[start_point:end_point]

            # print("AP+DRAM max pwr = ", np.max(ap_pwr_window))
            print("AP+DRAM avg pwr = ", np.mean(ap_pwr_window))
            print("GPU_SOC avg pwr = ", np.mean(gpu_soc_window))
            print("CPU_CV avg pwr = ", np.mean(cpu_cv_window))
            print("VIN_SYS_5V0 avg pwr = ", np.mean(vin_sys_window))

            module_avg_power = np.mean(ap_pwr_window)
            gpu_soc_avg_power = np.mean(gpu_soc_window)
            cpu_cv_avg_power = np.mean(cpu_cv_window)
            vin_sys_avg_power = np.mean(vin_sys_window)
            dla_avg_freq = np.mean(dla_freq_window)
            dla_avg_freq = np.mean(dla_freq_window)
            cpu_max_freq = np.max(cpu_freq_window)
            cpu_avg_util = np.mean(cpu_core_util_window)

        return module_avg_power, gpu_soc_avg_power, cpu_cv_avg_power, vin_sys_avg_power, dla_avg_freq, cpu_max_freq, cpu_avg_util

    def parse(self, fin_name, fout_name, auto_capture=False):

        if auto_capture:
            gpu_freq = []
            dla_freq = []
            cpu_util = []
            cpu_freq = []
            gpu_soc_pwr = []
            cpu_cv_pwr = []
            vin_sys_pwr = []
            vddq_pwr = []

        with open(fin_name, 'r') as fin, open(fout_name, 'w') as fout:
            lines = fin.readlines()
            fout.write("Pwr_GPU_SOC (mW), Pwr_CPU_CV (mW), Pwr_SYS_5VO (mW), Pwr_VDDQ_VDD2_1V8AO (mW), GR3D_FREQ (MHz), GR3D2_FREQ (MHz),EMC_FREQ (MHz), NVDLA0_FREQ (MHz), NVDLA1_FREQ (MHz)\n")
            for line in lines:
                gpu_mW = int(self.GPU_SOC_RE.search(line).group(1))
                cpu_mW = int(self.CPU_CV_RE.search(line).group(1))
                vin_mW = int(self.VIN_SYS_RE.search(line).group(1))
                vddq_mW = int(self.VDDQ_RE.search(line).group(1))
                gr3d_freq = int(self.mul_perc(self.GR3D_RE, line))
                gr3d2_freq = int(self.mul_perc(self.GR3D2_RE, line))
                emc_freq = int(self.mul_perc(self.EMC_FREQ_RE, line))
                dla0_freq = int(self.dla_pat(self.NVDLA0_RE, line))
                dla1_freq = int(self.dla_pat(self.NVDLA1_RE, line))
                cpu_core_util, cpu_core_freq = self.cpu_pat(line)

                fout.write(f"{gpu_mW},{cpu_mW},{vin_mW},{vddq_mW},{gr3d_freq},{gr3d2_freq},{emc_freq},{dla0_freq},{dla1_freq}\n")
                if auto_capture:
                    gpu_freq.append(gr3d_freq)
                    dla_freq.append(dla0_freq)
                    cpu_util.append(cpu_core_util)
                    cpu_freq.append(cpu_core_freq)
                    gpu_soc_pwr.append(gpu_mW)
                    cpu_cv_pwr.append(cpu_mW)
                    vin_sys_pwr.append(vin_mW)
                    vddq_pwr.append(vddq_mW)

        if auto_capture:
            import matplotlib.pyplot as plt
            # model = rpt.Window(model="l2", jump=1, width=10)
            ap_pwr = np.array(gpu_soc_pwr) + np.array(cpu_cv_pwr) + np.array(vin_sys_pwr)
            # algo = model.fit(ap_pwr)
            # exec_range = algo.predict(n_bkps=2)
            runtime_end_point = len(ap_pwr)
            test_config = fout_name.replace(".csv", "")

            # Find first shoot up
            start_point = self.get_start_point(ap_pwr)
            end_point = runtime_end_point

            #fig = rpt.display(ap_pwr, [start_point, end_point], figsize=(10,6))
            plt.figure(figsize=(10, 6))
            plt.plot(ap_pwr)
            plt.axvspan(start_point, end_point, color='pink', alpha=0.5)
            png_fname = f"{test_config}_power.png"
            print(png_fname)
            plt.savefig(png_fname)

            ap_pwr_window = ap_pwr[start_point:end_point]
            gpu_soc_window = np.array(gpu_soc_pwr)[start_point:end_point]
            cpu_cv_window = np.array(cpu_cv_pwr)[start_point:end_point]
            vin_sys_window = np.array(vin_sys_pwr)[start_point:end_point]
            dla_freq_window = np.array(dla_freq)[start_point:end_point]
            cpu_core_util_window = np.array(cpu_util)[start_point:end_point]
            #print("AP+DRAM max pwr = ", np.max(ap_pwr_window))
            print("AP+DRAM avg pwr = ", np.mean(ap_pwr_window))
            print("GPU_SOC avg pwr = ", np.mean(gpu_soc_window))
            print("CPU_CV avg pwr = ", np.mean(cpu_cv_window))
            print("VIN_SYS_5V0 avg pwr = ", np.mean(vin_sys_window))

            module_avg_power = np.mean(ap_pwr_window)
            gpu_soc_avg_power = np.mean(gpu_soc_window)
            cpu_cv_avg_power = np.mean(cpu_cv_window)
            vin_sys_avg_power = np.mean(vin_sys_window)
            dla_avg_freq = np.mean(dla_freq_window)
            cpu_avg_util = np.mean(cpu_core_util_window)

            return module_avg_power, gpu_soc_avg_power, cpu_cv_avg_power, vin_sys_avg_power, dla_avg_freq, cpu_avg_util
