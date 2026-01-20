#!/usr/local/bin/ipython -i                     # 指定使用 IPython 交互式解释器执行该脚本

# 导入 Mozaik 框架中用于实验的模块
from mozaik.experiments import *               # 导入所有实验相关的基础模块
from mozaik.experiments.vision import *          # 导入视觉实验相关模块
from mozaik.experiments.optogenetic import SingleOptogeneticArrayStimulus  # 导入单一光遗传刺激器的类
from mozaik.experiments.closed_loop import ClosedLoopOptogeneticStimulation  # 导入闭环光遗传刺激实验类

# 导入用于生成简单形状二值掩码的函数，用于定义刺激区域
from mozaik.sheets.direct_stimulator import simple_shapes_binary_mask

# 导入参数设置工具
from parameters import ParameterSet             # 导入参数集合类
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet  # 导入扩展参数集合类

# 导入 Python 内置的 SimpleNamespace 用于动态创建对象属性
from types import SimpleNamespace

from scipy.signal import medfilt
from scipy.linalg import solve_discrete_are

# 导入 numpy 库，用于数值计算
import numpy as np
import pandas as pd 
import math

class _NamedStaticMethod:
    def __init__(self, func, name):
        self.func = func
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

def named_static(name):
    def decorator(func):
        return staticmethod(_NamedStaticMethod(func, name))
    return decorator

# 定义闭环调控器的设置类
class RegulatorSetup:
    stim_circle_radius = 300  # um，定义刺激圆的半径
    radius_sub = 200
    centers = [(-200,200), (200,-200)]

    a = -0.483

    @staticmethod
    def append_history(state, update_dict):
        # 如果状态中还没有 history 属性，则初始化一个字典，
        # 包含 "LMS"、"error"、"Firing rate" 等关键指标，每个键对应一个空列表
        if state.history is None:
            names = ["LMS", "error", "Firing rate", "Current rate", "Spike Timestamp", "Target"]
            state.history = {n: [] for n in names}
        for k in update_dict:
            state.history[k].append(update_dict[k])
            
    @staticmethod
    def biophysical_converter(y):
        B, C, G = 1.57, 1.0, 4.11
        return 0.942057 * ((1 - 0.942057 * y) ** (-1.0 / G) - 1.0) ** (1.0 / B)

    @named_static("'RegulatorSetup.calculate_input'")
    def calculate_input(regulator):
        """
        使用 LMS 算法来计算下一次刺激输入：
         - 利用最近 lms_window 个误差构成回归向量 X；
         - 控制信号 u = W^T * X；
         - 权重更新：W = W + mu * error * X；
         - 将 u 限制在 [0,1] 内。
        """
        if regulator.state is None:
            # 生成目标放电率分段信号（单位 ms -> s）
            duration = 18000  # 总时长（ms）
            t = np.arange(duration) / 1000.0  # 时间向量，单位秒
            
            
            target_signal = 35 + 20 * np.sin(2 * np.pi * 1 * t - np.pi/2)
            # 修改：在 5.5-10.5 s 和 15.5-20.5 s 之间保持恒定为 14
            target_signal[(t >= 6.25) & (t < 9.25)] = 35
            target_signal[(t >= 12.25) & (t < 15.25)] = 35

            # 采样步长（s）
            dt = float(regulator.parameters.state_update_interval) / 1000.0
            # 近似模型增益：误差对控制的灵敏度（可调）
            g = 0.6  # 建议范围 [0.2, 1.5]，越大代表“u 对误差削减”越强
            # 系统矩阵（离散）
            A = np.array([[1.0, 0.0],
                        [dt,  1.0]], dtype=float)
            B = np.array([[-g],
                        [ 0.0]], dtype=float)
            # LQR 权重（可根据性能/能耗再调）
            Q = np.diag([1.0, 0.05])  # 惩罚 e 和 z（偏重 e，适度惩罚积分飘移）
            R = np.array([[0.2]])     # 惩罚 u（越大 -> 控制更保守、能耗更低）
            # 求解 DARE
            P = solve_discrete_are(A, B, Q, R)
            K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)  # 形状 (1,2)
            
            regulator.state = SimpleNamespace(
            	kp=0.05,            # 比例系数
                ki=0.000005,         # 积分系数
                kd=0.005,             # 微分系数
                integral=0,          # 初始积分值
                previous_error=0,    # 上一次误差（用于微分项计算）
                mu=0.00003,             # LMS 学习率
                lms_window=3,          # LMS窗口长度（采样点数）
                lms_weights=np.ones(3) / 3,  # 初始权重向量
                lms_error_history=[],  # 用于保存最近 lms_window 个采样点的误差
                error=0,               # 当前误差，由 update_state 更新
                history=None,          # 历史记录初始化为空
                control_signal=0.0,    # 初始控制信号
                target_signal=target_signal,
                export_dir=mozaik.controller.Global.root_directory, ### Saving into results directory
                #norm_signal = norm_signal
                lqr_A=A, lqr_B=B, lqr_Q=Q, lqr_R=R, lqr_P=P, lqr_K=K,lqr_dt=dt, lqr_g=g,    # LQR 参数与缓存
                x=np.zeros(2, dtype=float),# 状态变量：误差 e、误差积分 z
            )
        s = regulator.state
        # 当前误差由 update_state 更新，此处直接使用 s.error
        t_current = regulator.current_time() / 1000.0

        if (t_current < 3):
            s.control_signal = 0.0
            current_error = s.error
        else:
            current_error = s.error
            
            proportional = s.kp * s.error  # 计算比例项 P：当前误差乘以比例系数
	    # 更新积分项：将当前误差乘以状态更新间隔累加到积分中
            s.integral += s.error * regulator.parameters.state_update_interval
            integral = s.ki * s.integral  # 计算积分项 I：积分乘以积分系数
            integral = min(0.2, max(integral, -0.2))
            # 计算微分项 D：当前误差与前一次误差之差除以状态更新间隔，再乘以微分系数
            derivative = s.kd * (s.error - s.previous_error) / regulator.parameters.state_update_interval
            s.previous_error = s.error   # 将当前误差保存为前一误差，便于下一次更新时使用
            u = proportional + integral + derivative  # PID 控制信号：P + I + D
            u = min(1, max(u, 0))

            # 更新 LMS 的误差历史
            #s.lms_error_history.append(current_error)
            #if len(s.lms_error_history) > s.lms_window:
            #    s.lms_error_history.pop(0)
            # 若不足窗口长度，前面补0
            #X = np.array(s.lms_error_history)
            #if len(X) < s.lms_window:
            #    X = np.concatenate((np.zeros(s.lms_window - len(X)), X))
        
            # 计算控制信号： u = W^T * X
            #u = np.dot(s.lms_weights, X)
            # 更新 LMS 权重：W = W + mu * error * X
            #s.lms_weights = s.lms_weights + s.mu * current_error * X
        
            #u = 0.5 + (1 / 10) * np.log(u / (1 - u))
        
            #u = max(0, min(u, 1))

            # e 由 update_state() 已更新：e = target - smoothed_rate
            #e = float(s.error)
            # 误差积分（离散积分）
            #s.x[1] += s.lqr_dt * e     # z_{k+1} = z_k + dt * e_k
            #s.x[0]  = e                # 更新 e_k
            # LQR 控制律
            #u = float(-s.lqr_K @ s.x)
            # 限幅 &（可选）软化
            #u = max(0.0, min(1.0, u))
        
            #u_prime = RegulatorSetup.biophysical_converter(u)
            #u_prime = max(0, min(u_prime, 1))
        
            s.control_signal = u
        
        # control_signal_scaled = 0
        
        #t_current = regulator.current_time() / 1000.0
        # 计算开环控制信号：频率 0.5Hz，初始相位 -π/2，范围 [0,1]
        #s.control_signal = 0.5 + 0.5 * np.sin(2 * np.pi * 1 * t_current - np.pi/2)
        # 在指定时段内将控制信号置 0
        #if (t_current < 3):
        #    s.control_signal = 0.0
        #if (6.25 <= t_current < 9.25):
        #    s.control_signal = 0.5
        #if (12.25 <= t_current < 15.25):
        #    s.control_signal = 0.5
        
        #s.control_signal = 1.0
        
        #t_current = regulator.current_time() / 1000.0
        #if (t_current < 0.5):
        #    s.control_signal = 0.0
        #else:
        #    s.control_signal = 0.0
        
        #t_current = regulator.current_time() / 1000.0
        #cycle_duration = 2.0
        #full_cycle = 10 * cycle_duration
        #phase = t_current % full_cycle
        #step = int(phase//cycle_duration)
        #if (phase % cycle_duration) > 1.0:
        #    s.control_signal = 0.1 * (step + 1)
        #else:
        #    s.control_signal = 0.0
        
        # 根据当前时间索引，取出预先计算好的 norm_signal
        #idx = int(regulator.current_time() // regulator.parameters.update_interval) - 1
        # 防止越界
        #idx = max(0, min(idx, len(s.norm_signal) - 1))
        #s.control_signal = s.norm_signal[idx]

        # 记录 LMS 输出和当前误差到历史中
        RegulatorSetup.append_history(s, {"LMS": s.control_signal})
        
        # 生成刺激输入信号：使用 simple_shapes_binary_mask 生成圆形掩码
        circular_mask = simple_shapes_binary_mask(
            regulator.stimulator_coords_x,
            regulator.stimulator_coords_y,
            'circle',
            ParameterSet({'coords': [0, 0], 'radius': RegulatorSetup.stim_circle_radius})
        )

        #masks = []
        #for (cx, cy) in RegulatorSetup.centers:
        #    masks.append(simple_shapes_binary_mask(
        #        regulator.stimulator_coords_x,
        #    regulator.stimulator_coords_y,
        #    'circle',
        #    ParameterSet({'coords': [cx, cy], 'radius': RegulatorSetup.radius_sub})
        #    ))
        #circular_mask = np.clip(masks[0] + masks[1], 0, 1)

        input_signal = circular_mask * s.control_signal * np.ones((
            regulator.parameters.state_update_interval // regulator.parameters.update_interval,
            len(regulator.stimulator_coords_x),
            len(regulator.stimulator_coords_y)
        ))
        print(f"LMS control signal: {s.control_signal:.4f}")
        return input_signal.transpose((1, 2, 0))

    @named_static("'RegulatorSetup.update_state'")    
    def update_state(regulator):
        """
        使用 EWMA 方法估计神经元发放率，并计算当前误差：
            error = target - smoothed_rate
        同时确保同时更新“Firing rate”和“error”的历史记录。
        """
        # 获取最近 state_update_interval 内的脉冲记录
        #last_spiketrains = regulator.get_recording(
        #    "spikes",
        #    t_start=regulator.current_time() - regulator.parameters.state_update_interval,
        #    t_stop=regulator.current_time()
        #)
        
        # 判断哪些神经元位于以 (0,0) 为中心、半径为 stim_circle_radius 的圆内
        in_circle_mask = np.sqrt(np.sum(np.array(regulator.recorded_neuron_positions()) ** 2, axis=0)) < RegulatorSetup.stim_circle_radius

        # 处理Spike Timestamp
        if regulator.sheet.model.parameters.trial == 0:
            instantaneous_rates = []
            for i in range(len(regulator.last_spike_counts)):
                if in_circle_mask[i]:
                    rate = regulator.last_spike_counts[i] / (regulator.parameters.state_update_interval / 1000.0)
                    instantaneous_rates.append(rate)
            current_rate = np.mean(instantaneous_rates) if instantaneous_rates else 0.0
        else:
            positions = np.array(regulator.recorded_neuron_positions())
            in_circle_mask = np.zeros(positions.shape[1], dtype=bool)
            for (cx, cy) in RegulatorSetup.centers:
                d2 = (positions[0] - cx)**2 + (positions[1] - cy)**2
                in_circle_mask |= (d2 < RegulatorSetup.radius_sub**2)

            instantaneous_rates = []
            for i, spike_count in enumerate(regulator.last_spike_counts):
                if in_circle_mask[i]:
                    rate = spike_count / (regulator.parameters.state_update_interval / 1000.0)
                    instantaneous_rates.append(rate)
            current_rate = np.mean(instantaneous_rates) if instantaneous_rates else 0.0

        #positions = np.array(regulator.recorded_neuron_positions())
        #in_circle_mask = np.zeros(positions.shape[1], dtype=bool)
        #for (cx, cy) in RegulatorSetup.centers:
        #    d2 = (positions[0] - cx)**2 + (positions[1] - cy)**2
        #    in_circle_mask |= (d2 < RegulatorSetup.radius_sub**2)

        #instantaneous_rates = []
        #for i, spiketrain in enumerate(last_spiketrains):
        #    if in_circle_mask[i]:
        #        rate = len(spiketrain) / (regulator.parameters.state_update_interval / 1000.0)
        #        instantaneous_rates.append(rate)
        #current_rate = np.mean(instantaneous_rates) if instantaneous_rates else 0.0
        
        RegulatorSetup.append_history(regulator.state, {"Current rate": current_rate})

        # 使用 EWMA 平滑：alpha = 0.1
        alpha = 0.1
        if not hasattr(regulator.state, 'smoothed_rate'):
            regulator.state.smoothed_rate = current_rate
        else:
            regulator.state.smoothed_rate = alpha * current_rate + (1 - alpha) * regulator.state.smoothed_rate

        # --- 因果高斯平滑（σ=20 ms, 单边核，实时） ---
        # sigma_ms = 20.0
        # bin_ms = float(regulator.parameters.state_update_interval)  # 每次更新的时间步(ms)

        # # 首次调用时预生成核与缓冲
        # if not hasattr(regulator.state, '_causal_gauss'):
        #     sigma_bins = max(1.0, sigma_ms / bin_ms)
        #     truncate = 3.0                                   # 核长 ~ 3σ（可改为 4.0 更平滑但延迟略增）
        #     M = int(np.ceil(truncate * sigma_bins))          # 核长度（以 bins 为单位）
        #     t_bins = np.arange(0, M + 1, dtype=float)        # 只保留 t>=0 的单边核
        #     k = np.exp(-0.5 * (t_bins / sigma_bins) ** 2)
        #     k /= k.sum()                                     # 归一化，单位增益
        #     # 用一个长度 M+1 的环形/移位缓冲保存最近的 M+1 个 rate（0 位置为最新值）
        #     regulator.state._causal_gauss = SimpleNamespace(
        #         k=k, M=M, buf=np.zeros(M + 1, dtype=float)
        #     )

        # cg = regulator.state._causal_gauss
        # # 右移一格，最新样本放在 buf[0]
        # cg.buf[1:] = cg.buf[:-1]
        # cg.buf[0] = float(current_rate)

        # # 平滑输出：单边卷积（因果）
        # regulator.state.smoothed_rate = float(np.dot(cg.k, cg.buf))

        
        # Butterworth LPF参数设置
        #cutoff_frequency = 3  # 截止频率 (Hz)，可根据实际需求调整
        #sampling_interval = regulator.parameters.state_update_interval / 1000.0  # 转换为秒
        #a = math.exp(-2 * math.pi * cutoff_frequency * sampling_interval)
        
        # Butterworth LPF实现
        #if not hasattr(regulator.state, 'smoothed_rate'):
        #    regulator.state.smoothed_rate = current_rate
        #else:
        #    regulator.state.smoothed_rate = (1 - a) * current_rate + a * regulator.state.smoothed_rate
        
        # —— 用手写中值滤波平滑发放率 —— 
        #s = regulator.state
        #rate_history = s.history["Current rate"]
        #window = 5
        #if len(rate_history) >= window:
            # 只取最后 window 个采样点，算它们的中位数
        #    s.smoothed_rate = float(np.median(rate_history[-window:]))
        #else:
            # 数据还不够多，就直接用当前值
        #    s.smoothed_rate = current_rate    

        # 根据当前时间更新目标信号
        #index = int(regulator.current_time() // regulator.parameters.update_interval) - 1
        #if index < len(regulator.state.target_signal):
        #    target = regulator.state.target_signal[index]
        #else:
        #    target = regulator.state.target_signal[-1]
        
        t = regulator.current_time()
        fs = 1.0 / regulator.parameters.update_interval
        idx = int(t * fs)
        idx = max(0, min(idx, len(regulator.state.target_signal)-1))
        target = regulator.state.target_signal[idx]
        
        regulator.state.error = target - regulator.state.smoothed_rate

        print("Current instantaneous rate: %.3f, target: %.3f, Smoothed rate: %.3f, error: %.3f" %
              (current_rate, target, regulator.state.smoothed_rate, regulator.state.error))

        # 同时更新"Firing rate"和"error"的历史记录
        RegulatorSetup.append_history(regulator.state, {
            "Firing rate": regulator.state.smoothed_rate,
            "error": regulator.state.error,
            "Target": target
            #"Spike Timestamp": spike_timestamps
        })

        #RegulatorSetup.plot_history(regulator)
        #RegulatorSetup.plot_orientations_with_stimulus(regulator)

    @staticmethod 
    def plot_orientations_with_stimulus(regulator):
        recorded_x, recorded_y = regulator.recorded_neuron_positions()
        ors = regulator.recorded_neuron_orientations()
        stim_x, stim_y = regulator.stimulator_coords_x, regulator.stimulator_coords_y
        input_signal_nonzeros = (regulator.input_signal.mean(axis=-1) > 0).astype(float).flatten()
        import pylab
        fig = pylab.figure(figsize=(20, 20))
        fontsize = 50
        pylab.axis('equal')
        im = pylab.scatter(recorded_x, recorded_y, c=ors, cmap='hsv', vmin=0, vmax=np.pi, s=400)
        led_size = (regulator.parameters.spacing * 72 / fig.dpi) ** 2  
        pylab.scatter(stim_x.flatten(), stim_y.flatten(), alpha=0.4 * input_signal_nonzeros, color='k', marker='s', s=led_size)
        pylab.xlabel("x (um)", fontsize=fontsize)
        pylab.ylabel("y (um)", fontsize=fontsize)
        pylab.xticks(fontsize=fontsize)
        pylab.yticks(fontsize=fontsize)
        cbar = pylab.colorbar(im, aspect=17, ax=pylab.gca(), fraction=0.0527)
        cbar.set_label(label='Orientation', labelpad=-10, fontsize=fontsize)
        cbar.set_ticks([0, np.pi], labels=["0", "$\pi$"], fontsize=fontsize)
        fig.savefig("or_map.png")
        pylab.close()

    @staticmethod 
    def plot_history(regulator):
        state = regulator.state
        import pylab
        import numpy as np
        import pandas as pd

        recorded_positions = np.array(regulator.recorded_neuron_positions())
        in_circle_mask = np.sqrt(np.sum(recorded_positions ** 2, axis=0)) < RegulatorSetup.stim_circle_radius

        len_error = len(state.history["error"])
        len_firing = len(state.history["Firing rate"])
        len_lms = len(state.history["LMS"]) if "LMS" in state.history else 0
        min_length = min(len_error, len_firing, len_lms) if len_lms > 0 else min(len_error, len_firing)
        
        #print(state.history["error"])

        t = np.arange(min_length) * regulator.parameters.state_update_interval / 1000
        fig, ax = pylab.subplots(2, 1, figsize=(10, 8))
        ax[0].plot(t, np.array(state.history["error"])[:min_length], c='r', label="Error")
        ax[0].plot(t, np.array(state.history["Firing rate"])[:min_length], c='k', label="Firing rate (sp/s)")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Value")
        ax[0].legend()        
        if regulator.sheet.model.parameters.trial == 0:
            instantaneous_rates = []
            for i in range(len(regulator.last_spike_counts)):
                if in_circle_mask[i]:
                    rate = regulator.last_spike_counts[i] / (regulator.parameters.state_update_interval / 1000.0)
                    instantaneous_rates.append(rate)
            current_rate = np.mean(instantaneous_rates) if instantaneous_rates else 0.0
        else:
            positions = np.array(regulator.recorded_neuron_positions())
            in_circle_mask = np.zeros(positions.shape[1], dtype=bool)
            for (cx, cy) in RegulatorSetup.centers:
                d2 = (positions[0] - cx)**2 + (positions[1] - cy)**2
                in_circle_mask |= (d2 < RegulatorSetup.radius_sub**2)

            instantaneous_rates = []
            for i, spike_count in enumerate(regulator.last_spike_counts):
                if in_circle_mask[i]:
                    rate = spike_count / (regulator.parameters.state_update_interval / 1000.0)
                    instantaneous_rates.append(rate)
            current_rate = np.mean(instantaneous_rates) if instantaneous_rates else 0.0
        if "LMS" in state.history:
            ax[1].plot(t, np.array(state.history["LMS"])[:min_length], '--', label="Control Signal")
            ax[1].set_xlabel("Time (s)")
            ax[1].set_ylabel("Control Signal")
            ax[1].legend()
        fig.tight_layout()
        fig.savefig("autocontrol.png")
        pylab.close(fig)

        # 构建基础数据字典
        data_dict = {
            "Time (s)": t,
            "Current Rate (sp/s)": state.history["Current rate"][:min_length],
            "Firing Rate (sp/s)": state.history["Firing rate"][:min_length],
            "Target": state.history["Target"][:min_length],
            "Error": state.history["error"][:min_length]
        }

        if len_lms > 0:
            data_dict["Control Signal"] = state.history["LMS"][:min_length]
            
        # 从历史记录中逐个提取Spike Timestamp
        #data_dict["Spike Timestamp"] = [
        #    str(state.history["Spike Timestamp"][i]) for i in range(min_length)
        #]

        # 转为DataFrame并保存为CSV
        df = pd.DataFrame(data_dict)
        df.to_csv("history_data.csv", index=False)

# 定义闭环实验函数，将调节器和模型结合起来
def closed_loop_experiment(model):
    return [
        ClosedLoopOptogeneticStimulation(
            model,
            MozaikExtendedParameterSet({
                "num_trials": 1,     
                "duration": 18000,    
                "stimulator_array_list": [
                    {
                        "sheet": "V1_Exc_L2/3",           
                        "name": "closed_loop_array",       
                        "input_calculation_function": RegulatorSetup.calculate_input,
                        "state_update_function": RegulatorSetup.update_state,
                    }
                ],
            }),
        ),
    ]

