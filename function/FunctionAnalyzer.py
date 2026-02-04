import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class BasisFunctionAnalyzer:
    def __init__(self, model, M, K):
        """
        基函数分析器

        Args:
            model: 训练好的ADVR_NN模型
            M: 记忆深度
            K: 幅度函数数量
        """
        self.model = model
        self.M = M
        self.K = K
        self.device = model.device

    def generate_test_signals(self, num_samples=10000):
        """
        生成测试信号用于分析
        """
        # 生成不同幅度的测试信号
        amplitudes = np.linspace(0, 2, num_samples)
        phases = np.random.uniform(-np.pi, np.pi, num_samples)

        # 创建记忆效应
        signals = []
        for i in range(num_samples):
            # 当前信号
            x_current = amplitudes[i] * np.exp(1j * phases[i])

            # 记忆信号（这里简化处理，实际应为不同延迟）
            x_memory = []
            for m in range(self.M + 1):
                if i - m >= 0:
                    x_memory.append(signals[i - m])
                else:
                    x_memory.append(0 + 0j)

            signals.append(x_current)

        return np.array(signals)

    def analyze_basis_functions(self, signals):
        """
        分析基函数对应的Volterra项
        """
        # 将信号转换为tensor
        x_tensor = torch.from_numpy(signals).to(self.device)

        # 创建记忆窗口
        N = len(signals)
        memory_seq = []
        for i in range(N):
            if i < self.M:
                # 不足记忆深度补零
                seq = np.concatenate([np.zeros(self.M - i, dtype=complex),
                                      signals[:i + 1]])
            else:
                seq = signals[i - self.M:i + 1]
            memory_seq.append(seq)

        memory_seq = np.array(memory_seq)
        x_window = torch.from_numpy(memory_seq).to(self.device)

        # 获取幅度基函数
        with torch.no_grad():
            X_amp = torch.abs(x_window)  # |x(n-i)|
            amp_basis = self.model.DVR_layers(X_amp)  # f_k,m(|x(n-i)|)

            if self.model.activation == "ABS":
                amp_core = torch.abs(amp_basis)
            elif hasattr(self.model, 'act'):
                amp_core = self.model.act(amp_basis)
            else:
                amp_core = amp_basis

        # 重塑为 [batch, M+1, K]
        amp_core_reshaped = amp_core.view(N, self.M + 1, self.K)

        # 分析每个幅度基函数
        basis_analysis = {}

        for k in range(self.K):
            for m in range(self.M + 1):
                # 提取该基函数对输入幅度的响应
                f_km = amp_core_reshaped[:, m, k].cpu().numpy()
                x_amp = X_amp[:, m].cpu().numpy()

                # 尝试用多项式拟合
                analysis = self._analyze_function_shape(x_amp, f_km, k, m)
                basis_analysis[(k, m)] = analysis

        return basis_analysis

    def _analyze_function_shape(self, x_amp, f_km, k, m):
        """
        分析单个幅度函数的形状
        """
        # 去除零值附近的小信号
        mask = x_amp > 0.01
        if np.sum(mask) < 10:
            return {"type": "insufficient_data", "order": 0}

        x_amp_filtered = x_amp[mask]
        f_km_filtered = f_km[mask]

        # 尝试多项式拟合（1-5阶）
        best_order = 0
        best_r2 = -1
        poly_coeffs = None

        for order in range(1, 6):
            poly = PolynomialFeatures(degree=order, include_bias=True)
            X_poly = poly.fit_transform(x_amp_filtered.reshape(-1, 1))

            reg = LinearRegression()
            reg.fit(X_poly, f_km_filtered)

            r2 = reg.score(X_poly, f_km_filtered)

            if r2 > best_r2:
                best_r2 = r2
                best_order = order
                poly_coeffs = reg.coef_

        # 分析函数特性
        characteristics = self._analyze_function_characteristics(
            x_amp_filtered, f_km_filtered, poly_coeffs
        )

        # 判断对应的Volterra项类型
        volterra_type = self._classify_volterra_term(best_order, characteristics, m)

        return {
            "k": k,
            "m": m,
            "polynomial_order": best_order,
            "r_squared": best_r2,
            "coefficients": poly_coeffs,
            "characteristics": characteristics,
            "volterra_type": volterra_type
        }

    def _analyze_function_characteristics(self, x_amp, f_km, coeffs):
        """
        分析函数的特性
        """
        characteristics = {}

        # 1. 对称性分析（奇函数/偶函数）
        # 采样对称点
        x_pos = x_amp
        f_pos = f_km

        # 计算导数信息（近似）
        if len(x_pos) > 2:
            sorted_indices = np.argsort(x_pos)
            x_sorted = x_pos[sorted_indices]
            f_sorted = f_km[sorted_indices]

            # 数值微分
            df_dx = np.gradient(f_sorted, x_sorted)
            characteristics['max_slope'] = np.max(np.abs(df_dx))
            characteristics['avg_slope'] = np.mean(np.abs(df_dx))

            # 饱和特性
            if len(x_sorted) > 10:
                characteristics['saturation'] = np.std(f_sorted[-10:]) / np.std(f_sorted[:10])

        return characteristics

    def _classify_volterra_term(self, order, characteristics, m):
        """
        根据分析结果分类对应的Volterra项
        """
        if order == 1:
            # 线性项
            if m == 0:
                return "Linear term: x(n)"
            else:
                return f"Memory linear term: x(n-{m})"

        elif order == 2:
            # 二阶项
            if m == 0:
                return "2nd order: |x(n)|^2"
            else:
                return f"2nd order with memory: |x(n-{m})|^2 or x(n)x(n-{m})"

        elif order == 3:
            # 三阶项
            if m == 0:
                return "3rd order: |x(n)|^2 x(n)"
            else:
                # 根据相位恢复类型判断
                if characteristics.get('max_slope', 0) > 10:
                    return f"3rd order DDR-like: x(n)^2 conj(x(n-{m}))"
                else:
                    return f"3rd order memory: |x(n-{m})|^2 x(n)"

        elif order >= 4:
            # 高阶项
            return f"Higher order (order {order}) with memory depth {m}"

        else:
            return "Unknown"

    def visualize_basis_functions(self, analysis_results, top_n=10):
        """
        可视化基函数分析结果
        """
        # 提取要可视化的基函数
        sorted_results = sorted(
            analysis_results.items(),
            key=lambda x: abs(x[1].get('r_squared', 0)),
            reverse=True
        )[:top_n]

        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        axes = axes.flatten()

        for idx, ((k, m), result) in enumerate(sorted_results):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # 生成测试点用于绘图
            x_test = np.linspace(0, 2, 100)

            # 使用多项式系数重建函数
            if result['coefficients'] is not None:
                poly = PolynomialFeatures(
                    degree=result['polynomial_order'],
                    include_bias=True
                )
                X_poly = poly.fit_transform(x_test.reshape(-1, 1))
                y_fit = X_poly @ np.concatenate([[0], result['coefficients']])

                ax.plot(x_test, y_fit, 'r-', label='Polynomial fit')

            ax.set_title(f'Basis (k={k}, m={m})\n{result["volterra_type"]}')
            ax.set_xlabel('|x|')
            ax.set_ylabel('f(|x|)')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

        # 输出总结
        print("=" * 80)
        print("Basis Function Analysis Summary")
        print("=" * 80)

        # 按Volterra类型分类统计
        type_counter = {}
        for (k, m), result in analysis_results.items():
            volterra_type = result.get('volterra_type', 'Unknown')
            type_counter[volterra_type] = type_counter.get(volterra_type, 0) + 1

        print("\nVolterra Term Distribution:")
        for term_type, count in sorted(type_counter.items(), key=lambda x: x[1], reverse=True):
            print(f"  {term_type}: {count} basis functions")

        # 打印最重要的基函数
        print("\nTop 5 Most Significant Basis Functions:")
        for (k, m), result in sorted_results[:5]:
            print(f"  Basis (k={k}, m={m}):")
            print(f"    Volterra type: {result['volterra_type']}")
            print(f"    Polynomial order: {result['polynomial_order']}")
            print(f"    R²: {result['r_squared']:.4f}")
            print()

    def analyze_phase_recovery_terms(self):
        """
        分析相位恢复项对应的Volterra项
        """
        # 根据论文公式(5)的相位恢复项
        phase_terms = {
            'linear': 'x(n-m)',
            '1st_order': 'exp(jθ(n-m))',
            '2nd_order_type1': 'exp(jθ(n-m))·|x(n)|',
            '2nd_order_type2': 'x(n)',
            '2nd_order_type3': 'x(n-m)',
            'ddr_term': 'x(n)²·conj(x(n-m))'
        }

        print("=" * 80)
        print("Phase Recovery Terms and Corresponding Volterra Terms")
        print("=" * 80)

        for term_name, term_expr in phase_terms.items():
            # 解析对应的Volterra项
            if term_name == 'linear':
                volterra_eq = 'H₁(m)·x(n-m)'
            elif term_name == '1st_order':
                volterra_eq = '|x(n-m)|·exp(jθ(n-m)) = x(n-m)'
            elif term_name == '2nd_order_type1':
                volterra_eq = '|x(n-m)|·|x(n)|·exp(jθ(n-m)) = x(n-m)·|x(n)|'
            elif term_name == '2nd_order_type2':
                volterra_eq = '|x(n-m)|·x(n)'
            elif term_name == '2nd_order_type3':
                volterra_eq = '|x(n-m)|·x(n-m)'
            elif term_name == 'ddr_term':
                volterra_eq = '|x(n-m)|·x(n)²·conj(x(n-m))'

            print(f"{term_name}:")
            print(f"  Expression: f(|x|) × {term_expr}")
            print(f"  Volterra equivalent: {volterra_eq}")
            print()

