# Jitter Methodology & Statistical Foundations

Jitter is a root-cause analysis framework for on-device machine learning latency. When an ML model runs slower than expected on a mobile SoC (e.g. Google Tensor, Qualcomm Snapdragon, Apple Neural Engine, MediaTek Dimensity), a scalar benchmark summary (such as "median 24.3 ms") cannot explain *why* the degradation occurred.

Jitter replaces single-number benchmarks with an orchestrated system of **four controlled experimental reruns**, paired with **non-parametric statistical estimators** and a **rule-based diagnosis engine**.

---

## 1. The Four Controlled Reruns

Latency variance on mobile devices stems from distinct physical and operating system mechanisms. Jitter isolates these variables by manipulating process lifecycle, hardware temperature, execution delegates, and background contention.

```
                  ┌─────────────────────────────────────────┐
                  │          Initial Benchmark Run          │
                  │   (GPU / NPU requested, cold process)   │
                  └────────────────────┬────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│  Rerun 1: Warm   │          │ Rerun 2: Cooldown│          │ Rerun 3: CPU     │
│ Same process, no │          │ Rested baseline  │          │ Reference run    │
│ driver reload    │          │ (< 32°C battery) │          │ CPU fallback test│
└────────┬─────────┘          └────────┬─────────┘          └────────┬─────────┘
         │                             │                             │
         └─────────────────────────────┼─────────────────────────────┘
                                       ▼
                              ┌──────────────────┐
                              │  Rerun 4: Quiet  │
                              │ Airplane mode,   │
                              │ DND, stay-awake  │
                              └──────────────────┘
```

### 1.1 Cold vs. Warm Rerun (`cold_warm`)
- **Execution**: The first run executes in a freshly spawned process (`ProcessStart.COLD`). Immediately after, a second measured sequence runs inside the *same* process (`ProcessStart.WARM`) without restarting the runtime.
- **Hypothesis Tested**: **Warmup Latency Drift (`warmup_drift`)**. Cold process launches incur JIT compilation of GPU shader kernels (OpenCL/Vulkan), memory mapping of weight tensors from flash storage, and runtime graph optimization. Warm executions measure steady-state execution.

### 1.2 Cooldown Rerun (`cooldown`)
- **Execution**: The runner pauses the benchmark and monitors SoC temperature until the battery temperature drops below $32.0^\circ\text{C}$ (or rests for up to 240 seconds). The benchmark is then repeated from rested thermal equilibrium.
- **Hypothesis Tested**: **Thermal Throttling (`thermal_throttling`)**. Continuous ML inference dissipates thermal energy ($P \cdot dt$), heating the SoC into governor trip points (e.g. Android `THERMAL_STATUS_MODERATE` / `SEVERE`). Rested reruns isolate clock frequency downscaling from intrinsic compute requirements.

### 1.3 Delegate Swap Rerun (`delegate_swap`)
- **Execution**: The model is executed with execution delegate explicitly forced to standard CPU multithreading (`XNNPACK` / `TFLite CPU`), overriding requested GPU or NPU delegates.
- **Hypothesis Tested**: **Delegate Fallback (`delegate_fallback`)**. Quantization mismatches, unsupported tensor operators, or out-of-memory driver crashes silently cause the runtime to fall back to CPU execution without reporting an exception. Comparing GPU and CPU latency verifies true hardware acceleration.

### 1.4 Quiet Device Rerun (`quiet_device`)
- **Execution**: Android settings are placed into quiet mode: Airplane Mode active (radio off), Do Not Disturb (DND) active (notifications suppressed), background synchronization suspended, and display locked awake.
- **Hypothesis Tested**: **Background Contention Noise (`background_noise`)**. Tail latency outliers ($p_{95}, p_{99}$) frequently arise from asynchronous OS tasks (garbage collection, thread preemption, network I/O, push notifications). If variance collapses in quiet mode, the model itself is not erratic.

---

## 2. Statistical Core & Estimators

Mobile benchmark samples violate standard independent and identically distributed (i.i.d.) Gaussian assumptions:
1. **Autocorrelation**: Iterations exhibit serial correlation due to memory caching, frequency governor ramping, and progressive heating. Standard errors computed as $\sigma / \sqrt{n}$ severely understate uncertainty.
2. **Heavy Tails**: OS interrupts produce one-sided latency spikes that skew the sample mean and variance.

### 2.1 Autocorrelation & Effective Sample Size ($n_{\text{eff}}$)

Jitter estimates the sample autocorrelation sequence $\hat{\rho}_k$ using zero-padded FFT. To truncate the summation without noise accumulation, Jitter implements **Geyer's (1992) Initial Positive Sequence (IPS)** estimator:

$$\Gamma_m = \hat{\rho}_{2m} + \hat{\rho}_{2m+1}$$

Summation terminates at the largest $m = M$ such that $\Gamma_m > 0$. The integrated autocorrelation time $\tau_{\text{int}}$ and effective sample size $n_{\text{eff}}$ are defined as:

$$\tau_{\text{int}} = 1 + 2 \sum_{k=1}^{2M+1} \hat{\rho}_k, \quad n_{\text{eff}} = \frac{n}{\tau_{\text{int}}}$$

When $n_{\text{eff}} < 10.0$, Jitter triggers an `inconclusive` diagnostic flag because the data lacks sufficient degrees of freedom to differentiate systematic effects from random walks.

### 2.2 Stationary Block Bootstrap

Confidence intervals for median latency are computed using the **Politis & White (2004) stationary block bootstrap**. The optimal block length $b^*$ is selected from the empirical spectral density of the autocorrelation function, preserving serial dependency within resampled blocks.

### 2.3 Non-Parametric Shift & Effect Size

- **Hodges-Lehmann Location Shift**:
  $$\hat{\Delta}_{\text{HL}} = \text{median}\{y_j - x_i \mid 1 \le i \le n_x, 1 \le j \le n_y\}$$
  The median of all pairwise differences provides a robust non-parametric estimator of shift that is unaffected by extreme outliers.

- **Cliff's Delta ($\delta$)**:
  $$\delta = \frac{\#\{y_j > x_i\} - \#\{y_j < x_i\}}{n_x \cdot n_y}$$
  Qualitative effect magnitudes follow Romano et al. (2006):
  - $|\delta| < 0.147$: **Negligible**
  - $0.147 \le |\delta| < 0.330$: **Small**
  - $0.330 \le |\delta| < 0.474$: **Medium**
  - $|\delta| \ge 0.474$: **Large**

- **Two-Sample Block Permutation Test**:
  Block permutation shuffles contiguous blocks of observations between condition $A$ and condition $B$, testing the null hypothesis $H_0: \text{Median}_A = \text{Median}_B$ while preserving serial correlation.

### 2.4 Monotonic Trend & Drift

Monotonic thermal degradation is estimated using **Theil-Sen robust regression**:
$$\beta_{\text{TS}} = \text{median}\left\{ \frac{x_j - x_i}{j - i} \;\middle|\; 1 \le i < j \le n \right\}$$
Statistical significance of the monotonic trend is established using the two-sided **Mann-Kendall rank test** ($p < 0.05$).

### 2.5 Change-Point Warmup Detection

Warmup transition is detected using a two-sided Cumulative Sum (**CUSUM**) detector:
$$S_k = \sum_{i=1}^k (x_i - \tilde{x})$$
The boundary iteration where CUSUM reaches its maximum divergence indicates the transition point between cold JIT compilation and steady-state inference.

---

## 3. Failure Mode Diagnostic Signatures

| Failure Mode | Physical Cause | Primary Signature | Secondary Evidence |
| :--- | :--- | :--- | :--- |
| **`warmup_drift`** | Shader compilation, weight page-in, driver init | Cold initial iterations $\ge 1.50\times$ slower than warm steady-state | CUSUM detects $\ge 1$ discard iterations; warm rerun stable |
| **`thermal_throttling`** | CPU/GPU downclocking to protect silicon | Monotonic upward drift (Theil-Sen $\ge 0.02$ ms/iter, MK $p < 0.05$) | Thermal status $\ge 1$ (LIGHT); rested cooldown recovers $\ge 1.0$ ms |
| **`delegate_fallback`** | Silent driver failure falling back to CPU | Execution delegate mismatch OR parity with CPU reference run | Cliff's delta vs. CPU run $|\delta| \le 0.147$ |
| **`background_noise`** | OS thread preemption, notifications, WiFi | High MAD ratio ($\ge 0.25$), $\ge 2$ modified Z-score outliers | Tail latency ($p_{99}$) drops by $\ge 20\%$ in quiet device mode |
| **`regression`** | Sub-optimal model graph, unquantized op | Uniform latency increase across all 4 controlled conditions | Warmup, thermal, delegate, and noise hypotheses ruled out |
| **`inconclusive`** | Insufficient sample size or contradictory data | $n_{\text{eff}} < 10.0$ or iterations per run $< 15$ | Conflicting evidence across reruns |

---

## 4. Empirical Validation

Jitter's diagnosis engine is validated continuously against synthetic Monte Carlo benchmarks across multiple architectures (MobileNetV2, ResNet50, BERT-Tiny) and random seeds.

Validation metrics:
- **Macro-Averaged F1 Score**: `1.0000` (Target: $\ge 0.95$)
- **Expected Calibration Error (ECE)**: `0.0398` (Target: $\le 0.05$)
- **Brier Score**: `0.0020`
- **Mean Diagnosis Latency**: `84.19 ms`

See [diagnosis.md](file:///c:/Users/ayush/OneDrive/Desktop/jitter-marketing-site/jitter/docs/diagnosis.md) for full confusion matrix and empirical calibration results.
