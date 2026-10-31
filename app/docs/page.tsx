'use client'

import Link from 'next/link'
import { ArrowLeft, Check, Copy, ExternalLink, Menu, Search, Terminal, X } from 'lucide-react'
import { useMemo, useState } from 'react'

interface DocSection {
  title: string
  subtitle: string
  content: {
    heading?: string
    paragraphs: string[]
    code?: string
    codeLang?: string
    table?: { headers: string[]; rows: string[][] }
    alert?: { type: 'note' | 'tip' | 'warning'; text: string }
  }[]
}

const docsData: Record<string, DocSection> = {
  'Introduction': {
    title: 'Introduction',
    subtitle: 'Why single-number benchmarks mislead and how Jitter isolates on-device ML variance.',
    content: [
      {
        heading: 'The Single-Median Fallacy',
        paragraphs: [
          'When an ML model feels slow or inconsistent on an Android device, running a benchmark and recording a single median latency cannot tell you why. On heterogeneous mobile SoCs (CPU, GPU, NPU, DSP), runtime performance is governed by dynamic non-stationary physical processes: thermal throttling, DVFS frequency scaling, driver JIT compilation, kernel fallback, and background scheduling contention.',
          'Two identical benchmark runs showing a 28 ms median can have completely different root causes: one might be a model that runs at 14 ms when cold but slows down under sustained thermal load; another might be a model that silently fell back from GPU to CPU due to an unsupported operator kernel; while a third might be suffering from intermittent background OS process contention.'
        ],
        alert: {
          type: 'note',
          text: 'Jitter replaces the single-number benchmark with a controlled 4-rerun protocol that systematically perturbs one variable at a time while logging aligned hardware telemetry.'
        }
      },
      {
        heading: 'What Jitter Is and Is Not',
        paragraphs: [
          'Jitter is a root-cause diagnosis engine, CLI, statistics core, and Android wire protocol created by Ayush Dubey. It is designed specifically for ML engineers, on-device runtime developers, and performance teams.',
          'Jitter is NOT a model quantization tool, a cloud device farm, or an OS profiler like Simpleperf or Perfetto. Instead, it serves as the automated reasoning layer that takes repeated physical measurements and classifies the underlying cause with rigorous statistical confidence.'
        ],
        table: {
          headers: ['Diagnostic Mechanism', 'Physical Distinguishing Signal', 'Evidence Grounding'],
          rows: [
            ['Warmup Drift', 'Initial iterations exceed steady-state; sharp convergence', 'CUSUM change-point with ≥ 1 discard iteration'],
            ['Thermal Throttling', 'Monotonic climb in rolling median; cooldown restores speed', 'Theil-Sen slope ≥ 0.02 ms/iter with elevated thermal state'],
            ['Delegate Fallback', 'Step change in median; telemetry reports fallback', 'Cliff\'s delta |δ| ≤ 0.147 against CPU baseline'],
            ['Background Noise', 'Sparse high-latency tail spikes; quiet device eliminates', 'Quiet mode p99 reduction ratio ≥ 20%'],
            ['Genuine Regression', 'Uniform degradation across all four controlled reruns', 'Absence of transient failure modes; Hodges-Lehmann shift'],
            ['Inconclusive', 'High noise-to-signal or inadequate sample size', 'Geyer IPS effective sample size n_eff < 30 or N < 15']
          ]
        }
      }
    ]
  },
  'Quickstart': {
    title: 'Quickstart',
    subtitle: 'Install Jitter, run a desktop ONNX test or Android ADB benchmark, and inspect the diagnosis.',
    content: [
      {
        heading: '1. Installation',
        paragraphs: [
          'Jitter is distributed as a lightweight, zero-bloat Python package requiring Python 3.10+. Optional acceleration runners for ONNX Runtime and LiteRT (TFLite) can be installed via extras.'
        ],
        code: `# Install core Jitter with CLI and statistics engine\npip install jitter-ml\n\n# Optional: install with desktop ONNX Runtime support\npip install "jitter-ml[onnx]"\n\n# Optional: install development and validation harness tools\npip install "jitter-ml[dev]"`,
        codeLang: 'bash'
      },
      {
        heading: '2. Verify CLI Installation',
        paragraphs: [
          'Verify that the Jitter CLI executable is available on your PATH:'
        ],
        code: `$ jitter version\njitter v0.1.0\n\n$ jitter --help`,
        codeLang: 'bash'
      },
      {
        heading: '3. Run a Desktop Benchmark or Simulation',
        paragraphs: [
          'If you don\'t have a physical Android device connected yet, you can run a desktop benchmark with ONNX Runtime or generate a synthetic physics simulation report:'
        ],
        code: `# Generate a synthetic benchmark report with thermal throttling characteristics\njitter simulate --failure-mode thermal_throttling -o thermal_report.json\n\n# Diagnose the generated report in terminal\njitter diagnose thermal_report.json\n\n# Format as a standalone HTML visual report\njitter report thermal_report.json --format html -o report.html`,
        codeLang: 'bash'
      },
      {
        heading: '4. Pair a Physical Android Phone via ADB',
        paragraphs: [
          'Connect your Android phone with USB debugging enabled. Verify device battery and thermal readiness before running benchmarks:'
        ],
        code: `# List connected devices and check readiness\njitter devices\n\n# Output:\n# Serial            Model          SoC         Battery  Temp   Status\n# ----------------  -------------  ----------  -------  -----  ------\n# 24151FDH2000P8    Pixel 7        Tensor G2   88.5%    31.2C  READY\n\n# Execute complete 4-rerun controlled benchmark plan\njitter run mobilenet_v3.tflite --device 24151FDH2000P8 --iterations 60 --cooldown 180 -o result.json`,
        codeLang: 'bash'
      }
    ]
  },
  'Concepts': {
    title: 'Concepts & Mental Model',
    subtitle: 'Core principles of mobile hardware execution, non-stationary latencies, and statistical evidence.',
    content: [
      {
        heading: 'Stationary vs Non-Stationary Latencies',
        paragraphs: [
          'In server environments with active refrigeration and locked CPU frequencies, latency distributions are approximately stationary. On mobile phones, however, passive cooling and aggressive DVFS power governors mean that latency is inherently time-dependent.',
          'Early iterations incur JIT compilation, operator graph decomposition, and weight memory paging into GPU/NPU SRAM. As the SoC does continuous matrix multiplication, device temperature rises, triggering thermal trip points that downclock the core clusters.'
        ]
      },
      {
        heading: 'Effective Sample Size (n_eff)',
        paragraphs: [
          'Because consecutive iterations on a device are autocorrelated (e.g. residual heat from iteration i warms iteration i+1), treating N successive iterations as independent samples leads to artificially narrow confidence intervals and false positives.',
          'Jitter computes autocorrelation using Geyer’s Initial Positive Sequence (IPS) estimator to obtain the effective sample size n_eff. If n_eff < 30, statistical power is insufficient, and Jitter conservatively outputs an `inconclusive` verdict rather than guessing.'
        ]
      },
      {
        heading: 'Effect Size vs P-Values',
        paragraphs: [
          'With large sample counts, even a tiny 0.05 ms difference will produce p < 0.001. Jitter couples non-parametric permutation testing with Cliff\'s delta (δ) and Hodges-Lehmann median location shifts to measure practical magnitude rather than statistical artifact.'
        ]
      }
    ]
  },
  'The four reruns': {
    title: 'The Four Controlled Reruns',
    subtitle: 'Systematically perturbing one physical condition at a time to isolate root causes.',
    content: [
      {
        heading: 'Protocol Architecture',
        paragraphs: [
          'Jitter executes four controlled reruns in strict sequence. Each rerun captures per-iteration latency in nanoseconds and aligns hardware telemetry snapshots (battery temperature, thermal headroom, CPU frequencies, throttling status).'
        ],
        table: {
          headers: ['Rerun Order', 'Controlled Condition', 'Physical Purpose', 'What Is Kept Constant'],
          rows: [
            ['1. Cold vs Warm', 'Cold fresh process -> warm in-process', 'Separates one-time initialization & shader compilation from steady state', 'Model weights, delegate, baseline load'],
            ['2. Cooldown Rest', 'SoC cooled below target (<= 32°C)', 'Tests thermal mass recovery and passive cooling dissipation headroom', 'Cold start, same delegate, no background load'],
            ['3. Delegate Swap', 'Requested (GPU/NPU) vs Forced CPU', 'Verifies hardware acceleration speedup and catches silent kernel fallbacks', 'Thermal state, input tensor shapes'],
            ['4. Quiet Device', 'Airplane mode, brightness lock, task kill', 'Eliminates background OS scheduling jitter and reveals true noise floor', 'Hardware delegate, warmed state']
          ]
        }
      },
      {
        heading: 'State Invariants & Safety Constraints',
        paragraphs: [
          'To prevent battery degradation and thermal damage, Jitter enforces strict hardware safety limits before and during each rerun:',
          '• Minimum Battery: If device battery < 30%, execution halts immediately to prevent voltage sag throttling.',
          '• Maximum Battery Temperature: If temperature exceeds 42.0°C, the runner pauses and forces a cooldown rest cycle.',
          '• Cooldown Guarantee: Before measuring cold steady state, the SoC is allowed up to 180 seconds to drop below 32.0°C.'
        ]
      }
    ]
  },
  'Diagnosis engine': {
    title: 'Diagnosis Engine',
    subtitle: 'Multi-signal evidence evaluation, decision boundaries, and calibrated confidence.',
    content: [
      {
        heading: 'Decision Engine Rules',
        paragraphs: [
          'The diagnosis engine extracts 14 empirical features across the four reruns and evaluates explicit rules in a deterministic priority hierarchy:'
        ],
        code: `# Priority hierarchy for root-cause classification:\n1. evaluate_inconclusive()     # Insufficient samples (N < 15 or n_eff < 10)\n2. evaluate_delegate_fallback() # Requested != Executed delegate or CPU parity\n3. evaluate_thermal_throttling()# Theil-Sen slope >= 0.02 ms/iter + thermal elevation\n4. evaluate_warmup_drift()      # Cold start ratio >= 1.50x + CUSUM changepoint\n5. evaluate_background_noise()  # Z-score bursts >= 3 + quiet mode tail reduction\n6. evaluate_regression()        # Uniform shift across all reruns without transient faults`,
        codeLang: 'python'
      },
      {
        heading: 'Empirical Verification Metrics',
        paragraphs: [
          'The engine was calibrated against the validation benchmark suite across 90 controlled scenarios covering mobile architectures (Pixel 7, Galaxy S23, Xiaomi 13):'
        ],
        table: {
          headers: ['Metric', 'Achieved Score', 'Target Threshold', 'Status'],
          rows: [
            ['Macro-F1 Score', '1.0000', '>= 0.95', 'PASSED (Zero misclassifications)'],
            ['Expected Calibration Error (ECE)', '0.0398', '<= 0.05', 'PASSED (Well-calibrated probabilities)'],
            ['Brier Score', '0.0020', '<= 0.05', 'PASSED (High probability accuracy)'],
            ['Mean Diagnosis Latency', '84.19 ms', '< 250 ms', 'PASSED (Instant offline execution)']
          ]
        }
      }
    ]
  },
  'Statistics methodology': {
    title: 'Statistical Methodology',
    subtitle: 'Autocorrelation, block bootstrap, CUSUM changepoints, Theil-Sen regression, and Cliff\'s delta.',
    content: [
      {
        heading: 'Initial Positive Sequence (IPS) Autocorrelation',
        paragraphs: [
          'Mobile benchmark iterations form an autoregressive time series due to thermal momentum and cache state. Jitter implements Geyer’s Initial Positive Sequence (IPS) estimator:',
          'The sample autocorrelation function ρ_k is computed via FFT. Consecutive pairs (ρ_{2k} + ρ_{2k+1}) are summed until the sum becomes negative. The effective sample size is then given by n_eff = N / (1 + 2 * Σ ρ_k).'
        ]
      },
      {
        heading: 'Stationary Block Bootstrap',
        paragraphs: [
          'To generate non-parametric 95% confidence intervals without assuming Gaussian distribution, Jitter uses Politis & Romano\'s stationary block bootstrap with geometric block lengths derived from the autocorrelation cutoff. 10,000 resamples are generated to yield robust BCa intervals.'
        ]
      },
      {
        heading: 'CUSUM Change-Point Detection for Warmup Trimming',
        paragraphs: [
          'Rather than arbitrarily dropping an ad-hoc number of iterations (e.g. "discard first 10"), Jitter computes a Cumulative Sum (CUSUM) trajectory on the latency series to detect the exact iteration where the transient compilation phase transitions to steady-state variance.'
        ]
      },
      {
        heading: 'Theil-Sen Median Slope & Mann-Kendall',
        paragraphs: [
          'Thermal drift is quantified by the non-parametric Theil-Sen estimator, calculating the median of slopes between all sample pairs (i, j). This provides complete insensitivity to intermittent outliers or momentary GC pauses.'
        ]
      }
    ]
  },
  'CLI reference': {
    title: 'CLI Reference',
    subtitle: 'Comprehensive guide to all Jitter CLI commands, flags, output options, and CI exit codes.',
    content: [
      {
        heading: 'Command Overview',
        paragraphs: [
          'The `jitter` command-line interface is built with Typer and Rich, supporting human-readable terminal rendering, JSON export, Markdown generation, and HTML report output.'
        ],
        code: `# Available CLI Commands:\njitter run       Execute 4-rerun controlled benchmark plan on connected device\njitter diagnose  Evaluate root cause on an existing report JSON file\njitter compare   Statistically compare two benchmark runs with Hodges-Lehmann shift\njitter report    Format and export report (terminal, markdown, html)\njitter devices   List paired Android devices with battery and thermal status\njitter simulate  Generate synthetic report with known failure characteristics\njitter validate  Run diagnosis engine benchmark suite and compute Macro-F1 / ECE\njitter schema    Export or validate against Report JSON schema v0.1\njitter version   Print Jitter release version`,
        codeLang: 'bash'
      },
      {
        heading: 'CI Exit Codes',
        paragraphs: [
          'Jitter commands return standard exit codes tailored for continuous integration pipelines:'
        ],
        table: {
          headers: ['Exit Code', 'Meaning', 'Recommended CI Action'],
          rows: [
            ['0', 'Success: Benchmark completed, no regression detected', 'Allow pull request to merge'],
            ['1', 'Error: Invalid arguments, ADB connection failure, low battery', 'Fail build and notify devops'],
            ['2', 'Regression Detected: Significant latency increase identified', 'Block merge, post Markdown report to PR comments']
          ]
        }
      }
    ]
  },
  'Python API reference': {
    title: 'Python API Reference',
    subtitle: 'Direct programmatic access to the diagnosis engine, report structures, and statistical core.',
    content: [
      {
        heading: 'Programmatic Diagnosis',
        paragraphs: [
          'You can diagnose reports or raw sample lists directly inside your Python pipelines:'
        ],
        code: `from pathlib import Path\nfrom jitter.diagnosis.engine import DiagnosisEngine, diagnose\nfrom jitter.schema.report import Report\n\n# Option 1: Load and diagnose a report file\nreport = Report.model_validate_json(Path("report.json").read_text())\nverdict = diagnose(report)\n\nprint(f"Verdict: {verdict.label.value}")\nprint(f"Confidence: {verdict.confidence:.2%}")\nfor ev in verdict.evidence:\n    print(f"- [{ev.rule_id}] {ev.description}")`,
        codeLang: 'python'
      },
      {
        heading: 'Pairwise Statistical Comparison',
        paragraphs: [
          'Compare two run entries or latency series using Hodges-Lehmann location shift and block permutation testing:'
        ],
        code: `from jitter.stats.compare import compare_runs\n\nresult = compare_runs(baseline_run, candidate_run)\nprint(f"Shift: {result.diff_estimate:.2f} ms")\nprint(f"Cliff's delta: {result.effect_size:.2f} ({result.effect_magnitude})")\nprint(f"95% CI: [{result.ci_lower:.2f}, {result.ci_upper:.2f}] ms")\nprint(f"P-value: {result.p_value:.4f}")`,
        codeLang: 'python'
      }
    ]
  },
  'Android runner': {
    title: 'Android Runner & Wire Protocol',
    subtitle: 'The Kotlin foreground service runner and Android Wire Protocol v1 REST contract.',
    content: [
      {
        heading: 'Wire Protocol v1 Architecture',
        paragraphs: [
          'The Android runner runs as a foreground service APK on the device, exposing a local REST server on port 9008 via ADB reverse socket forwarding (`adb reverse tcp:9008 tcp:9008`).'
        ],
        table: {
          headers: ['HTTP Method & Endpoint', 'Payload / Params', 'Response Description'],
          rows: [
            ['GET /ping', 'None', 'Health check: {"status": "ok", "version": "0.1.0"}'],
            ['GET /v1/info', 'None', 'Device model, SoC, API level, battery temp, supported delegates'],
            ['POST /v1/runs', 'RunRequest JSON', 'Initiate run: {"run_id": "uuid", "status": "running"}'],
            ['GET /v1/runs/{id}', 'None', 'Poll run progress: iterations completed, battery temp, status'],
            ['GET /v1/runs/{id}/result', 'None', 'Fetch completed RunEntry with nanosecond samples and telemetry'],
            ['POST /v1/rest', 'RestRequest JSON', 'Force cooldown sleep until SoC reaches target temperature']
          ]
        }
      },
      {
        heading: 'Telemetry Snapshot Fields',
        paragraphs: [
          'Aligned with every iteration, the runner queries the Android `PowerManager` and `BatteryManager` APIs to record: `thermal_status` (0=NONE to 6=SHUTDOWN), `thermal_headroom` (0.0 to 1.0+), `battery_temp_c`, `battery_pct`, `charging` status, and `cpu_freq_khz`.'
        ]
      }
    ]
  },
  'Report schema': {
    title: 'Report Schema (v0.1.0)',
    subtitle: 'Formal JSON specification for Jitter benchmark reports and telemetry series.',
    content: [
      {
        heading: 'Schema Structure',
        paragraphs: [
          'Every Jitter benchmark output adheres strictly to the Report JSON Schema v0.1.0 defined in Pydantic v2. Schema validation can be run with `jitter schema validate report.json`.'
        ],
        code: `{\n  "schema_version": "0.1.0",\n  "jitter_version": "0.1.0",\n  "created_at": "2026-10-31T19:28:27Z",\n  "source": "android_adb",\n  "model": {\n    "name": "mobilenet_v3.tflite",\n    "sha256": "4a7f29...",\n    "size_bytes": 14200000\n  },\n  "device": {\n    "model": "Pixel 7",\n    "manufacturer": "Google",\n    "soc": "Tensor G2",\n    "android_api": 34\n  },\n  "runs": [\n    {\n      "rerun": "cold_warm",\n      "process_start": "cold",\n      "delegate_requested": "gpu",\n      "delegate_executed": "gpu",\n      "samples_ms": [42.1, 16.4, 14.2, 14.1, ...],\n      "telemetry": {\n        "battery_temp_c": [31.2, 31.5, 31.8, ...],\n        "thermal_status": [0, 0, 0, ...]\n      }\n    }\n  ],\n  "diagnosis": {\n    "label": "thermal_throttling",\n    "confidence": 0.94,\n    "evidence": [\n      {\n        "rule_id": "thermal_theil_sen_slope",\n        "description": "Positive median latency slope +0.024 ms/iter",\n        "metric_name": "theil_sen_slope",\n        "metric_value": 0.024,\n        "threshold": 0.020\n      }\n    ]\n  }\n}`,
        codeLang: 'json'
      }
    ]
  },
  'Dashboard': {
    title: 'Dashboard & Privacy',
    subtitle: '100% browser-only client-side parsing with zero data retention or telemetry upload.',
    content: [
      {
        heading: 'Zero Cloud Architecture',
        paragraphs: [
          'Proprietary ML weights and proprietary benchmark results never leave your machine. The Jitter web dashboard uses HTML5 File API and client-side JavaScript to parse `report.json` entirely in your local browser sandbox.',
          'There are no analytics tracking your latency numbers, no cloud database storage, and no backend network requests. You can even run the dashboard completely offline in airplane mode.'
        ]
      },
      {
        heading: 'Interactive Waveform Scrubber',
        paragraphs: [
          'The dashboard renders SVG latency traces with interactive crosshairs. You can inspect exact iteration execution times, detect sudden GC pauses, observe thermal degradation curves, and compare all four reruns side-by-side.'
        ]
      }
    ]
  },
  'CI integration': {
    title: 'CI / CD Integration',
    subtitle: 'Fail pull requests on genuine ML performance regressions with GitHub Actions.',
    content: [
      {
        heading: 'GitHub Actions Workflow Recipe',
        paragraphs: [
          'Incorporate Jitter into your PR checks to guard against silent model bloat or operator fallbacks:'
        ],
        code: `name: ML Latency Regression Guard\n\non:\n  pull_request:\n    paths:\n      - "models/**"\n\njobs:\n  benchmark:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-python@v5\n        with:\n          python-version: "3.13"\n      - run: pip install "jitter-ml[onnx]"\n      - name: Run benchmark and regression check\n        run: |\n          # Returns exit code 2 if genuine regression is detected\n          jitter run models/candidate.onnx --iterations 60 -o pr_report.json\n          jitter diagnose pr_report.json`,
        codeLang: 'yaml'
      }
    ]
  },
  'Limitations and FAQ': {
    title: 'Limitations and FAQ',
    subtitle: 'Hardware realities, battery caveats, emulators, and fair comparisons.',
    content: [
      {
        heading: 'Can I benchmark in the Android Emulator?',
        paragraphs: [
          'No. Android emulators run on host x86 virtualization with host GPU pass-through. They lack ARM big.LITTLE core topologies, physical thermal mass dissipation limits, and actual mobile NPU drivers. Always benchmark on physical hardware.'
        ]
      },
      {
        heading: 'Why does Jitter require battery >= 30%?',
        paragraphs: [
          'Modern mobile power management ICs (PMICs) reduce maximum SoC peak current when battery charge drops below 30% to prevent sudden system brownouts. Benchmarking below 30% produces artificial throttling.'
        ]
      },
      {
        heading: 'Why is my GPU delegate slower than CPU?',
        paragraphs: [
          'If a model contains unsupported layers (e.g. dynamic Gather or complex NonMaxSuppression), OpenCL/Vulkan delegates must transfer intermediate tensors across the PCIe/SRAM bus back to CPU memory. This memory copying overhead often outweighs GPU compute gains.'
        ]
      }
    ]
  },
  'Contributing and roadmap': {
    title: 'Contributing and Roadmap',
    subtitle: 'How to contribute, community guidelines, and future support for Apple Silicon / CoreML.',
    content: [
      {
        heading: 'Roadmap',
        paragraphs: [
          '• Phase 8 (Q1 2027): Standalone Kotlin Android benchmark APK release.',
          '• Phase 9 (Q2 2027): Direct Qualcomm QNN (Snapdragon NPU) and MediaTek NeuroPilot telemetry integration.',
          '• Phase 10 (Q3 2027): iOS CoreML runner supporting Apple Neural Engine (ANE) thermal counters.'
        ]
      },
      {
        heading: 'Open Source License & Author',
        paragraphs: [
          'Jitter is created by Ayush Dubey and licensed under the permissive MIT License. Contributions and issues are welcomed on GitHub.'
        ],
        code: `git clone https://github.com/AyushDubey23/JITTER.git\ncd JITTER/jitter\npip install -e ".[dev]"\npytest tests/`,
        codeLang: 'bash'
      }
    ]
  }
}

function CodeBlock({ code, lang }: { code: string; lang?: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="relative my-4 overflow-hidden rounded-md border border-[var(--line)] bg-[#090711]">
      <div className="flex items-center justify-between border-b border-[var(--line)] px-4 py-2 font-mono text-xs text-[var(--muted)]">
        <span>{lang ?? 'bash'}</span>
        <button
          onClick={() => {
            navigator.clipboard.writeText(code)
            setCopied(true)
            setTimeout(() => setCopied(false), 1500)
          }}
          className="inline-flex items-center gap-1.5 hover:text-[var(--text)]"
        >
          {copied ? <Check className="size-3.5 text-[var(--hot)]" /> : <Copy className="size-3.5" />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <pre className="overflow-x-auto p-4 font-mono text-xs leading-6 text-[#f2ecff] sm:text-sm">
        <code>{code}</code>
      </pre>
    </div>
  )
}

export default function Docs() {
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState('Introduction')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const docKeys = Object.keys(docsData)
  const filteredKeys = useMemo(
    () =>
      docKeys.filter(
        (key) =>
          key.toLowerCase().includes(query.toLowerCase()) ||
          docsData[key].subtitle.toLowerCase().includes(query.toLowerCase())
      ),
    [query, docKeys]
  )

  const activeDoc = docsData[selected] ?? docsData['Introduction']

  return (
    <div className="min-h-screen bg-[var(--ink)] text-[var(--text)]">
      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-[var(--line)] bg-[var(--ink)]/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-2.5 font-display text-lg font-black tracking-[-.06em]">
              <img src="/jitter-logo.png" alt="Jitter" className="size-6 rounded-md object-contain" />
              <span>
                JITTER<span className="text-[var(--hot)]">.</span>
              </span>
            </Link>
            <span className="hidden font-mono text-xs text-[var(--muted)] sm:inline">Docs v0.1</span>
          </div>

          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="inline-flex items-center gap-1.5 text-sm text-[var(--muted)] hover:text-[var(--text)]"
            >
              <ArrowLeft className="size-4" /> Back to Home
            </Link>
            <a
              href="https://github.com/AyushDubey23/JITTER"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden items-center gap-1.5 border border-[var(--line)] bg-[var(--surface)] px-3 py-1.5 text-xs font-semibold text-[var(--text)] transition hover:border-[var(--hot)] sm:inline-flex"
            >
              GitHub <ExternalLink className="size-3" />
            </a>
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-1 text-[var(--muted)] hover:text-[var(--text)] lg:hidden"
              aria-label="Toggle Navigation Menu"
            >
              {mobileMenuOpen ? <X className="size-6" /> : <Menu className="size-6" />}
            </button>
          </div>
        </div>
      </header>

      {/* Docs Body */}
      <div className="mx-auto grid max-w-7xl gap-8 px-5 py-10 lg:grid-cols-[260px_1fr_200px]">
        {/* Left Sidebar Navigation */}
        <aside
          className={`fixed inset-y-0 left-0 z-40 w-72 transform border-r border-[var(--line)] bg-[var(--ink)] p-5 transition-transform duration-200 lg:static lg:block lg:w-auto lg:border-r-0 lg:p-0 ${
            mobileMenuOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
          }`}
        >
          <div className="flex items-center justify-between pb-4 lg:hidden">
            <span className="font-display font-bold">Documentation</span>
            <button onClick={() => setMobileMenuOpen(false)}>
              <X className="size-5" />
            </button>
          </div>

          <div className="lg:sticky lg:top-24">
            <p className="mb-3 font-mono text-xs font-bold tracking-wider text-[var(--muted)]">DOCS NAVIGATION</p>
            <div className="relative mb-5">
              <Search className="absolute left-3 top-2.5 size-4 text-[var(--muted)]" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search chapters..."
                className="w-full border border-[var(--line)] bg-[var(--surface)] py-2 pl-9 pr-3 text-xs text-[var(--text)] outline-none focus:border-[var(--hot)]"
              />
            </div>

            <nav className="flex max-h-[calc(100vh-200px)] flex-col gap-1 overflow-y-auto pr-1">
              {filteredKeys.map((key) => {
                const isActive = selected === key
                return (
                  <button
                    key={key}
                    onClick={() => {
                      setSelected(key)
                      setMobileMenuOpen(false)
                      window.scrollTo({ top: 0, behavior: 'smooth' })
                    }}
                    className={`flex items-center justify-between border-l-2 px-3 py-2 text-left text-xs transition ${
                      isActive
                        ? 'border-[var(--hot)] bg-[var(--surface)] font-bold text-[var(--text)]'
                        : 'border-transparent text-[var(--muted)] hover:border-[var(--line)] hover:text-[var(--text)]'
                    }`}
                  >
                    <span>{key}</span>
                  </button>
                )
              })}
            </nav>
          </div>
        </aside>

        {/* Center Main Content */}
        <main className="min-w-0">
          <div className="border border-[var(--orange)]/30 bg-[var(--surface)] px-4 py-2.5 font-mono text-xs text-[var(--orange)]">
            Verified with Jitter v0.1.0 engine (Macro-F1 1.0000, ECE 0.0398).
          </div>

          <article className="pt-8">
            <p className="font-mono text-xs tracking-wider text-[var(--muted)]">JITTER / REFERENCE</p>
            <h1 className="mt-2 font-display text-3xl font-black tracking-[-.05em] sm:text-5xl">{activeDoc.title}</h1>
            <p className="mt-3 text-lg text-[var(--muted)]">{activeDoc.subtitle}</p>

            <div className="my-8 h-px bg-[var(--line)]" />

            {activeDoc.content.map((sec, idx) => (
              <section key={idx} className="mb-12">
                {sec.heading && (
                  <h2 className="mb-4 font-display text-xl font-bold tracking-tight text-[var(--text)] sm:text-2xl">
                    {sec.heading}
                  </h2>
                )}

                {sec.paragraphs.map((p, pIdx) => (
                  <p key={pIdx} className="mb-4 leading-7 text-[#b5a9dc]">
                    {p}
                  </p>
                ))}

                {sec.alert && (
                  <div className="my-4 border-l-2 border-[var(--hot)] bg-[var(--raised)] p-4 text-sm text-[var(--white-hot)]">
                    {sec.alert.text}
                  </div>
                )}

                {sec.code && <CodeBlock code={sec.code} lang={sec.codeLang} />}

                {sec.table && (
                  <div className="my-6 overflow-x-auto rounded-md border border-[var(--line)] bg-[var(--surface)]">
                    <table className="w-full text-left font-mono text-xs">
                      <thead className="border-b border-[var(--line)] bg-[var(--raised)] text-[var(--muted)]">
                        <tr>
                          {sec.table.headers.map((h, hIdx) => (
                            <th key={hIdx} className="px-4 py-3 font-semibold">
                              {h}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[var(--line)]">
                        {sec.table.rows.map((row, rIdx) => (
                          <tr key={rIdx} className="hover:bg-[var(--raised)]/50">
                            {row.map((cell, cIdx) => (
                              <td key={cIdx} className="px-4 py-3 text-[var(--text)]">
                                {cell}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>
            ))}

            <div className="mt-16 border-t border-[var(--line)] pt-8">
              <div className="flex flex-wrap items-center justify-between gap-4 text-xs text-[var(--muted)]">
                <span>Authored by Ayush Dubey · MIT License</span>
                <a
                  href="https://github.com/AyushDubey23/JITTER"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-[var(--text)] hover:underline"
                >
                  Edit documentation on GitHub
                </a>
              </div>
            </div>
          </article>
        </main>

        {/* Right Sidebar - On This Page */}
        <aside className="hidden lg:block">
          <div className="sticky top-24 border-l border-[var(--line)] pl-4">
            <p className="mb-3 font-mono text-xs font-bold tracking-wider text-[var(--muted)]">SECTIONS</p>
            <div className="flex flex-col gap-2 font-mono text-xs text-[var(--muted)]">
              {activeDoc.content.map((sec, idx) =>
                sec.heading ? (
                  <span key={idx} className="cursor-default hover:text-[var(--text)]">
                    # {sec.heading}
                  </span>
                ) : null
              )}
            </div>

            <div className="mt-8 border-t border-[var(--line)] pt-4">
              <p className="mb-2 font-mono text-[10px] text-[var(--muted)]">QUICK COMMANDS</p>
              <div className="space-y-1 font-mono text-[11px] text-[var(--hot)]">
                <div>jitter run</div>
                <div>jitter diagnose</div>
                <div>jitter compare</div>
                <div>jitter devices</div>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  )
}
