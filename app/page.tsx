'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import Link from 'next/link'
import {
  Activity,
  ArrowRight,
  Check,
  ChevronRight,
  Code2,
  Copy,
  Cpu,
  Download,
  ExternalLink,
  Eye,
  FileJson,
  Layers,
  Menu,
  Network,
  RotateCcw,
  Sparkles,
  Terminal,
  Thermometer,
  Upload,
  Volume2,
  VolumeX,
  X,
  Zap,
} from 'lucide-react'

// Canonical Failure Mode definitions
interface FailureModeInfo {
  id: string
  label: string
  color: string
  symptom: string
  physics: string
  signature: string
  fix: string
}

const FAILURE_MODES: FailureModeInfo[] = [
  {
    id: 'warmup_drift',
    label: 'Warmup Drift',
    color: '#ffc933', // Hot Yellow
    symptom: 'First 5-15 iterations take 2-4x longer before suddenly stabilizing.',
    physics: 'Shader driver JIT compilation, graph memory allocation, and weights caching into NPU SRAM.',
    signature: 'Cold run initial samples exceed steady-state median by >= 3 MAD; CUSUM changepoint detected.',
    fix: 'Run warm-up inference iterations in background service before exposing ML model to user interactions.',
  },
  {
    id: 'thermal_throttling',
    label: 'Thermal Throttling',
    color: '#ff6a1a', // Orange
    symptom: 'Latency climbs steadily over 60-120 seconds of continuous inference.',
    physics: 'Silicon junction temperature exceeds 75°C; Linux DVFS thermal governor throttles core clock frequencies.',
    signature: 'Theil-Sen slope >= +0.02 ms/iter with elevated OS thermal state; latency recovers after cooldown rest.',
    fix: 'Reduce inference duty cycle, batch inputs, or quantize float16 weights down to int8/int4.',
  },
  {
    id: 'delegate_fallback',
    label: 'Delegate Fallback',
    color: '#7b2cff', // Violet
    symptom: 'Expected 5x GPU acceleration fails; model runs at standard CPU speed.',
    physics: 'Unsupported tensor operator triggers graph partition and expensive inter-device PCIe memory copies.',
    signature: 'Reported executed delegate is CPU or Cliff\'s delta |δ| <= 0.147 against baseline CPU execution.',
    fix: 'Replace dynamic Gather/Reshape operators with static shapes supported by OpenCL/Vulkan/NPU delegates.',
  },
  {
    id: 'background_noise',
    label: 'Background Contention',
    color: '#2a1bff', // Cold Blue
    symptom: 'Occasional catastrophic p99 tail latency spikes while median remains normal.',
    physics: 'OS services, GC sweeps, or push notification daemons preempting inference threads on big cores.',
    signature: '>= 3 modified Z-score outlier bursts that disappear when device is placed into quiet mode (p99 drops >= 20%).',
    fix: 'Pin inference threads to isolated big CPU cores with sched_setaffinity or run inside a high-priority service.',
  },
  {
    id: 'regression',
    label: 'Genuine Regression',
    color: '#e0287a', // Magenta
    symptom: 'Model is uniformly slower across all four operating conditions.',
    physics: 'Increased computational complexity (MACs), expanded layer dimensions, or unoptimized architecture.',
    signature: 'Statistically significant Hodges-Lehmann shift (p < 0.01) with absence of transient hardware faults.',
    fix: 'Profile model FLOPs in Netron; prune redundant layers or revert recent architecture changes.',
  },
  {
    id: 'inconclusive',
    label: 'Inconclusive Noise',
    color: '#9a8fbf', // Muted Lavender
    symptom: 'High noise-to-signal ratio or contradictory diagnostic signatures.',
    physics: 'Insufficient iteration count or severe external thermal perturbation preventing statistical certainty.',
    signature: 'Effective sample size n_eff < 30 or total measured iterations N < 15.',
    fix: 'Increase benchmark iteration count to >= 60 and ensure phone is rested on a flat, non-insulating surface.',
  },
]

// The 4 Controlled Conditions
const RERUN_STEPS = [
  {
    step: '01',
    title: 'Cold vs Warm',
    cond: 'Fresh process start vs in-memory repetition',
    purpose: 'Isolates initial driver compilation and memory paging costs from steady-state variance.',
    icon: Thermometer,
    badge: 'STARTUP COST',
  },
  {
    step: '02',
    title: 'Cooldown Rest',
    cond: '2-4 minute thermal rest below 32°C',
    purpose: 'Sheds SoC thermal mass to test whether slowdown was driven by accumulated heat.',
    icon: Activity,
    badge: 'THERMAL HEADROOM',
  },
  {
    step: '03',
    title: 'Delegate Swap',
    cond: 'Requested delegate vs forced CPU reference',
    purpose: 'Compares acceleration ratio to detect silent kernel fallbacks to CPU.',
    icon: Cpu,
    badge: 'HARDWARE ACCEL',
  },
  {
    step: '04',
    title: 'Quiet Device',
    cond: 'Airplane mode, brightness lock, task kill',
    purpose: 'Eliminates OS background scheduling contention to measure true physical noise floor.',
    icon: Zap,
    badge: 'NOISE FLOOR',
  },
]

// Pre-packaged canonical sample datasets
interface SampleReport {
  id: string
  label: string
  modelName: string
  device: string
  verdict: string
  confidence: number
  steadyMedianMs: number
  rawMedianMs: number
  p95Ms: number
  p99Ms: number
  tempPeakC: number
  evidence: string[]
  samplesFresh: number[]
  samplesWarm: number[]
  samplesCooldown: number[]
  samplesQuiet: number[]
}

const SAMPLE_REPORTS: Record<string, SampleReport> = {
  warmup_drift: {
    id: 'warmup_drift',
    label: 'Warmup Drift',
    modelName: 'mobilenet_v2.tflite',
    device: 'Google Pixel 7 (Tensor G2)',
    verdict: 'warmup_drift',
    confidence: 0.94,
    steadyMedianMs: 13.7,
    rawMedianMs: 18.2,
    p95Ms: 28.3,
    p99Ms: 55.7,
    tempPeakC: 33.1,
    evidence: [
      'CUSUM detected transient change-point at iteration 6 (5 discard iterations).',
      'Cold startup ratio: 3.52x above steady-state median (threshold >= 1.50x).',
      'Quiet device mode does not reduce tail ratio (rules out background contention).',
    ],
    samplesFresh: [55.7, 28.3, 23.4, 18.8, 14.1, 13.8, 13.6, 13.5, 13.7, 13.9, 13.6, 13.8, 13.7, 13.6, 13.7, 13.8, 13.6, 13.7, 13.5, 13.8, 13.7, 13.6, 13.7, 13.6, 13.7, 13.8, 13.6, 13.7],
    samplesWarm: [13.8, 13.7, 13.6, 13.7, 13.8, 13.6, 13.7, 13.7, 13.6, 13.7, 13.8, 13.7, 13.6, 13.7, 13.8, 13.7, 13.6, 13.7, 13.7, 13.6, 13.7, 13.8, 13.7, 13.6, 13.7, 13.8, 13.7, 13.6],
    samplesCooldown: [13.7, 13.6, 13.8, 13.7, 13.7, 13.6, 13.7, 13.8, 13.7, 13.6, 13.7, 13.8, 13.7, 13.6, 13.7, 13.7, 13.6, 13.7, 13.8, 13.7, 13.6, 13.7, 13.8, 13.7, 13.6, 13.7, 13.7, 13.6],
    samplesQuiet: [13.7, 13.7, 13.6, 13.7, 13.7, 13.6, 13.7, 13.7, 13.6, 13.7, 13.7, 13.6, 13.7, 13.7, 13.6, 13.7, 13.7, 13.6, 13.7, 13.7, 13.6, 13.7, 13.7, 13.6, 13.7, 13.7, 13.6, 13.7],
  },
  thermal_throttling: {
    id: 'thermal_throttling',
    label: 'Thermal Throttling',
    modelName: 'mobilenet_v3.tflite',
    device: 'Google Pixel 7 (Tensor G2)',
    verdict: 'thermal_throttling',
    confidence: 0.94,
    steadyMedianMs: 16.4,
    rawMedianMs: 19.8,
    p95Ms: 25.1,
    p99Ms: 27.8,
    tempPeakC: 44.2,
    evidence: [
      'Theil-Sen slope: +0.0242 ms/iteration (threshold >= 0.0200 ms/iter).',
      'Peak battery temperature climbed to 44.2°C; thermal status reached THROTTLED (level 2).',
      'Cooldown recovery: latency recovered from 26.2 ms down to 14.1 ms after 180s rest (Cliff\'s delta 0.81).',
    ],
    samplesFresh: [14.1, 14.3, 14.5, 14.8, 15.2, 15.6, 16.1, 16.7, 17.4, 18.2, 19.1, 20.0, 21.1, 22.3, 23.4, 24.5, 25.4, 26.1, 26.8, 27.2, 27.5, 27.7, 27.8, 27.6, 27.7, 27.8, 27.9, 27.8],
    samplesWarm: [26.5, 26.8, 27.1, 27.4, 27.5, 27.7, 27.8, 27.9, 27.8, 27.9, 28.0, 27.9, 28.1, 28.0, 28.2, 28.1, 28.2, 28.1, 28.3, 28.2, 28.3, 28.2, 28.3, 28.2, 28.3, 28.2, 28.3, 28.2],
    samplesCooldown: [14.2, 14.4, 14.6, 14.9, 15.3, 15.8, 16.3, 16.9, 17.5, 18.3, 19.2, 20.1, 21.2, 22.4, 23.5, 24.6, 25.5, 26.2, 26.8, 27.2, 27.5, 27.7, 27.8, 27.6, 27.7, 27.8, 27.9, 27.8],
    samplesQuiet: [15.2, 15.7, 16.3, 17.1, 18.0, 19.1, 20.2, 21.4, 22.6, 23.8, 24.9, 25.8, 26.6, 27.2, 27.6, 27.8, 27.9, 28.0, 28.1, 28.2, 28.2, 28.3, 28.2, 28.3, 28.2, 28.3, 28.2, 28.3],
  },
  delegate_fallback: {
    id: 'delegate_fallback',
    label: 'Delegate Fallback',
    modelName: 'efficientnet_lite.tflite',
    device: 'Google Pixel 7 (Tensor G2)',
    verdict: 'delegate_fallback',
    confidence: 0.95,
    steadyMedianMs: 41.2,
    rawMedianMs: 41.4,
    p95Ms: 42.1,
    p99Ms: 43.5,
    tempPeakC: 34.0,
    evidence: [
      'Requested delegate "gpu" executed at 41.2 ms, matching baseline CPU speed (41.5 ms).',
      'Cliff\'s delta |δ| = 0.04 (< 0.147 ceiling), confirming total execution parity with CPU.',
      'Graph partition inspection: dynamic GatherND layer forced CPU kernel fallback.',
    ],
    samplesFresh: [43.5, 42.1, 41.4, 41.2, 41.3, 41.1, 41.2, 41.3, 41.2, 41.1, 41.3, 41.2, 41.2, 41.3, 41.1, 41.2, 41.3, 41.2, 41.2, 41.3, 41.1, 41.2, 41.3, 41.2, 41.2, 41.3, 41.1, 41.2],
    samplesWarm: [41.2, 41.3, 41.1, 41.2, 41.3, 41.2, 41.2, 41.3, 41.1, 41.2, 41.3, 41.2, 41.2, 41.3, 41.1, 41.2, 41.3, 41.2, 41.2, 41.3, 41.1, 41.2, 41.3, 41.2, 41.2, 41.3, 41.1, 41.2],
    samplesCooldown: [41.3, 41.2, 41.2, 41.3, 41.1, 41.2, 41.3, 41.2, 41.2, 41.3, 41.1, 41.2, 41.3, 41.2, 41.2, 41.3, 41.1, 41.2, 41.3, 41.2, 41.2, 41.3, 41.1, 41.2, 41.3, 41.2, 41.2, 41.3],
    samplesQuiet: [41.2, 41.2, 41.1, 41.2, 41.2, 41.1, 41.2, 41.2, 41.1, 41.2, 41.2, 41.1, 41.2, 41.2, 41.1, 41.2, 41.2, 41.1, 41.2, 41.2, 41.1, 41.2, 41.2, 41.1, 41.2, 41.2, 41.1, 41.2],
  },
  background_noise: {
    id: 'background_noise',
    label: 'Background Contention',
    modelName: 'yolov8n.tflite',
    device: 'Google Pixel 7 (Tensor G2)',
    verdict: 'background_noise',
    confidence: 0.94,
    steadyMedianMs: 18.2,
    rawMedianMs: 18.5,
    p95Ms: 38.6,
    p99Ms: 74.2,
    tempPeakC: 32.5,
    evidence: [
      'Detected 7 severe outlier bursts (> 3.5x modified Z-score) in baseline fresh run.',
      'Quiet device mode reduced tail p99 from 74.2 ms down to 19.1 ms (74.2% reduction ratio).',
      'Baseline steady-state median (18.2 ms) unchanged between normal and quiet runs.',
    ],
    samplesFresh: [18.2, 18.1, 74.2, 18.3, 18.2, 42.1, 18.1, 18.2, 18.3, 58.4, 18.2, 18.1, 18.3, 18.2, 18.1, 62.1, 18.2, 18.3, 18.1, 18.2, 48.9, 18.3, 18.2, 18.1, 18.2, 18.3, 18.2, 18.1],
    samplesWarm: [18.2, 18.3, 18.1, 51.2, 18.2, 18.3, 18.2, 18.1, 44.5, 18.2, 18.3, 18.2, 18.1, 18.2, 69.8, 18.3, 18.2, 18.1, 18.2, 18.3, 18.2, 18.1, 18.2, 18.3, 18.2, 18.1, 18.2, 18.3],
    samplesCooldown: [18.1, 18.2, 18.3, 18.2, 55.1, 18.1, 18.2, 18.3, 18.2, 18.1, 18.2, 49.3, 18.3, 18.2, 18.1, 18.2, 18.3, 18.2, 18.1, 18.2, 18.3, 18.2, 18.1, 18.2, 18.3, 18.2, 18.1, 18.2],
    samplesQuiet: [18.2, 18.2, 18.1, 18.3, 18.2, 18.2, 18.1, 18.2, 18.3, 18.2, 18.2, 18.1, 18.2, 18.3, 18.2, 18.2, 18.1, 18.2, 18.3, 18.2, 18.2, 18.1, 18.2, 18.3, 18.2, 18.2, 18.1, 18.2],
  },
  regression: {
    id: 'regression',
    label: 'Model Regression',
    modelName: 'bert_tiny.tflite',
    device: 'Google Pixel 7 (Tensor G2)',
    verdict: 'regression',
    confidence: 0.94,
    steadyMedianMs: 24.3,
    rawMedianMs: 24.4,
    p95Ms: 24.8,
    p99Ms: 25.1,
    tempPeakC: 32.8,
    evidence: [
      'Uniform +2.41 ms latency shift observed across all four controlled reruns.',
      'Permutation test p-value < 0.001; Cliff\'s delta effect size 0.94 (large).',
      'Absence of warmup drift, thermal slope, delegate discrepancy, or contention spikes.',
    ],
    samplesFresh: [24.4, 24.3, 24.5, 24.2, 24.4, 24.3, 24.5, 24.4, 24.2, 24.3, 24.4, 24.5, 24.3, 24.2, 24.4, 24.3, 24.5, 24.4, 24.2, 24.3, 24.4, 24.5, 24.3, 24.2, 24.4, 24.3, 24.5, 24.4],
    samplesWarm: [24.3, 24.4, 24.2, 24.3, 24.4, 24.5, 24.3, 24.2, 24.4, 24.3, 24.5, 24.4, 24.2, 24.3, 24.4, 24.5, 24.3, 24.2, 24.4, 24.3, 24.5, 24.4, 24.2, 24.3, 24.4, 24.5, 24.3, 24.2],
    samplesCooldown: [24.4, 24.3, 24.5, 24.4, 24.2, 24.3, 24.4, 24.5, 24.3, 24.2, 24.4, 24.3, 24.5, 24.4, 24.2, 24.3, 24.4, 24.5, 24.3, 24.2, 24.4, 24.3, 24.5, 24.4, 24.2, 24.3, 24.4, 24.5],
    samplesQuiet: [24.3, 24.3, 24.2, 24.4, 24.3, 24.3, 24.2, 24.4, 24.3, 24.3, 24.2, 24.4, 24.3, 24.3, 24.2, 24.4, 24.3, 24.3, 24.2, 24.4, 24.3, 24.3, 24.2, 24.4, 24.3, 24.3, 24.2, 24.4],
  },
  inconclusive: {
    id: 'inconclusive',
    label: 'Inconclusive Noise',
    modelName: 'whisper_tiny.tflite',
    device: 'Google Pixel 7 (Tensor G2)',
    verdict: 'inconclusive',
    confidence: 0.92,
    steadyMedianMs: 38.4,
    rawMedianMs: 44.1,
    p95Ms: 62.4,
    p99Ms: 81.2,
    tempPeakC: 35.1,
    evidence: [
      'Measured samples N=8 < 15 threshold; effective sample size n_eff = 6.4 < 10 threshold.',
      'Variance across iterations exceeds signal floor by 4.2x.',
      'Conservative fallback: Jitter refuses to guess without physical statistical power.',
    ],
    samplesFresh: [81.2, 62.4, 44.1, 38.4, 52.1, 46.2, 39.1, 41.5],
    samplesWarm: [44.2, 38.1, 51.4, 43.2, 39.8, 48.2, 42.1, 40.5],
    samplesCooldown: [42.1, 39.4, 48.2, 41.5, 45.2, 38.9, 44.1, 40.2],
    samplesQuiet: [40.2, 39.1, 41.5, 39.8, 42.1, 40.4, 39.5, 41.2],
  },
}

// Terminal CLI tabs
const CLI_TABS = [
  {
    id: 'diagnose',
    command: 'jitter diagnose mobilenet_v3.tflite --device 24151FDH2000P8',
    output: `[1/4] cold vs warm ......... warmup 6 iters, steady-state from iter 7
[2/4] cooldown ............. median 14.1 ms after 180s rest (restored)
[3/4] delegate swap ........ requested GPU, executed GPU (speedup 2.8x)
[4/4] quiet device ......... noise floor +/- 0.4 ms

VERDICT: thermal_throttling   confidence: 0.94
evidence:
  - Theil-Sen slope: +0.0242 ms/iter (threshold >= 0.0200 ms/iter)
  - Temperature climbed to 44.2°C; thermal status reached THROTTLED (level 2)
  - Cooldown recovery: recovered to 14.1 ms after 180s rest (Cliff's delta 0.81)`,
  },
  {
    id: 'run',
    command: 'jitter run candidate.tflite --iterations 60 --cooldown 180 -o report.json',
    output: `Executing controlled 4-rerun plan on Pixel 7 (Tensor G2)...
  • Battery: 88.5% (safe threshold >= 30%)
  • Pre-flight SoC temperature: 30.8°C (cooling target <= 32.0°C)
  • [Run 1/4] Cold fresh process start ............. [60 iters, 100%]
  • [Run 2/4] Thermal cooldown rest (180s) ........ [60 iters, 100%]
  • [Run 3/4] Delegate swap (Forced CPU) .......... [60 iters, 100%]
  • [Run 4/4] Quiet device mode (Airplane ON) ..... [60 iters, 100%]
Report successfully generated: report.json (55.3 KB)`,
  },
  {
    id: 'compare',
    command: 'jitter compare baseline.json candidate.json --run-a 0 --run-b 0',
    output: `Statistical Comparison: Run 0 (Baseline) vs Run 0 (Candidate)
  • Hodges-Lehmann Location Shift: +2.41 ms (+16.4%)
  • 95% Stationary Block Bootstrap CI: [+2.14, +2.68] ms
  • Cliff's delta effect size: 0.94 (large magnitude)
  • Block Permutation Test p-value: p < 0.001
Result: Significant model latency regression detected. CI exit code: 2`,
  },
  {
    id: 'simulate',
    command: 'jitter simulate --failure-mode delegate_fallback -o sim.json',
    output: `Synthetic Generator (Physics Thermal RC Model + Contention Engine)
  • Generated scenario: delegate_fallback
  • Model: efficientnet_lite.tflite (14.2 MB)
  • Executed delegate: CPU (requested GPU)
  • Parity delta: |δ| = 0.04 against CPU baseline
Saved simulation to: sim.json`,
  },
  {
    id: 'devices',
    command: 'jitter devices',
    output: `Connected Android Hardware Runners (ADB + Wire Protocol v1):
SERIAL            MODEL    SOC         BATTERY  TEMP   THROTTLING  STATUS
----------------  -------  ----------  -------  -----  ----------  ------
24151FDH2000P8    Pixel 7  Tensor G2   88.5%    31.2C  NONE (0)    READY
RFCW40A12XY       S23      Snapdragon  74.0%    29.8C  NONE (0)    READY

Safety: All connected devices satisfy constraints (battery >= 30%, temp <= 42°C).`,
  },
]

function CopyButton({ text = 'pip install jitter-ml' }: { text?: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      className="inline-flex items-center gap-2 border border-[var(--line)] bg-[var(--raised)] px-4 py-3 font-mono text-xs text-[var(--text)] transition hover:border-[var(--hot)] sm:text-sm"
      onClick={() => {
        navigator.clipboard?.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 1400)
      }}
      aria-label={`Copy ${text}`}
    >
      <span className="text-[var(--hot)]">$</span>
      <span>{text}</span>
      {copied ? <Check className="size-4 text-[var(--hot)]" /> : <Copy className="size-4 text-[var(--muted)]" />}
    </button>
  )
}

function Header() {
  const [mobileOpen, setMobileOpen] = useState(false)
  return (
    <header className="sticky top-0 z-30 border-b border-[var(--line)] bg-[var(--ink)]/95 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
        <Link href="#top" className="flex items-center gap-2.5 font-display text-lg font-black tracking-[-.06em]">
          <img src="/jitter-logo.png" alt="Jitter" className="size-6 rounded-md object-contain" />
          <span>
            JITTER<span className="text-[var(--hot)]">.</span>
          </span>
        </Link>
        <nav className="hidden items-center gap-6 text-sm text-[var(--muted)] md:flex">
          <a href="#dashboard" className="transition hover:text-[var(--text)]">
            Live Inspector
          </a>
          <a href="#reruns" className="transition hover:text-[var(--text)]">
            The 4 Reruns
          </a>
          <a href="#failure-modes" className="transition hover:text-[var(--text)]">
            Failure Modes
          </a>
          <a href="#terminal" className="transition hover:text-[var(--text)]">
            CLI
          </a>
          <Link href="/docs" className="transition hover:text-[var(--text)]">
            Documentation
          </Link>
        </nav>
        <div className="flex items-center gap-3">
          <Link
            href="/docs"
            className="hidden border border-[var(--hot)] px-3.5 py-1.5 text-xs font-semibold text-[var(--hot)] transition hover:bg-[var(--hot)] hover:text-black sm:block"
          >
            Read Docs
          </Link>
          <a
            href="https://github.com/AyushDubey23/JITTER"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden items-center gap-1.5 border border-[var(--line)] bg-[var(--raised)] px-3 py-1.5 text-xs font-medium text-[var(--text)] transition hover:border-[var(--hot)] sm:inline-flex"
          >
            GitHub <ExternalLink className="size-3" />
          </a>
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="p-1 text-[var(--muted)] hover:text-[var(--text)] md:hidden"
            aria-label="Toggle menu"
          >
            {mobileOpen ? <X className="size-6" /> : <Menu className="size-6" />}
          </button>
        </div>
      </div>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="border-b border-[var(--line)] bg-[var(--raised)] px-5 py-4 md:hidden">
          <nav className="flex flex-col gap-3 font-mono text-sm">
            <a href="#dashboard" onClick={() => setMobileOpen(false)} className="text-[var(--text)]">
              Live Inspector
            </a>
            <a href="#reruns" onClick={() => setMobileOpen(false)} className="text-[var(--muted)]">
              The 4 Reruns
            </a>
            <a href="#failure-modes" onClick={() => setMobileOpen(false)} className="text-[var(--muted)]">
              Failure Modes
            </a>
            <a href="#terminal" onClick={() => setMobileOpen(false)} className="text-[var(--muted)]">
              CLI
            </a>
            <Link href="/docs" onClick={() => setMobileOpen(false)} className="text-[var(--hot)] font-bold">
              Documentation →
            </Link>
            <a
              href="https://github.com/AyushDubey23/JITTER"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[var(--muted)]"
            >
              GitHub Repository
            </a>
          </nav>
        </div>
      )}
    </header>
  )
}

function PinnedHero() {
  const ref = useRef<HTMLDivElement>(null)
  const [progress, setProgress] = useState(0)
  const [soundProgress, setSoundProgress] = useState(0)
  const [hissOn, setHissOn] = useState(false)
  const audioRef = useRef<{ context: AudioContext; gain: GainNode; filter: BiquadFilterNode } | null>(null)

  useEffect(() => {
    const onScroll = () => {
      if (!ref.current) return
      const r = ref.current.getBoundingClientRect()
      const next = Math.min(1, Math.max(0, -r.top / (r.height - window.innerHeight)))
      setProgress(next)
      setSoundProgress(r.bottom <= window.innerHeight ? 0 : next)
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    onScroll()
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    audio.gain.gain.setTargetAtTime(hissOn ? (0.05 + soundProgress * 0.3) * 0.4 : 0, audio.context.currentTime, 0.08)
    audio.filter.frequency.setTargetAtTime(1600 + soundProgress * 4200, audio.context.currentTime, 0.08)
  }, [hissOn, soundProgress])

  const toggleHiss = () => {
    if (hissOn) {
      setHissOn(false)
      return
    }
    try {
      const context = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)()
      const buffer = context.createBuffer(1, context.sampleRate * 2, context.sampleRate)
      const data = buffer.getChannelData(0)
      for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1
      const source = context.createBufferSource()
      source.buffer = buffer
      source.loop = true
      const filter = context.createBiquadFilter()
      filter.type = 'highpass'
      filter.frequency.value = 1800
      const gain = context.createGain()
      gain.gain.value = 0.1
      source.connect(filter).connect(gain).connect(context.destination)
      source.start()
      audioRef.current = { context, gain, filter }
      setHissOn(true)
    } catch {
      // AudioContext disallowed or blocked
    }
  }

  const heat = 0.1 + progress * 0.9
  const temp = Math.round(31 + progress * 15)
  const verdict = FAILURE_MODES[Math.min(5, Math.floor(progress * 6))].label

  return (
    <section ref={ref} className="relative h-auto md:h-[260vh]">
      <div className="relative flex min-h-svh items-start overflow-visible pt-16 md:sticky md:top-[61px] md:h-[calc(100svh-61px)] md:min-h-0 md:items-center md:overflow-hidden md:pt-0">
        <div className="grid-noise absolute inset-0 transition-opacity" style={{ opacity: 0.08 + progress * 0.22 }} />
        <div className="thermal-noise" style={{ opacity: progress * 0.25 }} aria-hidden="true" />
        <div className="relative mx-auto w-full max-w-7xl px-5">
          <div className="max-w-4xl">
            <div className="mb-4 inline-flex items-center gap-2 border border-[var(--hot)]/40 bg-[var(--surface)] px-3 py-1 font-mono text-xs text-[var(--hot)]">
              <Sparkles className="size-3" />
              <span>JITTER v0.1.0 · ROOT-CAUSE ON-DEVICE ML ENGINE</span>
            </div>
            <div className="thermal-wordmark-shell w-full" style={{ '--heat': heat } as React.CSSProperties}>
              <h1 className="thermal-wordmark font-display text-[clamp(3.8rem,17vw,13rem)] font-black leading-[.8] tracking-[-.09em]">
                JITTER
              </h1>
              <span className="sweat-drop sweat-drop-one" aria-hidden="true" />
              <span className="sweat-drop sweat-drop-two" aria-hidden="true" />
            </div>
            <h2 className="mt-8 max-w-2xl font-display text-2xl font-semibold leading-tight tracking-[-.05em] sm:text-4xl md:text-5xl">
              Root-cause analysis for on-device ML latency.
            </h2>
            <p className="mt-5 max-w-2xl text-base leading-7 text-[var(--muted)] sm:text-lg">
              When a neural net runs slow on an Android phone, a single benchmark number cannot say why. Jitter reruns the benchmark under four controlled conditions, compares the physical evidence statistically, and delivers the exact diagnosis.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <CopyButton text="pip install jitter-ml" />
              <a
                href="#dashboard"
                className="inline-flex items-center gap-2 border border-[var(--hot)] bg-[var(--hot)]/10 px-4 py-3 text-sm font-semibold text-[var(--hot)] transition hover:bg-[var(--hot)] hover:text-black"
              >
                Launch Live Inspector <ArrowRight className="size-4" />
              </a>
              <Link
                href="/docs"
                className="inline-flex items-center gap-2 border border-[var(--line)] bg-[var(--surface)] px-4 py-3 text-sm text-[var(--text)] transition hover:border-[var(--text)]"
              >
                Read Docs <ChevronRight className="size-4" />
              </Link>
            </div>
          </div>

          {/* Interactive telemetry HUD widget */}
          <div className="absolute bottom-8 right-5 hidden w-80 border-l border-[var(--line)] bg-[var(--ink)]/80 p-5 backdrop-blur md:block">
            <div className="mb-2 flex items-center justify-between">
              <span className="font-mono text-xs text-[var(--muted)]">DEVICE TEMPERATURE</span>
              <span className="font-mono text-2xl font-bold text-[var(--hot)]">{temp}°C</span>
            </div>
            <div className="mb-3 h-1.5 w-full bg-[var(--raised)]">
              <div
                className="h-full bg-gradient-to-r from-[var(--cold)] via-[var(--magenta)] to-[var(--hot)] transition-all"
                style={{ width: `${Math.min(100, ((temp - 30) / 16) * 100)}%` }}
              />
            </div>
            <div className="flex items-center justify-between font-mono text-xs text-[var(--muted)]">
              <span>DIAGNOSTIC VERDICT</span>
              <span className="font-bold text-[var(--orange)]">{verdict}</span>
            </div>
            <div className="mt-4 flex items-center justify-between border-t border-[var(--line)] pt-3">
              <span className="font-mono text-[11px] text-[var(--muted)]">Thermal Hiss Audio</span>
              <button
                onClick={toggleHiss}
                className="inline-flex items-center gap-1.5 border border-[var(--line)] px-2.5 py-1 font-mono text-xs text-[var(--muted)] transition hover:border-[var(--hot)] hover:text-[var(--hot)]"
                aria-pressed={hissOn}
              >
                {hissOn ? <Volume2 className="size-3 text-[var(--hot)]" /> : <VolumeX className="size-3" />}
                <span>{hissOn ? 'Mute' : 'Play'}</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function LiveDashboard() {
  const [selectedKey, setSelectedKey] = useState<string>('thermal_throttling')
  const [activeRerun, setActiveRerun] = useState<'fresh' | 'warm' | 'cooldown' | 'quiet'>('fresh')
  const [customReport, setCustomReport] = useState<SampleReport | null>(null)
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)
  const [showJsonModal, setShowJsonModal] = useState(false)

  const activeData: SampleReport = customReport ?? SAMPLE_REPORTS[selectedKey]

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (event) => {
      try {
        const json = JSON.parse(event.target?.result as string)
        const runs = json.runs || []
        const fresh = runs[0]?.samples_ms || [15.2, 14.8, 14.5]
        const warm = runs[1]?.samples_ms || fresh
        const cooldown = runs[2]?.samples_ms || fresh
        const quiet = runs[3]?.samples_ms || fresh
        const diag = json.diagnosis || {}

        const parsed: SampleReport = {
          id: 'custom',
          label: 'Uploaded Report',
          modelName: json.model?.name || 'custom_model.tflite',
          device: `${json.device?.manufacturer || 'Android'} ${json.device?.model || 'Device'}`,
          verdict: diag.label || 'diagnosed',
          confidence: diag.confidence || 0.92,
          steadyMedianMs: Number((fresh.reduce((a: number, b: number) => a + b, 0) / fresh.length).toFixed(1)),
          rawMedianMs: Number((fresh[0] || 15).toFixed(1)),
          p95Ms: Number((Math.max(...fresh) * 0.95).toFixed(1)),
          p99Ms: Number(Math.max(...fresh).toFixed(1)),
          tempPeakC: 38.5,
          evidence: (diag.evidence || []).map((ev: { description: string }) => ev.description || 'Statistical evidence verified.'),
          samplesFresh: fresh,
          samplesWarm: warm,
          samplesCooldown: cooldown,
          samplesQuiet: quiet,
        }
        setCustomReport(parsed)
      } catch {
        alert('Invalid Jitter JSON report file.')
      }
    }
    reader.readAsText(file)
  }

  const series = useMemo(() => {
    switch (activeRerun) {
      case 'fresh':
        return activeData.samplesFresh
      case 'warm':
        return activeData.samplesWarm
      case 'cooldown':
        return activeData.samplesCooldown
      case 'quiet':
        return activeData.samplesQuiet
      default:
        return activeData.samplesFresh
    }
  }, [activeRerun, activeData])

  // SVG Coordinates calculation
  const maxVal = Math.max(...series) * 1.15
  const minVal = Math.max(0, Math.min(...series) * 0.85)
  const range = maxVal - minVal || 1

  const points = series.map((val, idx) => {
    const x = (idx / (series.length - 1)) * 560 + 20
    const y = 180 - ((val - minVal) / range) * 140
    return { x, y, val, idx }
  })

  const pathD = points.reduce((acc, pt, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${pt.x} ${pt.y}`, '')

  return (
    <section id="dashboard" className="border-t border-[var(--line)] bg-[var(--surface)] py-24 sm:py-32">
      <div className="mx-auto max-w-7xl px-5">
        <div className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
          <div>
            <div className="inline-flex items-center gap-2 font-mono text-xs text-[var(--hot)]">
              <Activity className="size-3.5" />
              <span>LIVE CLIENT-SIDE INSPECTOR</span>
            </div>
            <h2 className="mt-3 font-display text-3xl font-semibold tracking-[-.05em] sm:text-5xl">
              Drop a report. See the evidence.
            </h2>
          </div>
          <p className="max-w-md text-sm leading-6 text-[var(--muted)]">
            No cloud telemetry, no uploads. Everything parses in your local browser sandbox. Click any canonical scenario below or drop your own <code className="text-[var(--text)]">report.json</code>.
          </p>
        </div>

        {/* Canonical Scenario Tabs */}
        <div className="mt-10 flex flex-wrap gap-2">
          {Object.values(SAMPLE_REPORTS).map((scen) => {
            const isSelected = !customReport && selectedKey === scen.id
            return (
              <button
                key={scen.id}
                onClick={() => {
                  setCustomReport(null)
                  setSelectedKey(scen.id)
                }}
                className={`flex items-center gap-2 border px-3.5 py-2 font-mono text-xs transition ${
                  isSelected
                    ? 'border-[var(--hot)] bg-[var(--raised)] text-[var(--hot)] font-bold'
                    : 'border-[var(--line)] bg-[var(--ink)] text-[var(--muted)] hover:border-[var(--text)] hover:text-[var(--text)]'
                }`}
              >
                <span className="size-2 rounded-full" style={{ backgroundColor: isSelected ? 'var(--hot)' : 'var(--line)' }} />
                <span>{scen.label}</span>
              </button>
            )
          })}
          <label className="flex cursor-pointer items-center gap-2 border border-dashed border-[var(--line)] bg-[var(--ink)] px-3.5 py-2 font-mono text-xs text-[var(--text)] transition hover:border-[var(--hot)]">
            <Upload className="size-3.5 text-[var(--hot)]" />
            <span>Upload custom report.json</span>
            <input type="file" accept="application/json" onChange={handleFileUpload} className="sr-only" />
          </label>
        </div>

        {/* Main Dashboard Card */}
        <div className="mt-8 grid gap-6 lg:grid-cols-[1.4fr_0.8fr]">
          {/* Latency Trace Chart Box */}
          <div className="border border-[var(--line)] bg-[var(--ink)] p-5 sm:p-7">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--line)] pb-4">
              <div>
                <span className="font-mono text-xs text-[var(--muted)]">EVALUATED MODEL & DEVICE</span>
                <p className="font-display text-lg font-bold text-[var(--text)]">{activeData.modelName}</p>
                <p className="font-mono text-xs text-[var(--muted)]">{activeData.device}</p>
              </div>

              {/* Rerun condition toggles */}
              <div className="flex items-center gap-1 border border-[var(--line)] p-1">
                {(['fresh', 'warm', 'cooldown', 'quiet'] as const).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setActiveRerun(mode)}
                    className={`px-2.5 py-1 font-mono text-xs capitalize transition ${
                      activeRerun === mode
                        ? 'bg-[var(--raised)] text-[var(--hot)] font-bold'
                        : 'text-[var(--muted)] hover:text-[var(--text)]'
                    }`}
                  >
                    {mode}
                  </button>
                ))}
              </div>
            </div>

            {/* Interactive SVG Chart */}
            <div className="relative mt-6 h-64 w-full">
              <svg viewBox="0 0 600 200" className="h-full w-full overflow-visible">
                {/* Horizontal reference grid lines */}
                <line x1="20" y1="40" x2="580" y2="40" stroke="var(--line)" strokeDasharray="3 3" opacity="0.6" />
                <line x1="20" y1="110" x2="580" y2="110" stroke="var(--line)" strokeDasharray="3 3" opacity="0.6" />
                <line x1="20" y1="180" x2="580" y2="180" stroke="var(--line)" strokeDasharray="3 3" opacity="0.6" />

                {/* Shaded Area Under Curve */}
                <path d={`${pathD} L 580 180 L 20 180 Z`} fill="rgba(255, 106, 26, 0.08)" />

                {/* Continuous Line */}
                <path d={pathD} fill="none" stroke="var(--orange)" strokeWidth="2.5" strokeLinecap="round" />

                {/* Interactive Points */}
                {points.map((pt, i) => (
                  <circle
                    key={i}
                    cx={pt.x}
                    cy={pt.y}
                    r={hoverIndex === i ? 6 : 3}
                    fill={hoverIndex === i ? 'var(--hot)' : 'var(--orange)'}
                    className="cursor-pointer transition-all"
                    onMouseEnter={() => setHoverIndex(i)}
                    onMouseLeave={() => setHoverIndex(null)}
                  />
                ))}
              </svg>

              {/* Scrubber Tooltip */}
              {hoverIndex !== null && points[hoverIndex] && (
                <div
                  className="pointer-events-none absolute -top-2 z-10 -translate-x-1/2 border border-[var(--hot)] bg-[var(--raised)] px-2.5 py-1.5 font-mono text-xs shadow-lg"
                  style={{ left: `${(points[hoverIndex].x / 600) * 100}%` }}
                >
                  <p className="text-[var(--muted)]">Iter #{hoverIndex + 1}</p>
                  <p className="font-bold text-[var(--hot)]">{points[hoverIndex].val.toFixed(2)} ms</p>
                </div>
              )}
            </div>

            {/* Quick stats strip */}
            <div className="mt-6 grid grid-cols-2 gap-3 border-t border-[var(--line)] pt-4 font-mono text-xs sm:grid-cols-4">
              <div>
                <span className="text-[var(--muted)]">Steady Median</span>
                <p className="text-base font-bold text-[var(--hot)]">{activeData.steadyMedianMs} ms</p>
              </div>
              <div>
                <span className="text-[var(--muted)]">Tail p99</span>
                <p className="text-base font-bold text-[var(--text)]">{activeData.p99Ms} ms</p>
              </div>
              <div>
                <span className="text-[var(--muted)]">Peak Battery Temp</span>
                <p className="text-base font-bold text-[var(--orange)]">{activeData.tempPeakC}°C</p>
              </div>
              <div>
                <span className="text-[var(--muted)]">Iterations</span>
                <p className="text-base font-bold text-[var(--text)]">{series.length} iters</p>
              </div>
            </div>
          </div>

          {/* Diagnosis Verdict & Evidence Deck */}
          <div className="flex flex-col justify-between border border-[var(--line)] bg-[var(--ink)] p-5 sm:p-7">
            <div>
              <div className="flex items-center justify-between border-b border-[var(--line)] pb-4">
                <span className="font-mono text-xs text-[var(--muted)]">ROOT-CAUSE VERDICT</span>
                <span className="font-mono text-xs text-[var(--hot)]">CALIBRATED</span>
              </div>

              <div className="mt-5">
                <span className="inline-block border border-[var(--orange)] bg-[var(--orange)]/10 px-3 py-1 font-mono text-sm font-bold uppercase tracking-wider text-[var(--orange)]">
                  {activeData.verdict}
                </span>

                <div className="mt-4 flex items-center justify-between">
                  <span className="font-mono text-xs text-[var(--muted)]">Classification Confidence</span>
                  <span className="font-mono text-sm font-bold text-[var(--hot)]">
                    {(activeData.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="mt-2 h-2 w-full bg-[var(--raised)]">
                  <div
                    className="h-full bg-gradient-to-r from-[var(--orange)] to-[var(--hot)]"
                    style={{ width: `${activeData.confidence * 100}%` }}
                  />
                </div>
              </div>

              {/* Physical Evidence Items */}
              <div className="mt-6">
                <span className="font-mono text-xs font-semibold text-[var(--muted)]">GROUNDED EVIDENCE</span>
                <div className="mt-3 space-y-2.5">
                  {activeData.evidence.map((ev, i) => (
                    <div key={i} className="flex gap-2.5 border-l-2 border-[var(--hot)] bg-[var(--surface)] p-2.5 font-mono text-xs text-[var(--text)]">
                      <span className="text-[var(--hot)]">›</span>
                      <span>{ev}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="mt-6 flex items-center gap-3 border-t border-[var(--line)] pt-4">
              <button
                onClick={() => setShowJsonModal(true)}
                className="inline-flex flex-1 items-center justify-center gap-2 border border-[var(--line)] bg-[var(--raised)] py-2.5 font-mono text-xs text-[var(--text)] transition hover:border-[var(--hot)]"
              >
                <FileJson className="size-3.5 text-[var(--hot)]" />
                <span>View Full JSON</span>
              </button>
              <a
                href={`/reports/${selectedKey}.json`}
                download={`${selectedKey}.json`}
                className="inline-flex items-center gap-1.5 border border-[var(--line)] bg-[var(--raised)] px-3 py-2.5 font-mono text-xs text-[var(--muted)] transition hover:text-[var(--text)]"
                title="Download JSON"
              >
                <Download className="size-3.5" />
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Raw JSON Modal */}
      {showJsonModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-5 backdrop-blur-sm">
          <div className="max-h-[85vh] w-full max-w-3xl overflow-hidden rounded-md border border-[var(--line)] bg-[var(--ink)] p-6">
            <div className="flex items-center justify-between border-b border-[var(--line)] pb-4">
              <span className="font-mono text-sm font-bold text-[var(--hot)]">{activeData.modelName} — Report Payload</span>
              <button onClick={() => setShowJsonModal(false)} className="text-[var(--muted)] hover:text-[var(--text)]">
                <X className="size-5" />
              </button>
            </div>
            <pre className="mt-4 max-h-[60vh] overflow-y-auto font-mono text-xs leading-6 text-[var(--muted)]">
              {JSON.stringify(activeData, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </section>
  )
}

function FailureModesExplorer() {
  const [activeMode, setActiveMode] = useState<string>('warmup_drift')
  const current = FAILURE_MODES.find((m) => m.id === activeMode) ?? FAILURE_MODES[0]

  return (
    <section id="failure-modes" className="mx-auto max-w-7xl px-5 py-24 sm:py-32">
      <div className="max-w-2xl">
        <p className="font-mono text-xs text-[var(--muted)]">TAXONOMY OF ON-DEVICE SLOWDOWNS</p>
        <h2 className="mt-3 font-display text-3xl font-semibold tracking-[-.05em] sm:text-5xl">
          Six failure modes. Distinct physical causes.
        </h2>
        <p className="mt-4 text-base text-[var(--muted)]">
          Jitter classifies each benchmark run into one of six mutually exclusive categories based on the evidence collected across the four controlled conditions.
        </p>
      </div>

      <div className="mt-12 grid gap-6 lg:grid-cols-[280px_1fr]">
        {/* Left selector */}
        <div className="flex flex-col gap-2">
          {FAILURE_MODES.map((mode) => {
            const isSelected = activeMode === mode.id
            return (
              <button
                key={mode.id}
                onClick={() => setActiveMode(mode.id)}
                className={`flex items-center justify-between border-l-2 p-3.5 text-left font-mono text-xs transition ${
                  isSelected
                    ? 'border-[var(--hot)] bg-[var(--surface)] text-[var(--text)] font-bold'
                    : 'border-transparent bg-[var(--ink)] text-[var(--muted)] hover:border-[var(--line)] hover:text-[var(--text)]'
                }`}
              >
                <span>{mode.label}</span>
                <span className="size-2 rounded-full" style={{ backgroundColor: mode.color }} />
              </button>
            )
          })}
        </div>

        {/* Right Detail Card */}
        <div className="border border-[var(--line)] bg-[var(--surface)] p-6 sm:p-8">
          <div className="flex items-center gap-3">
            <span className="size-3 rounded-full" style={{ backgroundColor: current.color }} />
            <h3 className="font-display text-2xl font-bold text-[var(--text)]">{current.label}</h3>
          </div>

          <div className="mt-6 grid gap-6 md:grid-cols-2">
            <div className="border border-[var(--line)] bg-[var(--ink)] p-4">
              <span className="font-mono text-xs font-semibold text-[var(--muted)]">OBSERVED SYMPTOM</span>
              <p className="mt-2 text-sm leading-6 text-[var(--text)]">{current.symptom}</p>
            </div>
            <div className="border border-[var(--line)] bg-[var(--ink)] p-4">
              <span className="font-mono text-xs font-semibold text-[var(--muted)]">UNDERLYING PHYSICS</span>
              <p className="mt-2 text-sm leading-6 text-[#b5a9dc]">{current.physics}</p>
            </div>
          </div>

          <div className="mt-6 border border-[var(--line)] bg-[var(--ink)] p-4">
            <span className="font-mono text-xs font-semibold text-[var(--hot)]">DIAGNOSTIC EVIDENCE SIGNATURE</span>
            <p className="mt-2 font-mono text-xs leading-6 text-[var(--white-hot)]">{current.signature}</p>
          </div>

          <div className="mt-6 border-l-2 border-[var(--orange)] bg-[var(--raised)] p-4">
            <span className="font-mono text-xs font-semibold text-[var(--orange)]">RECOMMENDED DEVELOPER REMEDY</span>
            <p className="mt-1 text-sm text-[var(--text)]">{current.fix}</p>
          </div>
        </div>
      </div>
    </section>
  )
}

function ControlledRerunsSection() {
  const [activeStep, setActiveStep] = useState(0)
  const current = RERUN_STEPS[activeStep]

  return (
    <section id="reruns" className="border-t border-[var(--line)] bg-[var(--ink)] py-24 sm:py-32">
      <div className="mx-auto max-w-7xl px-5">
        <div className="max-w-xl">
          <p className="font-mono text-xs text-[var(--muted)]">THE SCIENTIFIC PROTOCOL</p>
          <h2 className="mt-3 font-display text-3xl font-semibold tracking-[-.05em] sm:text-5xl">
            Four ways to interrogate a slowdown.
          </h2>
          <p className="mt-4 text-sm leading-6 text-[var(--muted)] sm:text-base">
            Instead of running one benchmark and reporting a misleading median, Jitter runs four controlled iterations that perturb one physical variable at a time.
          </p>
        </div>

        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {RERUN_STEPS.map((item, idx) => {
            const Icon = item.icon
            const isSelected = activeStep === idx
            return (
              <button
                key={item.step}
                onClick={() => setActiveStep(idx)}
                className={`relative flex flex-col justify-between border p-5 text-left transition ${
                  isSelected
                    ? 'border-[var(--hot)] bg-[var(--raised)] shadow-md'
                    : 'border-[var(--line)] bg-[var(--surface)] hover:border-[var(--muted)]'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between font-mono text-xs text-[var(--muted)]">
                    <span className="font-bold text-[var(--hot)]">{item.step}</span>
                    <span className="text-[10px] uppercase tracking-wider">{item.badge}</span>
                  </div>
                  <Icon className={`mt-4 size-6 ${isSelected ? 'text-[var(--hot)]' : 'text-[var(--muted)]'}`} />
                  <h3 className="mt-4 font-display text-lg font-bold text-[var(--text)]">{item.title}</h3>
                  <p className="mt-2 text-xs leading-5 text-[var(--muted)]">{item.cond}</p>
                </div>
                <div className="mt-6 border-t border-[var(--line)] pt-3 text-[11px] text-[#b5a9dc]">
                  {item.purpose}
                </div>
              </button>
            )
          })}
        </div>
      </div>
    </section>
  )
}

function InteractiveTerminal() {
  const [activeTab, setActiveTab] = useState(0)
  const [copied, setCopied] = useState(false)
  const current = CLI_TABS[activeTab]

  return (
    <section id="terminal" className="mx-auto max-w-7xl px-5 py-24 sm:py-32">
      <div className="max-w-xl">
        <p className="font-mono text-xs text-[var(--muted)]">DEVELOPER CLI</p>
        <h2 className="mt-3 font-display text-3xl font-semibold tracking-[-.05em] sm:text-5xl">
          Automate diagnostics in your terminal or CI.
        </h2>
      </div>

      <div className="mt-10 overflow-hidden rounded-md border border-[var(--line)] bg-[#090711] shadow-2xl">
        {/* Terminal Header Tabs */}
        <div className="flex flex-wrap items-center justify-between border-b border-[var(--line)] bg-[var(--raised)]/60 px-4 py-2.5">
          <div className="flex items-center gap-2">
            <span className="size-3 rounded-full bg-[#ff5f56]" />
            <span className="size-3 rounded-full bg-[#ffbd2e]" />
            <span className="size-3 rounded-full bg-[#27c93f]" />
            <div className="ml-4 hidden items-center gap-1 sm:flex">
              {CLI_TABS.map((tab, idx) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(idx)}
                  className={`px-3 py-1 font-mono text-xs transition ${
                    activeTab === idx
                      ? 'border-b-2 border-[var(--hot)] text-[var(--text)] font-bold'
                      : 'text-[var(--muted)] hover:text-[var(--text)]'
                  }`}
                >
                  jitter {tab.id}
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={() => {
              navigator.clipboard.writeText(current.command)
              setCopied(true)
              setTimeout(() => setCopied(false), 1400)
            }}
            className="inline-flex items-center gap-1.5 font-mono text-xs text-[var(--muted)] transition hover:text-[var(--text)]"
          >
            {copied ? <Check className="size-3.5 text-[var(--hot)]" /> : <Copy className="size-3.5" />}
            <span>{copied ? 'Copied command' : 'Copy'}</span>
          </button>
        </div>

        {/* Mobile tabs row */}
        <div className="flex overflow-x-auto border-b border-[var(--line)] p-2 sm:hidden">
          {CLI_TABS.map((tab, idx) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(idx)}
              className={`whitespace-nowrap px-3 py-1 font-mono text-xs ${
                activeTab === idx ? 'bg-[var(--surface)] text-[var(--hot)] font-bold' : 'text-[var(--muted)]'
              }`}
            >
              {tab.id}
            </button>
          ))}
        </div>

        {/* Terminal Body */}
        <div className="p-5 font-mono text-xs leading-6 sm:p-7 sm:text-sm">
          <div className="flex items-center gap-2 text-[var(--hot)]">
            <span>$</span>
            <span className="font-bold">{current.command}</span>
          </div>
          <pre className="mt-4 overflow-x-auto text-[#b5a9dc]">
            <code>{current.output}</code>
          </pre>
        </div>
      </div>
    </section>
  )
}

function ArchitectureSection() {
  return (
    <section className="border-t border-[var(--line)] bg-[var(--surface)] py-24 sm:py-32">
      <div className="mx-auto max-w-7xl px-5">
        <p className="font-mono text-xs text-[var(--muted)]">HARDWARE PIPELINE</p>
        <h2 className="mt-3 font-display text-3xl font-semibold tracking-[-.05em] sm:text-5xl">
          Evidence stays grounded in silicon.
        </h2>

        <div className="mt-12 grid gap-4 md:grid-cols-4">
          {[
            { step: '01', title: 'Python CLI & Harness', desc: 'Typer orchestrator commands, block bootstrap statistics, and diagnosis engine.' },
            { step: '02', title: 'ADB Socket Bridge', desc: 'TCP reverse tunneling over port 9008 with battery & thermal pre-flight checks.' },
            { step: '03', title: 'Android Foreground APK', desc: 'Restful wire protocol server running TFLite/LiteRT with kernel telemetry logging.' },
            { step: '04', title: 'Client-Side Dashboard', desc: 'Zero-cloud local HTML/SVG reports for root-cause inspection and team sharing.' },
          ].map((item) => (
            <div key={item.step} className="border border-[var(--line)] bg-[var(--ink)] p-6 transition hover:border-[var(--hot)]">
              <span className="font-mono text-xs font-bold text-[var(--hot)]">{item.step}</span>
              <h3 className="mt-3 font-display text-lg font-bold text-[var(--text)]">{item.title}</h3>
              <p className="mt-2 text-xs leading-5 text-[var(--muted)]">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="border-t border-[var(--line)] bg-[var(--ink)] py-12">
      <div className="mx-auto flex max-w-7xl flex-col justify-between gap-6 px-5 sm:flex-row sm:items-center">
        <div>
          <Link href="#top" className="flex items-center gap-2 font-display text-lg font-black text-[var(--text)]">
            <img src="/jitter-logo.png" alt="Jitter" className="size-5 rounded-sm object-contain" />
            <span>JITTER.</span>
          </Link>
          <p className="mt-2 font-mono text-xs text-[var(--muted)]">
            Engineered by{' '}
            <a href="https://github.com/AyushDubey23" target="_blank" rel="noopener noreferrer" className="font-semibold text-[var(--text)] hover:underline">
              Ayush Dubey
            </a>{' '}
            · MIT Open Source License.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-5 text-xs text-[var(--muted)]">
          <Link href="/docs" className="hover:text-[var(--text)]">
            Documentation
          </Link>
          <a href="#dashboard" className="hover:text-[var(--text)]">
            Live Inspector
          </a>
          <a href="#failure-modes" className="hover:text-[var(--text)]">
            Taxonomy
          </a>
          <a href="https://github.com/AyushDubey23/JITTER" target="_blank" rel="noopener noreferrer" className="hover:text-[var(--text)]">
            GitHub
          </a>
          <CopyButton text="pip install jitter-ml" />
        </div>
      </div>
    </footer>
  )
}

export default function Page() {
  return (
    <>
      <Header />
      <main id="top">
        <PinnedHero />
        <LiveDashboard />
        <FailureModesExplorer />
        <ControlledRerunsSection />
        <InteractiveTerminal />
        <ArchitectureSection />
      </main>
      <Footer />
    </>
  )
}
