# Jitter — Root-Cause Analysis for On-Device ML Latency

**Created and Engineered by Ayush Dubey**

Jitter is an architecture, Python library, and Android runner designed to diagnose inconsistent latency in on-device machine learning inference (TensorFlow Lite, ONNX Runtime).

---

## Overview

A single-number benchmark is an observation, not an explanation. The same median latency regression can be caused by:
- **Warmup drift**: Startup iterations taking significantly longer before steady-state caching and execution stabilize.
- **Thermal throttling**: Continuous load raising device temperature, causing kernel frequency downclocking.
- **Delegate fallback**: Failure to run on accelerated hardware (NNAPI / GPU), falling back silently to CPU.
- **Background noise**: External processes and OS scheduling variations introducing tail latency spikes.
- **Genuine code regressions**: Structural regressions persisting across controlled runs.

Jitter conducts four controlled reruns to separate and pinpoint the exact root causes of latency variance.

---

## Controlled Reruns Protocol

1. **Cold vs. Warm**: Distinguishes startup compilation and cache cost from steady state.
2. **Cooldown**: Allows heat dissipation before re-executing identical workloads.
3. **Delegate Swap**: Audits requested hardware delegates against actual telemetry executed on hardware.
4. **Quiet Device**: Establishes true system noise floor under regulated load.

---

## Tech Stack

- **Framework**: [Next.js](https://nextjs.org/) (App Router)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **Fonts**: Unbounded, Instrument Sans, JetBrains Mono

---

## Getting Started

### Installation

```bash
# Using npm
npm install

# Or using pnpm
pnpm install
```

### Development Server

Run the development server locally:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to view the site.

### Production Build

```bash
npm run build
npm run start
```

---

## Author & Maintainer

- **Ayush Dubey**
- License: MIT
