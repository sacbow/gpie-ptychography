# gpie-ptychography

**gpie-ptychography** is a lightweight ptychography simulation framework designed to
support **demonstrations and benchmarks of ptychographic reconstruction using gPIE**.

The primary goal of this repository is to provide:
- reproducible forward ptychography simulations,
- clean experiment definitions,
- and minimal infrastructure needed to couple physical ptychography models to **gPIE-based inference**.

As a secondary goal, this project provides a small **domain-specific language (DSL)**
for constructing ptychography forward models as explicit computation graphs.

---

## Scope and Philosophy

This project is **not** a standalone ptychography solver.

Instead, it focuses on:
- defining forward ptychography models,
- generating diffraction data under controlled conditions,
- and serving as a testbed for message-passing–based reconstruction algorithms implemented in gPIE.

The design emphasizes clarity and extensibility over performance.

---

## Core Concepts

Ptychography forward models are represented as a **forward-only computation graph** consisting of:

- **Wave**: complex-valued wavefields (ndarray-like objects)
- **Propagator**: deterministic operators (slice, FFT, multiplication, etc.)
- **Measurement** (to be added): physical measurement models (e.g., Poisson intensity)

This structure mirrors the implicit forward/backward structure used in standard algorithms
(ePIE, rPIE), while keeping the forward model explicit and inspectable.

---

## Project Structure
ptychography/
├── backend/ # Array / FFT backend abstraction (NumPy / CuPy)
│ ├── array.py
│ ├── fft.py
│ └── rng.py
│
├── core/ # Forward computation graph primitives
│ ├── wave.py # Wave node (ndarray-like graph variable)
│ ├── propagator.py # Propagator base class
│ ├── ops.py # Arithmetic, FFT, slice propagators
│ └── shortcuts.py # NumPy-like DSL shortcuts (e.g. fft2)
│
├── data/ # Ptychography experiment data containers
│ ├── diffraction.py # DiffractionData (intensity + position)
│ └── ptychography.py # Ptychography experiment manager
│
├── utils/
│ └── types.py # Shared typing utilities (ArrayLike, etc.)
│
├── tests/ # Unit tests
│ ├── test_wave_and_propagator.py
│ ├── test_ops.py
│ ├── test_diffraction_data.py
│ └── test_ptychography.py
│
├── README.md
└── pyproject.toml

---

## Disclaimer
This project is under active development.
