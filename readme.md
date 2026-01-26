# gpie-ptychography

**gpie-ptychography** is a lightweight ptychography simulation framework designed to
support **demonstrations and benchmarks of ptychographic reconstruction using gPIE**.

The primary goal of this repository is to provide:
- reproducible forward ptychography simulations,
- clean separation between experimental context and computation graph structure,
- and a flexible testbed for developing and benchmarking ptychographic reconstruction algorithms.

This project is developed as a companion to gPIE, but it is designed to be useful independently as a forward-modeling and data-generation tool.

---

## Scope and Philosophy

This project is not intended to be a full-featured ptychography solver or a turnkey simulation package for experimental users.

Instead, it focuses on:
- explicitly defining forward ptychography models,
- generating diffraction data under controlled conditions,
- and serving as an infrastructure layer for algorithmic research on phase retrieval.

---

## Core Concepts

A ptychography experiment is represented by separating:

### 1. Experimental context
A `PtychoContext` object encapsulates all scan- and geometry-dependent information:
- scan positions
- object and probe shapes
- pixel pitch and coordinate conventions

This context defines where and how the experiment is performed.

### 2. Forward computation graph
The forward model itself is expressed as a forward-only computation graph composed of:
- `Wave`
    Complex-valued wavefields (ndarray-like symbolic variables)
- `Propagator`
    Deterministic operators such as slicing, multiplication, and FFT
- `Measurement`
    Physical measurement models (e.g. Poisson intensity measurements)

The graph is constructed using a small Python-based DSL and then executed explicitly.
This makes the forward model inspectable, debuggable, and easy to modify.

An introductory notebook demonstrating a simulation workflow is provided in the `examples/user_guide` directory.

---

## Project Structure
```
ptychography/
├── backend/        # Array / FFT / RNG backend abstraction (NumPy / CuPy)
│   ├── array.py
│   ├── fft.py
│   └── rng.py
│
├── core/           # Forward computation graph primitives
│   ├── wave.py         # Wave node (symbolic ndarray)
│   ├── propagator.py   # Propagator base class
│   ├── ops.py          # Arithmetic and FFT propagators
│   ├── slice.py        # Object slicing operator
│   ├── replicate.py   # Wave replication operator
│   ├── shortcuts.py   # NumPy-like DSL shortcuts (fft2, etc.)
│   ├── graph.py        # Graph construction and execution
│   └── model.py        # PtychoModel (graph + context binding)
│
├── data/           # Data containers
│   ├── context.py      # PtychoContext
│   └── diffraction.py  # DiffractionData
│
├── optics/         # Optics-related utilities
│   ├── aperture.py
│   └── probe.py
│
├── scan/           # Scan pattern generators
│   ├── raster.py
│   ├── fermat.py
│   └── utils.py
│
├── visualize/      # Visualization helpers (matplotlib-based)
│
├── examples/       # Tutorials and demonstration notebooks
│
├── tests/          # Unit tests
└── README.md
```

---

## Disclaimer
This project is under active development.
