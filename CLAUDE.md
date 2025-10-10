# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CS336 (Stanford) assignment focusing on implementing core components of a transformer language model from scratch. The assignment covers:

- Basic neural network operations (Linear, Embedding)
- Transformer components (Multi-head attention, RoPE, RMSNorm, SwiGLU)
- Tokenization (BPE)
- Training utilities (AdamW optimizer, gradient clipping, learning rate scheduling)
- Serialization and checkpointing

## Common Development Commands

### Environment Setup
The project uses `uv` for environment management:
```bash
# Install uv if not already installed
pip install uv

# The environment is automatically managed when using uv run
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_model.py

# Run tests with verbose output
uv run pytest -v

# Run a specific test function
uv run pytest tests/test_model.py::test_linear
```

### Code Structure
- `cs336_basics/`: Main implementation directory
- `tests/`: Test files with reference implementations
- `tests/adapters.py`: Main file to implement - contains all function stubs that need to be completed
- `data/`: Directory for dataset files (created during setup)

### Key Implementation Files
- `tests/adapters.py`: All functions that need to be implemented are here
- Each function in adapters.py corresponds to a test in the tests/ directory

### Dependencies
Main dependencies include:
- PyTorch
- NumPy
- einops (for tensor rearrangements)
- jaxtyping (for type hints)
- tiktoken (for tokenization)
- pytest (for testing)

### Development Workflow
1. Implement functions in `tests/adapters.py`
2. Run corresponding tests with `uv run pytest`
3. All tests initially fail with `NotImplementedError` - your goal is to replace these with working implementations
4. Tests use snapshot testing to verify correctness against reference implementations

### Data Setup
All required datasets have been dowloaded.

### Code Architecture
The project follows a modular approach where each transformer component is implemented separately:
- Linear and Embedding layers
- Attention mechanisms (scaled dot-product attention, multi-head attention)
- Positional encodings (RoPE)
- Normalization (RMSNorm)
- Feed-forward networks (SwiGLU)
- Full transformer blocks and language models
- Training utilities and tokenization

Each function in `adapters.py` should be implemented independently and will be tested in isolation.