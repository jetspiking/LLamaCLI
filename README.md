# LLamaCLI
LLama CLI Binaries Based on LLamaSharp

# Description
This project contains multiple binaries that can be considered plug & play. Due to using LLamaSharp as a foundation, the correct installation of dependencies is no longer an issue.

# Runtimes
The following runtimes are supported:
- Windows
- Linux
- MacOS

The following platforms are supported:
- CPU (AVX, no AVX)
- GPU (CUDA Windows, CUDA Linux)

Currently only CUDA 12 is supported.

# Usage
```sh
LLamaCLI --modelFile <path> --outputFile <path> --inputText <text> [options]
```
## Required Arguments:
- `--modelFile` : Path to the LLama model file.
- `--outputFile` : Path to the output file.
- `--inputText` : Input text for processing.

## GPU & Backend Options:
- `--cuda` : Enable CUDA backend.
- `--vulkan` : Enable Vulkan backend.
- `--autoFallback` : Allow auto-fallback.
- `--skipCheck` : Skip validation check.
- `--avx <level>` : AVX level (`None`, `Avx`, `Avx2`, `Avx512`).

## Model Parameters:
- `--contextSize <int>` : Set context size.
- `--mainGpu <int>` : Set main GPU ID.
- `--gpuLayerCount <int>` : Set GPU layer count.
- `--threads <int>` : Set number of threads.
- `--batchThreads <int>` : Set batch threads.
- `--batchSize <int>` : Set batch size.
- `--uBatchSize <int>` : Set U-batch size.
- `--useMemorymap` : Enable memory mapping.
- `--useMemoryLock` : Enable memory locking.

## Inference Parameters:
- `--maxTokens <int>` : Maximum tokens to generate.
- `--tokensKeep <int>` : Number of tokens to retain.
- `--antiPrompts <text>` : Comma-separated anti-prompts.

## Advanced Model Parameters:
- `--splitMode <mode>` : GPU split mode.
- `--seqMax <int>` : Set max sequence length.
- `--ropeFrequencyBase <float>` : Set Rope frequency base.
- `--ropeFrequencyScale <float>` : Set Rope frequency scale.
- `--flashAttention` : Enable FlashAttention.

## General:
- `--help` : Show this help message.
