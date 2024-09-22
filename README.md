# Music Transformer VAE

![License](https://img.shields.io/github/license/yourusername/music_transformer_vae)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7.1%2B-yellow)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Generating Music](#generating-music)
  - [Latent Space Manipulation](#latent-space-manipulation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

**Music Transformer VAE** is a powerful model that combines the strengths of Transformers and Variational Autoencoders (VAE) for generating and manipulating musical data. This project leverages PyTorch to build a model capable of capturing complex musical patterns and generating coherent and creative compositions.

## Features

- **Variational Autoencoder (VAE)** integration for latent space manipulation.
- **Multi-Head Self-Attention** mechanism for capturing long-range dependencies in music.
- **Modular Architecture** with easy customization and extension.
- **Supports MIDI Data** for training and generation.
- **Flexible Generation** allowing for latent space exploration and controlled music generation.

## Architecture

The model architecture consists of:

- **VAE Module**: Encodes input data into a latent space and decodes it back, enabling manipulation of the latent vectors.
- **Transformer Blocks**: Utilize multi-head self-attention to model dependencies in musical sequences.
- **Embedding Layers**: Handle token and positional embeddings for input data.
- **Linear Layers**: For final prediction and output generation.

![Model Architecture](docs/model_architecture.png)

## Installation

### Prerequisites

- Python 3.8 or higher
- [PyTorch](https://pytorch.org/) 1.7.1 or higher

### Clone the Repository

```bash
git clone https://github.com/yourusername/music_transformer_vae.git
cd music_transformer_vae

