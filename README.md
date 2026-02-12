# Secret Hitler LLM Simulator

<div align="center">

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->

*Deception, Persuasion, and Trust: Evaluating Large
Language Models in a Complex Hidden Role Game*

[Installation](#installation) •
[Quick Start](#quick-start) •
[Results](#experimental-results)

</div>

---

## Abstract

This repository contains the implementation and evaluation framework for studying Large Language Model (LLM) behavior in **Secret Hitler**, a social deduction game involving deception, persuasion, and strategic reasoning. We present a comprehensive simulator that enables LLMs to play Secret Hitler autonomously, along with extensive analysis tools for evaluating their performance across multiple dimensions including strategic decision-making, persuasive communication, and deceptive behavior.

This framework supports:

- **Multi-Agent LLM Gameplay**: Autonomous LLM players with configurable reasoning strategies
- **Persuasion Analysis**: Automated annotation of persuasive techniques based on Cialdini's principles
- **Deception Detection**: Measurement and analysis of deceptive behavior patterns
- **Human Game Crawling**: Data collection from real human games for baseline comparison
- **Comprehensive Evaluation**: Statistical analysis, visualization, and comparison across models

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Running Simulations](#running-simulations)
  - [Analyzing Results](#analyzing-results)
  - [Persuasion Annotation](#persuasion-annotation)
  - [Crawling Human Games](#crawling-human-games)
- [Player Types](#player-types)
- [Evaluation Metrics](#evaluation-metrics)
- [Configuration](#configuration)
- [Experimental Results](#experimental-results)
- [Acknowledgments](#acknowledgments)

## Features

### 🎮 **Game Simulation**

- Full implementation of Secret Hitler game mechanics
- Support for multiple LLM backends (OpenAI, vLLM, local models)
- Configurable game parameters and player compositions
- Detailed logging of game states, actions, and communications

### 🧠 **LLM Player Strategies**

- **Base LLM Player**: Standard prompted gameplay
- **Chain-of-Thought (CoT)**: Explicit reasoning traces
- **Memory-Enhanced**: Episodic memory of game events
- **Role-Aware Messaging**: Context-sensitive communication
- **Strategy-Guided**: Integration of expert strategy documents
- **Full Configuration**: Combined advanced features

### 📊 **Analysis & Evaluation**

- Win rate analysis by role and game phase
- Persuasion technique detection and classification
- Deception rate measurement over game progression
- Statistical significance testing (Chi-square, Mann-Whitney U)
- Belief state tracking and accuracy evaluation
- Voting pattern analysis

### 🎨 **Visualization**

- Policy progression plots
- Persuasion technique heatmaps and spider plots
- Game state evaluation comparison charts
- ELO-based performance analysis
- Model comparison dashboards

### 🏷️ **Annotation Tools**

- Web-based annotation interface for persuasion techniques
- Support for Cialdini's principles and game-specific tactics
- Automated LLM-based annotation pipeline
- Inter-annotator agreement evaluation

## Installation

### Prerequisites

- Python 3.12 or higher
- Git
- (Optional) CUDA-enabled GPU for local LLM inference

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/ItsNiklas/secret-hitler-player.git
   cd secret-hitler-player
   ```

2. **Install dependencies**

   ```bash
   pip install -e .
   ```

3. **Configure the game**

   Create a YAML config file:

   ```yaml
   # config.yaml
   game:
     players: 5
     player_types: ["LLM", "CPU", "CPU", "CPU", "CPU"]

   llm:
     default:
       api_key: "your-key"
       base_url: "http://localhost:8080/v1/"
   ```

   See [config.example.yaml](config.example.yaml) for more options.

4. **Verify installation**

   ```bash
   cd simulator
   python HitlerGame.py --help
   ```

## Quick Start

### Run a Single Game

```bash
cd simulator
python HitlerGame.py --config ../config.yaml
```

This will simulate one game with the default configuration (LLM players vs rule-based players).

### Analyze Results

```bash
cd eval
python gamestats.py runsF1-Q3
python plot_comparison.py runsF1-Q3 runsF1-Llama33-70B
```

### Annotate Persuasion Techniques

```bash
cd annotation
python persuasion_annotation.py ../eval/runsF1-Q3/game_001.json
```

## Usage

### Running Simulations

#### Basic Simulation

```python
cd simulator
python HitlerGame.py
```

### Analyzing Results

#### Game Statistics

```bash
cd eval
python gamestats.py runsF1-Q3
```

Output includes:

- Overall win rates (Liberal vs Fascist)
- Win condition breakdown (policy, Hitler execution, etc.)
- Per-role performance statistics
- Statistical significance tests

#### Model Comparison

```bash
python plot_comparison.py runsF1-Model1 runsF1-Model2 runsF1-Model3
```

Generates comparative visualizations:

- Win rate comparison charts
- Policy progression plots
- Game state evaluation accuracy

#### Persuasion Analysis

```bash
python parse_annotation.py runsF1-Q3
python heatmap.py  # Generate persuasion heatmap
python heatmap-spider.py  # Generate spider plot
```

#### Reasoning Pattern Analysis

```bash
python reasoning.py runsF1-Q3
```

Analyzes reasoning categories:

- A: Game state observation
- B: Probability-based reasoning
- C: Player statement analysis
- D: Intuition/random guessing

#### Voting Behavior

```bash
python vote_analyzer.py runsF1-Q3
```

Examines:

- Voting patterns by game phase
- Role-based voting tendencies
- Confidence changes over time

### Persuasion Annotation

#### Automated Annotation

```bash
cd annotation
python persuasion_annotation.py <game_file.json> -f <output_folder>
```

Uses an LLM to automatically identify persuasion techniques based on:

- Cialdini's principles (reciprocity, commitment, social proof, etc.)
- Secret Hitler-specific tactics
- Jailbreak and manipulation techniques

#### Manual Annotation (Web Interface)

```bash
cd annotation/web
python -m http.server 8000
# Navigate to http://localhost:8000
```

Features:

- Context-aware message viewing
- Multi-label technique selection
- Progress tracking and saving
- Export to JSON

#### Evaluate Annotations

```bash
python evaluate_annotations.py Gemma312B llama-ann
```

Computes:

- Inter-annotator agreement (Cohen's Kappa)
- Confusion matrices
- Per-technique precision/recall

### Crawling Human Games

Collect real human gameplay data from secrethitler.io:

```bash
cd crawl
python scrape.py  # Collect game IDs
python dump.py    # Download game replays
python analyze.py # Analyze human behavior
```

Collected data is used for:

- Baseline comparison
- Validation of LLM behavior
- Training annotation models

## Player Types

### LLM Player (`llm_player.py`)

Autonomous LLM agent with configurable features:

- **Perception**: Receives game state, role information, chat history
- **Reasoning**: Optional CoT, memory retrieval, strategy consultation
- **Action**: Voting, nominations, policy selection, execution decisions
- **Communication**: Role-specific message generation

### Rule Player (`rule_player.py`)

Deterministic baseline using heuristics:

- Liberals: Trust confirmed liberals, vote cautiously
- Fascists: Coordinate to advance fascist policies
- Hitler: Blend in while supporting fascist agenda

### Random Player (`random_player.py`)

Uniformly random action selection for control experiments.

### Human Player (`human_player.py`)

Interactive CLI for human participation in mixed games.

## Evaluation Metrics

### Performance Metrics

- **Win Rate**: Overall and by role (Liberal/Fascist/Hitler)
- **Policy Success Rate**: Ability to enact desired policies
- **Game Length**: Average rounds to completion
- **Execution Accuracy**: Correct identification of Hitler

### Behavioral Metrics

- **Persuasion Technique Usage**: Frequency and diversity of techniques
- **Deception Rate**: Consistency of role-aligned vs deceptive statements
- **Strategic Reasoning**: Quality of decision justifications
- **Belief Accuracy**: Correctness of role inferences

### Statistical Tests

- **Chi-square**: Independence tests for categorical data
- **Mann-Whitney U**: Non-parametric comparison of distributions
- **Cramér's V**: Effect size for contingency tables
- **Cohen's Kappa**: Inter-annotator agreement

## Configuration

### Game Parameters

Modify in `simulator/HitlerGameState.py`:

```python
NUM_PLAYERS = 5  # Total players (5-10 supported)
NUM_FASCIST_POLICIES = 6  # Policies needed for fascist win
NUM_LIBERAL_POLICIES = 5  # Policies needed for liberal win
ENABLE_VETO_POWER = True  # After 5 fascist policies
```

### LLM Configuration

Adjust prompts and parameters in `simulator/players/llm_player.py`:

```python
ENABLE_COT = True  # Chain-of-thought reasoning
ENABLE_MEMORY = True  # Episodic memory
ENABLE_ROLE_MESSAGES = True  # Role-specific messaging
ENABLE_STRATEGY_DOCS = True  # Strategy guide integration
TEMPERATURE = 0.7  # Sampling temperature
MAX_TOKENS = 500  # Response length limit
```

### Visualization Settings

Customize plots in `eval/plot_config.py`:

```python
UNIBLAU = "#003A5D"  # Primary color
ROLE_COLORS = {...}  # Role-specific colors
DPI = 300  # Export resolution
USE_LATEX = True  # LaTeX rendering
```

## Acknowledgments

This work builds upon several excellent open-source projects:

- **Secret Hitler Simulator Base**: [Mycleung/Secret-Hitler](https://github.com/Mycleung/Secret-Hitler)
- **LLM Integration Foundation**: [edis0n-zhang/secret-hitler-player](https://github.com/edis0n-zhang/secret-hitler-player/)
- **Strategy Documentation**: [Secret Hitler Strategy Guide](https://secrethitler.tartanllama.xyz/)

Special thanks to:

- The Secret Hitler game designers for creating this compelling social deduction game
- The open-source LLM community for making powerful models accessible
