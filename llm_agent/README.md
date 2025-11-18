# MCDC-Onboarding

An interactive AI tutor that guides you through building Monte Carlo particle transport simulations using the [MCDC](https://github.com/CEMeNT-PSAAP/MCDC) Python package. Perfect for beginners learning nuclear engineering concepts!

## What It Does

MCDC-Tutor walks you through a 7-step workflow to create complete MCDC simulations:

1.  **Materials** – Define what your geometry is made of (water, uranium, steel, etc.)
2.  **Surfaces** – Create geometric boundaries (planes, spheres, cylinders)
3.  **Cells** – Build regions of space filled with materials
4.  **Source** – Specify where particles start and their energy
5.  **Tally** – Set up detectors to measure flux, reactions, and more
6.  **Settings** – Configure simulation parameters (particles, batches)
7.  **Run** – Generate a working Python script with `mcdc.run()`

The tutor uses Google's Gemini AI to answer questions, show relevant examples from the MCDC documentation, and help you write correct code.

## Quick Start

### 1. Prerequisites

-   Python 3.8 or higher
-   pip package manager
-   A Google account (for API key)

### 2. Get a Gemini API Key

1.  Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2.  Sign in with your Google account
3.  Click **"Create API key"**
4.  Choose **"Create API key in new project"** (or an existing project)
5.  Copy the API key (it starts with `AIza...`)

> **Never** commit your API key to version control. Use environment variables as shown below.

### 3. Install Dependencies

From the **project root directory** (`mcdc/`):

```bash
pip install -r llm_agent/requirements.txt
```

Key dependencies:

langchain – AI agent framework

langchain-google-genai – Gemini integration

langchain-chroma – Vector database for documentation

pandas – Nuclear data processing

### 4. Set Your API Key as environment variable


```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 5. Run the Tutor
From the project root directory (mcdc/):

```bash
python -m llm_agent/onboarding/tutor
```
The interactive session will begin. Follow the prompts to build your simulation!

## Q&A

To test out the Q&A, run

```bash
python llm_agent/qa.py or llm_agent/improved_qa.py
```
and ask any questions you may have.

Any feedback is very helpful.
