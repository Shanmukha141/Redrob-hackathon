# Redrob Hackathon Submission

## Overview
This repository contains our complete ranking pipeline. We built a resource-aware, hybrid retrieval system. Stage 1 applies strict rule-based Python filters (discarding title traps, checking YoE, and validating >48 months of core AI/ML skills) to eliminate honeypots and unqualified candidates instantly. Stage 2 ranks the viable pool using a locally hosted SentenceTransformer with custom text-weighting, applying multipliers for behavioral signals (e.g., notice period, response rate) and domain expertise.

## Setup & Environment
[cite_start]This code is fully compliant with the offline, no-network constraint[cite: 80]. The required `all-MiniLM-L6-v2` model has been pre-downloaded and is included in the `local_model/` directory. 

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt