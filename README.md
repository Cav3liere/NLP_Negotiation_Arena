# 🤖 The Negotiation Arena: LLM Multi-Agent Simulation

Rigidity, Emergent Persuasion, and the Limits of Compromise in LLM Negotiation Agents. Final project for Natural Language Processing.

## 🎯 Project Overview
This project investigates whether modern Large Language Models actually adapt their negotiation styles to strategic incentives or just mechanically follow prompt scripts. 
Using a custom multi-agent simulation framework, two LLM agents (a buyer and a seller) negotiate under three distinct game-theoretic rules (Cooperative, Deceptive, and Competitive) across 100+ multi-turn runs.

## 🛠️ Tech Stack & Libraries
- **Core Framework:** Python, Google GenAI API (`google-genai`)
- **Data Manipulation & Analysis:** Pandas, NumPy
- **NLP & Linguistic Metrics:** VADER (`vaderSentiment`), TextStat (Flesch Reading Ease), Scikit-Learn (TF-IDF Cosine Similarity)
- **Visualization:** Matplotlib, Seaborn

## 📊 Key Findings & Experiments
- **Behavioral Adaptability:** In Cooperative and Deceptive scenarios (with a valid Zone of Agreement - ZOPA), agents reliably converge to fair splits and adjust their tone and tactics (e.g., higher emotional sentiment in cooperative talks vs. negative/manipulative tone in deceptive ones).
- **The Competitive Rigidity:** In the Competitive scenario (no ZOPA), agents overwhelmingly hold their stated price bounds rather than searching for alternative compromises.
- **Temperature Invariance:** Running temperature sensitivity experiments ($T = 0.4, 0.7, 1.0$) revealed near-total invariance in competitive outcomes, suggesting the model treats price limits as rigid, non-negotiable instructions rather than soft constraints.

## 📂 Repository Structure
- `agent.py`: Wrapper class for Gemini API integration, prompt management, and JSON output parsing.
- `arena.py`: Core simulation engine managing turn-taking, chat history, agreement tokens, and price-crossing checks.
- `runner.py`: Batch execution script for running multi-scenario simulations and temperature experiments.
- `analysis.py`: Quantitative and linguistic analysis pipeline (generates success rates, price trajectories, and sentiment metrics).
- `results/`: Directory containing raw JSON logs and negotiation transcripts.
