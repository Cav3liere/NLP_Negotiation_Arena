import json
import re
import os
from collections import defaultdict
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scenario_config import SCENARIO_BOUNDS

# Rilevamento a parole chiave delle tattiche retoriche (approssimativo)
RHETORICAL_PATTERNS = {
    "scarcity_urgency": re.compile(r"(?i)\b(other buyer|waiting|limited|last chance|someone else|hurry|cash ready)\b"),
    "emotional_guilt":  re.compile(r"(?i)\b(family|grandfather|losing money|struggling|honest|favor|unfair|hard times)\b"),
    "walk_away_threat": re.compile(r"(?i)\b(walk away|leave it|final offer|take it or leave|elsewhere|competitor|no deal)\b"),
    "skepticism_callout": re.compile(r"(?i)\b(tactic|manipulat|trick|gimmick|nice try|won't work|calling.*out)\b"),
}

_sentiment_analyzer = SentimentIntensityAnalyzer()
_LOG_LINE_RE = re.compile(r'^(Buyer|Seller)\s*\(\$(-?\d+)\):\s*(.*)$')


def load_results(base_dir: str = "results") -> Dict[str, List[Dict[str, Any]]]:
    all_results = defaultdict(list)
    dirs_to_scan = [base_dir, os.path.join(base_dir, "temp_experiment")]
    
    for current_dir in dirs_to_scan:
        if not os.path.exists(current_dir):
            continue
        for fname in sorted(os.listdir(current_dir)):
            if not fname.endswith(".json") or fname == "all_results.json":
                continue
            try:
                with open(os.path.join(current_dir, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                scenario_name = data["scenario"]
                if "temp_experiment" in current_dir:
                    scenario_name = f"{scenario_name}_T={data.get('temperature', 'N/A')}"
                    
                all_results[scenario_name].append(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️ File '{fname}' saltato: {e}")
    return dict(all_results)


def _extract_messages(transcript: List[str]) -> List[str]:
    messages = []
    for line in transcript:
        m = _LOG_LINE_RE.match(line)
        if m and m.group(3).strip():
            messages.append(m.group(3).strip())
    return messages


def analyze_transcript_linguistics(transcript: List[str]) -> Dict[str, int]:
    counts = {key: 0 for key in RHETORICAL_PATTERNS}
    for line in transcript:
        for tactic, pattern in RHETORICAL_PATTERNS.items():
            if pattern.search(line):
                counts[tactic] += 1
    return counts


def analyze_language_complexity(transcript: List[str]) -> Dict[str, Optional[float]]:
    """Language complexity: leggibilità (Flesch) e diversità lessicale (TTR)."""
    messages = _extract_messages(transcript)
    if not messages:
        return {"flesch": None, "lexical_diversity": None}
    full_text = " ".join(messages)
    words = re.findall(r"[a-zA-Z']+", full_text.lower())
    flesch = textstat.flesch_reading_ease(full_text) if words else None
    ttr = len(set(words)) / len(words) if words else None
    return {"flesch": flesch, "lexical_diversity": ttr}


def analyze_emotional_tone(transcript: List[str]) -> Optional[float]:
    """Emotional tone: sentiment medio (VADER compound, -1..+1) sul dialogo."""
    scores = []
    for line in transcript:
        m = _LOG_LINE_RE.match(line)
        if m and m.group(3).strip():
            scores.append(_sentiment_analyzer.polarity_scores(m.group(3))["compound"])
    return float(np.mean(scores)) if scores else None


def analyze_logical_coherence(transcript: List[str]) -> Optional[float]:
    """Logical coherence (proxy): similarità TF-IDF tra messaggi consecutivi."""
    messages = _extract_messages(transcript)
    if len(messages) < 2:
        return None
    try:
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(messages)
    except ValueError:
        return None
    sims = [cosine_similarity(tfidf[i], tfidf[i + 1])[0][0] for i in range(len(messages) - 1)]
    return float(np.mean(sims)) if sims else None


def compute_metrics(results: List[Dict[str, Any]], scenario_name: str) -> Dict[str, Any]:
    n = len(results)
    agreed = [r for r in results if r.get("agreed", False)]
    agreement_rate = len(agreed) / n if n > 0 else 0.0
    avg_rounds = float(np.mean([r["rounds"] for r in results])) if results else 0.0
    avg_price = float(np.mean([r["final_price"] for r in agreed])) if agreed else None

    bounds = SCENARIO_BOUNDS.get(scenario_name.split("_")[0], {"seller": (150, 200), "buyer": (60, 155)})
    s_min, s_max = bounds["seller"]
    b_min, b_max = bounds["buyer"]
    zopa_exists = s_min <= b_max

    seller_u, buyer_u, violations = [], [], 0
    for r in agreed:
        p = r["final_price"]
        seller_u.append((p - s_min) / (s_max - s_min) if s_max != s_min else 0.0)
        buyer_u.append((b_max - p) / (b_max - b_min) if b_max != b_min else 0.0)
        if p < s_min or p > b_max:
            violations += 1

    tactic_totals = defaultdict(int)
    flesch_vals, ttr_vals, sentiment_vals, coherence_vals = [], [], [], []
    for r in results:
        t = r.get("transcript", [])
        for k, v in analyze_transcript_linguistics(t).items():
            tactic_totals[k] += v
        complexity = analyze_language_complexity(t)
        if complexity["flesch"] is not None:
            flesch_vals.append(complexity["flesch"])
        if complexity["lexical_diversity"] is not None:
            ttr_vals.append(complexity["lexical_diversity"])
        tone = analyze_emotional_tone(t)
        if tone is not None:
            sentiment_vals.append(tone)
        coherence = analyze_logical_coherence(t)
        if coherence is not None:
            coherence_vals.append(coherence)

    return {
        "n_runs": n,
        "agreement_rate": round(agreement_rate, 3),
        "avg_rounds": round(avg_rounds, 2),
        "avg_final_price": round(avg_price, 2) if avg_price is not None else None,
        "zopa_exists": zopa_exists,
        "seller_utility": round(float(np.mean(seller_u)), 3) if seller_u else None,
        "buyer_utility": round(float(np.mean(buyer_u)), 3) if buyer_u else None,
        "constraint_violations": violations,
        "linguistic_tactics": dict(tactic_totals),
        "flesch_reading_ease": round(float(np.mean(flesch_vals)), 1) if flesch_vals else None,
        "lexical_diversity": round(float(np.mean(ttr_vals)), 3) if ttr_vals else None,
        "avg_sentiment": round(float(np.mean(sentiment_vals)), 3) if sentiment_vals else None,
        "turn_coherence": round(float(np.mean(coherence_vals)), 3) if coherence_vals else None,
    }


def plot_price_evolution(results: Dict[str, List[Dict[str, Any]]], save_dir: str = "plots") -> None:
    os.makedirs(save_dir, exist_ok=True)
    n_scenarios = len(results)
    if n_scenarios == 0:
        return
    fig, axes = plt.subplots(1, n_scenarios, figsize=(4 * n_scenarios, 5), sharey=True)
    if n_scenarios == 1:
        axes = [axes]
    fig.suptitle("Price Trajectory across Rounds")

    for ax, (scenario, runs) in zip(axes, results.items()):
        for run in runs:
            buyer_p, seller_p = [], []
            for line in run.get("transcript", []):
                m = re.match(r'(Buyer|Seller)\s*\(\$(\d+)\):', line)
                if m:
                    (buyer_p if m.group(1) == "Buyer" else seller_p).append(int(m.group(2)))
            n_min = min(len(buyer_p), len(seller_p))
            if n_min > 0:
                ax.plot(range(1, n_min + 1), buyer_p[:n_min], color="#2980b9", alpha=0.35)
                ax.plot(range(1, n_min + 1), seller_p[:n_min], color="#c0392b", alpha=0.35)
        ax.set_title(scenario.capitalize())
        ax.set_xlabel("Round")
        ax.set_ylim(40, 220)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/price_convergence.png", dpi=300)
    plt.close()


def plot_linguistic_features(metrics: Dict[str, Dict[str, Any]], save_dir: str = "plots") -> None:
    os.makedirs(save_dir, exist_ok=True)
    scenarios = list(metrics.keys())
    tactics = list(RHETORICAL_PATTERNS.keys())
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, tactic in enumerate(tactics):
        values = [metrics[s]["linguistic_tactics"].get(tactic, 0) for s in scenarios]
        ax.bar(x + (idx - 1.5) * 0.2, values, 0.2, label=tactic.replace('_', ' '))
    ax.set_title("Rhetorical Tactics by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenarios], rotation=15, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/linguistic_features.png", dpi=300)
    plt.close()


def plot_language_metrics(metrics: Dict[str, Dict[str, Any]], save_dir: str = "plots") -> None:
    os.makedirs(save_dir, exist_ok=True)
    scenarios = list(metrics.keys())
    x = np.arange(len(scenarios))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(x, [metrics[s]["flesch_reading_ease"] or 0 for s in scenarios], color="#27ae60")
    axes[0].set_title("Language Complexity (Flesch)")

    sent = [metrics[s]["avg_sentiment"] or 0 for s in scenarios]
    axes[1].bar(x, sent, color=["#c0392b" if v < 0 else "#2980b9" for v in sent])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylim(-1, 1)
    axes[1].set_title("Emotional Tone (sentiment)")

    axes[2].bar(x, [metrics[s]["turn_coherence"] or 0 for s in scenarios], color="#8e44ad")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Logical Coherence (TF-IDF)")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in scenarios], rotation=15, ha='right')
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/language_metrics.png", dpi=300)
    plt.close()


def plot_success_rates(metrics: Dict[str, Dict[str, Any]], save_dir: str = "plots") -> None:
    os.makedirs(save_dir, exist_ok=True)
    scenarios = list(metrics.keys())
    
    successes = [int(metrics[scenario]["n_runs"] * metrics[scenario]["agreement_rate"]) for scenario in scenarios]
    failures = [metrics[scenario]["n_runs"] - succ for scenario, succ in zip(scenarios, successes)]
    
    x = np.arange(len(scenarios))
    width = 0.5
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x, failures, width, label='Failed (No Agreement)', color='#e74c3c')
    ax.bar(x, successes, width, bottom=failures, label='Success (Agreement)', color='#2ecc71')
    
    ax.set_title("Agreement Success Rate by Scenario", fontsize=14, pad=15)
    ax.set_ylabel("Number of Negotiations", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenarios], rotation=15, ha='right')
    ax.legend(loc="upper right")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/success_rates.png", dpi=300)
    plt.close()


def print_summary_table(metrics: Dict[str, Dict[str, Any]]) -> None:
    print("\n| Scenario | Agree % | Rounds | Price | Seller U | Buyer U | Viol. | ZOPA | Flesch | TTR | Sentim. | Coher. |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s, m in metrics.items():
        price = f"${m['avg_final_price']}" if m['avg_final_price'] is not None else "N/A"
        print(f"| {s:<20} | {m['agreement_rate']:>6.0%} | {m['avg_rounds']:>6.2f} | {price:>6} | "
              f"{m['seller_utility'] or 0:.2f} | {m['buyer_utility'] or 0:.2f} | {m['constraint_violations']} | "
              f"{'yes' if m['zopa_exists'] else 'NO'} | {m['flesch_reading_ease'] or 0:.1f} | "
              f"{m['lexical_diversity'] or 0:.3f} | {m['avg_sentiment'] or 0:.2f} | {m['turn_coherence'] or 0:.3f} |")


if __name__ == "__main__":
    results = load_results()
    if results:
        def custom_order(scenario_name):
            name = scenario_name.lower()
            if "cooperative" in name: return 1
            if "deceptive" in name: return 2
            if "competitive" in name:
                if "1.0" in name: return 3
                if "0.4" in name: return 5
                return 4
            return 6
            
        results = {k: results[k] for k in sorted(results.keys(), key=custom_order)}
        
        metrics = {s: compute_metrics(runs, s) for s, runs in results.items()}
        print_summary_table(metrics)
        plot_price_evolution(results)
        plot_linguistic_features(metrics)
        plot_language_metrics(metrics)
        plot_success_rates(metrics)