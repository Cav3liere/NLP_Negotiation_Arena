import json
import os
import random
import time
from typing import Dict, List, Any

from agent import NegotiationAgent
from arena import NegotiationArena
from scenario_config import SCENARIO_BOUNDS, CONTEXTS

MODEL_ID = "gemini-3.6-flash"

# {s_min}/{s_max}/{b_min}/{b_max} vengono presi da SCENARIO_BOUNDS
SCENARIO_TEMPLATES: Dict[str, Dict[str, str]] = {
    "competitive": {
        "seller": (
            "You are an aggressive, stubborn street dealer selling a phone.\n"
            "Starting price: {s_max}. Absolute minimum: {s_min}.\n"
            "Use pressure tactics, express frustration, and be theatrical.\n"
            "Accept and include <AGREEMENT> only if the buyer offers {s_min} or more.\n"
            "2-3 sentences max. Output ONLY: {{\"price\": <int>, \"message\": \"<text>\"}}"
        ),
        "buyer": (
            "You are an extremely stingy buyer who thinks the phone is overpriced.\n"
            "Opening offer: {b_min}. Absolute maximum: {b_max}.\n"
            "Use aggressive bargaining: question quality, threaten to walk away.\n"
            "Accept and include <AGREEMENT> only if the seller drops to {b_max} or less.\n"
            "2-3 sentences max. Output ONLY: {{\"price\": <int>, \"message\": \"<text>\"}}"
        ),
    },
    "cooperative": {
        "seller": (
            "You are a friendly seller who genuinely wants a fair deal.\n"
            "Starting price: {s_max}. Minimum acceptable: {s_min}.\n"
            "Make reasonable concessions and explain your reasoning.\n"
            "Accept and include <AGREEMENT> if the buyer offers {s_min} or more.\n"
            "2-3 sentences max. Output ONLY: {{\"price\": <int>, \"message\": \"<text>\"}}"
        ),
        "buyer": (
            "You are a polite, reasonable buyer looking for a fair price.\n"
            "Opening offer: {b_min}. Maximum budget: {b_max}.\n"
            "Raise your offer steadily and propose fair compromises.\n"
            "Accept and include <AGREEMENT> if the seller drops to {b_max} or less.\n"
            "2-3 sentences max. Output ONLY: {{\"price\": <int>, \"message\": \"<text>\"}}"
        ),
    },
    "deceptive": {
        "seller": (
            "You are a manipulative seller using psychological tactics.\n"
            "Starting price: {s_max}. Minimum acceptable: {s_min}.\n"
            "Use false scarcity, emotional appeals, and artificial reluctance.\n"
            "Accept and include <AGREEMENT> if the buyer offers {s_min} or more.\n"
            "2-3 sentences max. Output ONLY: {{\"price\": <int>, \"message\": \"<text>\"}}"
        ),
        "buyer": (
            "You are a sharp, skeptical buyer who calls out manipulative tactics.\n"
            "Opening offer: {b_min}. Maximum budget: {b_max}.\n"
            "Call out the seller's tricks; only raise your offer for genuine reductions.\n"
            "Accept and include <AGREEMENT> if the seller drops to {b_max} or less.\n"
            "2-3 sentences max. Output ONLY: {{\"price\": <int>, \"message\": \"<text>\"}}"
        ),
    },
}


def build_prompts(scenario_name: str) -> Dict[str, str]:
    bounds = SCENARIO_BOUNDS[scenario_name]
    s_min, s_max = bounds["seller"]
    b_min, b_max = bounds["buyer"]
    template = SCENARIO_TEMPLATES[scenario_name]
    return {
        "seller": template["seller"].format(s_min=s_min, s_max=s_max, b_min=b_min, b_max=b_max),
        "buyer": template["buyer"].format(s_min=s_min, s_max=s_max, b_min=b_min, b_max=b_max),
    }


def run_batch(
    scenario_name: str,
    n_runs: int = 20,
    max_turns: int = 8,
    temperature: float = 0.7,
    results_dir: str = "results",
    filename_suffix: str = "",
) -> List[Dict[str, Any]]:
    """Esegue un batch di negoziazioni per uno scenario.

    filename_suffix permette di salvare varianti (es. esperimento
    temperatura) senza sovrascrivere i run principali dello stesso scenario.
    Il campo 'scenario' nel JSON resta invariato (serve ad analysis.py per
    i vincoli ZOPA corretti); la temperatura usata viene comunque salvata
    nel campo 'temperature' per poterle distinguere in analisi.
    """
    prompts = build_prompts(scenario_name)
    results = []
    os.makedirs(results_dir, exist_ok=True)

    for i in range(n_runs):
        out_filepath = f"{results_dir}/{scenario_name}{filename_suffix}_run{i+1}.json"

        if os.path.exists(out_filepath):
            print(f"⏭️  {scenario_name}{filename_suffix} run {i+1}/{n_runs} già presente, salto.")
            with open(out_filepath, "r", encoding="utf-8") as f:
                results.append(json.load(f))
            continue

        print(f"=== {scenario_name.upper()}{filename_suffix} run {i+1}/{n_runs} (T={temperature}) ===")
        context = random.choice(CONTEXTS)

        seller = NegotiationAgent(MODEL_ID, "Seller", f"{prompts['seller']}\nContext: {context}", temperature=temperature)
        buyer = NegotiationAgent(MODEL_ID, "Buyer", f"{prompts['buyer']}\nContext: {context}", temperature=temperature)

        # verbose=False: niente muro di testo round-per-round nei batch,
        # solo il riepilogo qui sotto. Il transcript completo resta comunque
        # salvato nel JSON per l'analisi.
        result = NegotiationArena(seller, buyer, max_turns=max_turns, verbose=False).run()

        print(f"  → Agreed: {result.agreed} | Price: {result.final_price} | Rounds: {result.rounds}")

        run_data = {
            "run": i + 1,
            "scenario": scenario_name,
            "model_id": MODEL_ID,
            "temperature": temperature,
            "context": context,
            "agreed": result.agreed,
            "final_price": result.final_price,
            "rounds": result.rounds,
            "transcript": result.transcript
        }
        results.append(run_data)

        with open(out_filepath, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2, ensure_ascii=False)

        time.sleep(2)

    return results


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # --- Batch principale: 20 run per scenario, temperatura di default 0.7 ---
    all_results = {s: run_batch(s, n_runs=20, max_turns=8) for s in SCENARIO_BOUNDS}

    with open("results/all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n✅ Batch principale completato.")

    # --- Esperimento temperatura: solo su competitive, 0.4 (bassa) e 1.0 (alta) ---
    print("\n=== ESPERIMENTO TEMPERATURA (competitive) ===")
    run_batch("competitive", n_runs=20, max_turns=8, temperature=0.4,
               results_dir="results/temp_experiment", filename_suffix="_temp_low")
    run_batch("competitive", n_runs=20, max_turns=8, temperature=1.0,
               results_dir="results/temp_experiment", filename_suffix="_temp_high")

    print("\n✅ Tutti gli esperimenti sono stati completati.")