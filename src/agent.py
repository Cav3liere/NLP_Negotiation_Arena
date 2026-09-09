import os
import json
import re
import time
from typing import Dict, Any
from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_api_key = os.getenv("GEMINI_API_KEY")

_client = genai.Client(
    api_key=_api_key
)

_last_call_time = 0.0
_MIN_INTERVAL = 4.5  # Pausa base tra le chiamate

def _throttle() -> None:
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_call_time = time.time()

class NegotiationAgent:
    """Agente negoziatore basato su un LLM (Gemini, via Google AI Studio)."""

    def __init__(self, model_id: str, name: str, personality_prompt: str, temperature: float = 0.7) -> None:
        self.model_id = model_id
        self.name = name
        self.personality_prompt = personality_prompt
        self.temperature = temperature

    def _extract_price(self, raw_price: Any) -> int:
        if isinstance(raw_price, (int, float)):
            return int(raw_price)
        if isinstance(raw_price, str):
            match = re.search(r'\d+', raw_price)
            if match:
                return int(match.group(0))
        return 0

    def get_response(self, chat_history: str, max_retries: int = 10) -> Dict[str, Any]:
        system_instruction = (
            "You are a negotiation agent. You must respond strictly in a single, valid JSON object "
            "with exactly two keys: 'price' (integer only, without currency symbols) and 'message' (string)."
        )
        full_prompt = f"{self.personality_prompt}\n\nChat History:\n{chat_history}"

        for attempt in range(max_retries):
            try:
                _throttle()
                response = _client.models.generate_content(
                    model=self.model_id,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self.temperature,
                        max_output_tokens=1024,
                        response_mime_type="application/json",
                    ),
                )
                if not response.text:
                    raise ValueError("Risposta vuota dal modello")
                data = json.loads(response.text.strip())
                return {
                    "price": self._extract_price(data.get("price", 0)),
                    "message": str(data.get("message", "")).strip()
                }
            except Exception as e:
                error_msg = str(e)
                print(f"⚠️ [{self.name}] Tentativo {attempt + 1}/{max_retries} fallito: {error_msg}")

                # Rileva il blocco quote forzando l'ibernazione per ripristinare i limiti
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print(f"⏳ [{self.name}] Limite Google raggiunto. Pausa forzata di 65 secondi per ripristino...")
                    time.sleep(65)
                else:
                    time.sleep(10)

        return {"price": 0, "message": "[Agent failed to respond]"}