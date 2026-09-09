import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class NegotiationResult:
    agreed: bool = False
    final_price: Optional[int] = None
    rounds: int = 0
    transcript: List[str] = field(default_factory=list)


class NegotiationArena:
    """Gestisce il ciclo di turni e le condizioni di chiusura tra due agenti."""

    def __init__(self, seller: Any, buyer: Any, max_turns: int = 10, verbose: bool = True) -> None:
        self.seller = seller
        self.buyer = buyer
        self.max_turns = max_turns
        self.verbose = verbose
        self.history = ""
        self.result = NegotiationResult()
        self.last_offers: Dict[str, Optional[int]] = {"Buyer": None, "Seller": None}

    def _log(self, text: str) -> None:
        if self.verbose:
            print(text)

    def _clean_message(self, text: str) -> str:
        text = re.sub(r'(?i)\bprice\b\s*[:\-]\s*\d+', '', text)
        text = re.sub(r'(?i)\bmessage\b\s*[:\-]\s*', '', text)
        text = re.sub(r'[{}"]', '', text)
        return text.replace(', ,', ',').strip().lstrip(',').strip()

    def _is_agreement(self, message: str) -> bool:
        return "<AGREEMENT>" in message

    def _log_turn(self, role: str, message: str, price: int) -> None:
        icon = "🛒" if role == "Buyer" else "💰"
        self.result.transcript.append(f"{role} (${price}): {message}")
        self._log(f"{icon} {role} (${price}): {message}")

    def _close_deal(self, final_price: int) -> None:
        self.result.agreed = True
        self.result.final_price = final_price
        self._log(f"\n🤝 DEAL CLOSED at ${final_price} after {self.result.rounds} round(s).")

    def _offers_crossed(self) -> bool:
        b, s = self.last_offers["Buyer"], self.last_offers["Seller"]
        return b is not None and s is not None and b >= s

    def _handle_turn(self, agent: Any, role: str, opponent: str) -> Optional[NegotiationResult]:
        resp = agent.get_response(self.history)
        msg = self._clean_message(resp["message"])
        price = resp["price"]

        if msg and msg != "[Agent failed to respond]":
            self.history += f"\n{role}: {msg}"
            self._log_turn(role, msg, price)
            self.last_offers[role] = price

        if self._is_agreement(resp["message"]):
            deal_price = self.last_offers[opponent] if self.last_offers[opponent] is not None else price
            self._close_deal(deal_price)
            return self.result

        if self._offers_crossed():
            self._close_deal(self.last_offers["Buyer"])
            return self.result

        return None

    def run(self) -> NegotiationResult:
        self._log("=== START NEGOTIATION ===\n")
        self.history = "[SYSTEM: Begin the negotiation based on your personality.]"

        for turn in range(self.max_turns):
            self.result.rounds = turn + 1
            self._log(f"--- Round {self.result.rounds} ---")

            if self._handle_turn(self.buyer, "Buyer", "Seller") is not None:
                break
            if self._handle_turn(self.seller, "Seller", "Buyer") is not None:
                break
        else:
            self._log("\n❌ NO AGREEMENT REACHED.")

        self._log(f"\nAgreed: {self.result.agreed} | Price: {self.result.final_price} | Rounds: {self.result.rounds}")
        return self.result