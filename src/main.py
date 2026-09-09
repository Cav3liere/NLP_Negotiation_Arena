import os
from agent import NegotiationAgent
from arena import NegotiationArena

MODEL_ID = "gemini-3.6-flash"

SELLER_PROMPT = """You are an experienced, slightly arrogant street-market dealer selling a used telephone.
Your starting price is 200. Your absolute minimum is 150 — you will never go below that.
Rules:
- If the buyer offers 150 or more, accept immediately and include the token <AGREEMENT> in your message.
- Make decreasing concessions: first drop up to 20, then up to 10, then at most 5.
- Never raise your price after dropping it.
- Keep your message to 2-3 sentences max.
Output ONLY a valid JSON object: {"price": <int>, "message": "<text>"}"""

BUYER_PROMPT = """You are a savvy, fast-talking telephone flipper looking for a bargain.
Your opening offer is 60. Your absolute maximum is 155 — you will not go above that.
Rules:
- If the seller offers 155 or less, accept immediately and include the token <AGREEMENT> in your message.
- Make decreasing concessions: first raise up to 15, then up to 10, then at most 5.
- Never lower your offer after raising it.
- Keep your message to 2-3 sentences max.
Output ONLY a valid JSON object: {"price": <int>, "message": "<text>"}"""

if __name__ == "__main__":
    seller = NegotiationAgent(MODEL_ID, "Seller", SELLER_PROMPT)
    buyer = NegotiationAgent(MODEL_ID, "Buyer", BUYER_PROMPT)

    result = NegotiationArena(seller, buyer, max_turns=12).run()

    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(f"Agreement: {result.agreed}\nFinal price: {result.final_price}\nRounds: {result.rounds}\n")
        f.write("\n--- TRANSCRIPT ---\n" + "\n".join(result.transcript))

    print("\n📄 Transcript saved to transcript.txt")