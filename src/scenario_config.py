# Vincoli (min, max): per il seller (prezzo minimo, prezzo di partenza),
# per il buyer (offerta di partenza, budget massimo).
# NB: in "competitive" s_min (170) > b_max (130) -> nessuna zona di accordo
# possibile (ZOPA): un accordo qui implica che un agente ha violato il vincolo.
SCENARIO_BOUNDS = {
    "cooperative": {"seller": (150, 200), "buyer": (60, 155)},
    "deceptive":   {"seller": (150, 200), "buyer": (60, 155)},
    "competitive": {"seller": (170, 200), "buyer": (60, 130)},
}

CONTEXTS = [
    "The item is a used smartphone in good condition.",
    "The item is a refurbished phone with minor scratches.",
    "The item is a vintage phone, still working perfectly.",
    "The item is a second-hand phone, recently serviced.",
]