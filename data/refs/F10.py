# F10: средняя вероятность и доля голосов «красного»; основа — ISLR, гл. 8, упр. 5.
# Усреднение вероятностей и большинство жёстких голосов могут дать разные классы.
probs = [0.05, 0.10, 0.15, 0.20, 0.30, 0.55, 0.55, 0.60, 0.60, 0.62, 0.65]

if probs is None:
    raise SystemExit("Впишите свои probs")
mean_p = sum(probs) / len(probs)
vote_share = sum(p > 0.5 for p in probs) / len(probs)
print(
    f"# усреднение → {'красный' if mean_p > 0.5 else 'зелёный'}; голосование → {'красный' if vote_share > 0.5 else 'зелёный'}"
)
print(f"REFERENCE: {mean_p:.4f}; {vote_share:.4f}")
