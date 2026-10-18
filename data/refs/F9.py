# F9: ручной k-means по образцу ISLR (James et al., гл. 12 / гл. 10, упр. 3).
# Итоговые центры выводятся в порядке соответствующих начальных центров.
points = [(2, 5), (1, 4), (2, 3), (6, 2), (7, 3), (5, 1), (4, 4)]
centers = [(6, 2), (7, 3)]

if points is None or centers is None:
    raise SystemExit("Впишите свои points и centers")
for it in range(100):
    clusters = [[], []]
    # При равных расстояниях выбирается первый центр.
    for p in points:
        d = [(p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in centers]
        clusters[0 if d[0] <= d[1] else 1].append(p)
    assert all(clusters), "Один из кластеров пуст — выберите другие начальные центры"
    new = [
        (sum(p[0] for p in cl) / len(cl), sum(p[1] for p in cl) / len(cl))
        for cl in clusters
    ]
    if new == centers:
        break
    centers = new
print(f"# итераций до сходимости: {it}; кластеры: {clusters}")
print(
    f"REFERENCE: {centers[0][0]:.4f}; {centers[0][1]:.4f}; {centers[1][0]:.4f}; {centers[1][1]:.4f}"
)
