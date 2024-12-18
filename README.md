# Zadanie 11. Faktoryzacja Doolitle’a
Elementy macierzy dolnej lewostronnej $L$ i górnej prawostronnej $U$ takich że $A = L \times U$ są obliczane w nastepujący sposób:
##
$k$ - numer iteracji
$${u_{kj}} = {a_{kj}} - \sum_{m=1}^{k-1} {l_{km}}{u_{mj}}, j = k, k+1, ..., n$$
$${l_{ik}} = \frac{{a_{ik}} - \sum_{m=1}^{k-1} {l_{im}}{u_{mk}}}{u_{kk}}, i = k+1, ..., n$$

Macierz A wygenerować losowo.
Sprawdzić poprawność rozwiązania licząc normę Frobeniusa $∥L \times U − A∥_F$ (nie mierzyć czasu sprawdzenia).
Norma Frobeniusa: https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm.
Dobrać wymiar zadania tak, żeby obliczenia sekwencyjne trwały ok. 2 minut. Przeprowadzić testy dla różnej liczby rdzeni. Policzyć przyspieszenia zgodnie z definicją z wykładu
