# AML Fraud Detection — PaySim

Wykrywanie transakcji fraudowych na danych mobile money, łączące reguły AML
oparte na doświadczeniu z KYC/AML (XTB) z modelem XGBoost i wyjaśnialnością SHAP.

## Problem

Wykrywanie fraudu w płatnościach mobilnych na skrajnie niezbalansowanym zbiorze
danych — fraud stanowi zaledwie 0.13% wszystkich transakcji. Projekt pokazuje
pełną ścieżkę: od reguł biznesowych opartych na wiedzy AML, przez model ML,
po wyjaśnialność decyzji i prosty dashboard do przeglądu alertów.

## Dane

Dataset: [PaySim — Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
(Kaggle, autor: Edgar Lopez-Rojas). Symulacja ~6.36 mln transakcji mobile money.

Pobierz plik z Kaggle i umieść go jako `data/paysim.csv` (folder `data/` nie jest
wersjonowany w repo — zobacz `.gitignore`).

## Kluczowe insighty z EDA

1. **Skrajne niezbalansowanie klas** — fraud stanowi 0.13% wszystkich transakcji
   (8213 z 6 362 620). Standardowe metryki jak accuracy są bezużyteczne, liczy
   się PR-AUC i recall/precision.
2. **Fraud występuje wyłącznie w dwóch typach transakcji** — TRANSFER (4097
   przypadków) i CASH_OUT (4116 przypadków). CASH_IN, DEBIT i PAYMENT nie
   zawierają fraudu w ogóle.
3. **Silny sygnał: zerowe saldo odbiorcy mimo wpływu środków.** Wśród transakcji
   CASH_OUT/TRANSFER, gdzie saldo odbiorcy nie zmienia się mimo dodatniej kwoty,
   aż 70.5% to fraud (vs bazowe 0.13%) — klasyczny wzorzec konta-słupa
   (mule account).
4. **Odwrotny sygnał po stronie nadawcy** — gdy saldo nadawcy się nie zmienia
   mimo wysłania środków, fraud występuje w zaledwie 0.002% przypadków.
   Sprawcy niemal zawsze aktywnie opróżniają konto.
5. **Rozkład kwot różni się kształtem** — fraud silniej koncentruje się w
   niskich kwotach, z dodatkowym skupiskiem przy bardzo wysokich wartościach.
6. **Ograniczenie datasetu** — PaySim generuje 1–3 transakcje na konto
   (symulacja, nie prawdziwa historia klienta), więc klasyczne velocity checks
   (częstotliwość transakcji w czasie) nie mają tu zastosowania.

## Podejście

### 1. Warstwa reguł AML

Cztery reguły zbudowane na bazie wzorców znanych z praktyki AML/KYC:
zerowe saldo odbiorcy, wysoki błąd salda, structuring (kwoty blisko progów
raportowania), pełne opróżnienie konta nadawcy.

| Metryka | Wynik |
|---|---|
| Recall | 99.5% (8173/8213 fraudów wykrytych) |
| Precision | 0.49% (1.65 mln fałszywych alarmów) |

Same reguły wyłapują niemal cały fraud, ale generują ogromną liczbę fałszywych
alarmów — klasyczny problem "alert fatigue" w systemach opartych wyłącznie na
regułach statycznych. To bezpośrednio uzasadnia potrzebę warstwy ML.

### 2. Model XGBoost

Trening na cechach numerycznych i kategorycznych (typ transakcji, kwoty,
błędy sald, flagi reguł), z `scale_pos_weight` kompensującym niezbalansowanie
klas.

| Metryka | Wynik |
|---|---|
| Precision | 98.79% |
| ROC-AUC | 0.9993 |

### 3. Wyjaśnialność (SHAP)

Analiza SHAP pokazuje, że najważniejszymi cechami dla modelu są `errorBalanceOrig`
i `newbalanceOrig` — model samodzielnie odkrył sygnały zbliżone do tych
zakodowanych ręcznie w regułach AML, co sugeruje, że drzewa decyzyjne potrafią
automatycznie wychwycić wzorce, które w podejściu regułowym trzeba definiować
ręcznie. Reguły miały jednak wartość jako punkt wyjścia do zrozumienia problemu
i budowy cech.

### 4. Dashboard (Streamlit)

Interaktywny panel z:
- suwakiem progu ryzyka (pokazuje trade-off recall/precision na żywo),
- tabelą oflagowanych transakcji,
- wykresem SHAP waterfall wyjaśniającym decyzję modelu dla wybranej transakcji.

## Wyniki — reguły vs model

| Podejście | Recall | Precision |
|---|---|---|
| Reguły AML | 99.5% | 0.49% |
| XGBoost | ~98%* | 98.79% |

\* recall modelu do uzupełnienia z classification_report

## Jak uruchomić

```bash
pip install -r requirements.txt

# 1. Wytrenuj model (notebook) — zapisze fraud_model.pkl, X_test.csv, y_test.csv
jupyter notebook notebook.ipynb

# 2. Odpal dashboard
streamlit run app.py
```

## Ograniczenia

- PaySim nie zawiera historii transakcyjnej per konto (1–3 transakcje/konto),
  więc reguły velocity-based (częstotliwość transakcji w czasie) nie mają tu
  zastosowania.
- Dataset syntetyczny — wzorce fraudu mogą nie w pełni odzwierciedlać realne
  dane transakcyjne.
- Reguła structuring (kwoty blisko progów raportowania) nie znalazła
  potwierdzenia w danych — PaySim najwyraźniej nie symuluje tego zjawiska.

## Tech stack

Python, pandas, XGBoost, SHAP, scikit-learn, Streamlit
