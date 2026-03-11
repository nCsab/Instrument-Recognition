# COMPARISON — InstrumentClassifier Benchmark

Teljes NSynth dataset: **4548 train / 975 val / 975 test** minta (csak akusztikus, 6 hangszercsalád)

## Összefoglaló

| Feature | Modell | Dim | Pontosság | Val. acc. |
|:--|:--|:--:|:--:|:--:|
| **MFCC** | RandomForest | 39 | **99.4%** | 98.4% |
| **FFT** | RandomForest | 11 | **91.0%** | 91.1% |

## Részletes eredmények

### MFCC + RandomForest — 99.4%

| Hangszer | Precision | Recall | F1-score | Minták |
|:--|:--:|:--:|:--:|:--:|
| guitar | 99% | 100% | 100% | 295 |
| keyboard | 100% | 97% | 98% | 66 |
| string | 99% | 100% | 99% | 168 |
| brass | 100% | 100% | 100% | 173 |
| reed | 99% | 99% | 99% | 144 |
| mallet | 100% | 100% | 100% | 129 |

### FFT + RandomForest — 91.0%

| Hangszer | Precision | Recall | F1-score | Minták |
|:--|:--:|:--:|:--:|:--:|
| guitar | 91% | 95% | 93% | 295 |
| keyboard | 97% | 44% | 60% | 66 |
| string | 85% | 96% | 91% | 168 |
| brass | 88% | 97% | 92% | 173 |
| reed | 96% | 90% | 92% | 144 |
| mallet | 96% | 94% | 95% | 129 |

## Tanulságok

1. **MFCC >> FFT** teljes dataseten (99.4% vs 91.0%)
2. Az FFT **keyboard felismerése nagyon gyenge** (recall: 44%) — a zongorahangot összekeveri más hangszerekkel
3. Az MFCC 39 dimenziós "ujjlenyomata" sokkal gazdagabb mint az FFT 11 száma
4. Több adat = sokkal jobb eredmény (30 mintával: 81%/72%, teljes: 99.4%/91.0%)

## Még elérhető kombinációk

A projekt jelenleg 2 feature extractort (MFCC, FFT) és 1 modellt (RandomForest) tartalmaz.

Következő lépések:
- [ ] Mel Filterbank feature extractor hozzáadása
- [ ] SVM modell hozzáadása
- [ ] MLP (Keras) modell hozzáadása
- [ ] Teljes 3×3 benchmark (9 kombináció)
