# COMPARISON — InstrumentClassifier Benchmark

Teljes NSynth dataset: **4548 train / 975 val / 975 test** (csak akusztikus, 6 hangszercsalád)
5 random seed átlaga: [7, 21, 42, 77, 123]

## Összefoglaló — 4 Feature × 3 Modell

| Kombináció | Dim | Átlag | Szórás | Min | Max |
|:--|:--:|:--:|:--:|:--:|:--:|
| **MFCC + MLP** | 39 | **99.5%** | 0.002 | 99.0% | 99.6% |
| **MFCC + SVM** | 39 | **99.4%** | 0.002 | 99.0% | 99.7% |
| **MelBank-120 + MLP** | 120 | **99.3%** | 0.004 | 98.9% | 100.0% |
| MFCC + RandomForest | 39 | 99.2% | 0.002 | 98.9% | 99.5% |
| MelBank-120 + SVM | 120 | 99.1% | 0.002 | 98.8% | 99.5% |
| MelBank-120 + RandomForest | 120 | 98.8% | 0.004 | 98.2% | 99.5% |
| MelBank-40 + RandomForest | 40 | 96.8% | 0.004 | 96.3% | 97.4% |
| FFT + MLP | 31 | 95.9% | 0.004 | 95.3% | 96.4% |
| MelBank-40 + MLP | 40 | 95.7% | 0.020 | 92.1% | 97.4% |
| FFT + RandomForest | 31 | 95.0% | 0.006 | 94.1% | 96.0% |
| MelBank-40 + SVM | 40 | 89.0% | 0.007 | 88.4% | 90.3% |
| FFT + SVM | 31 | 88.2% | 0.011 | 86.7% | 90.0% |

## Feature típus összehasonlítás (legjobb modellel)

| Feature | Legjobb modell | Átlag | Szórás | Dim | MCU ciklus |
|:--|:--|:--:|:--:|:--:|:--:|
| MFCC | MLP | 99.5% | 0.002 | 39 | ~50K |
| MelBank-120 | MLP | 99.3% | 0.004 | 120 | ~30K |
| MelBank-40 | RandomForest | 96.8% | 0.004 | 40 | ~20K |
| FFT | MLP | 95.9% | 0.004 | 31 | ~15K |

## Seed-enkénti részletek

| Kombináció | Seed 7 | Seed 21 | Seed 42 | Seed 77 | Seed 123 |
|:--|:--:|:--:|:--:|:--:|:--:|
| MFCC+MLP | 99.5% | 99.0% | 99.6% | 99.6% | 99.6% |
| MFCC+SVM | 99.0% | 99.5% | 99.5% | 99.7% | 99.5% |
| MelBank-120+MLP | 99.0% | 98.9% | 99.5% | 100.0% | 98.9% |
| MFCC+RandomForest | 98.9% | 99.5% | 99.3% | 99.0% | 99.5% |
| MelBank-120+SVM | 99.5% | 98.8% | 99.2% | 99.0% | 98.9% |
| MelBank-120+RandomForest | 98.8% | 99.0% | 98.6% | 98.2% | 99.5% |
| MelBank-40+RandomForest | 97.4% | 96.4% | 97.0% | 96.3% | 96.9% |
| FFT+MLP | 95.3% | 95.8% | 96.3% | 96.4% | 95.5% |
| MelBank-40+MLP | 92.1% | 97.4% | 96.6% | 97.3% | 95.1% |
| FFT+RandomForest | 94.9% | 94.1% | 96.0% | 94.9% | 95.1% |
| MelBank-40+SVM | 88.4% | 90.3% | 88.8% | 88.4% | 89.3% |
| FFT+SVM | 87.4% | 87.8% | 88.8% | 90.0% | 86.7% |

## STM32 ajánlás

| Cél | Kombináció | Pontosság | MCU ciklus |
|:--|:--|:--:|:--:|
| **Legjobb MCU kompromisszum** | MelBank-120 + MLP | 99.3% | ~30K |
| Legjobb pontosság | MFCC + MLP | 99.5% | ~50K |
| Legolcsóbb | FFT + MLP | 95.9% | ~15K |

## Következtetések

- **MFCC+MLP** az abszolút legjobb (99.5%), de az MCU-n +20K extra ciklus a DCT-ért
- **MelBank-120+MLP** a legjobb MCU kompromisszum (99.3%, max 100.0% egy seed-del)
- Alacsony szórás (< 0.01) = robusztus modell
- MelBank-40+MLP szórása magas (0.020) — kevésbé stabil
- Az MLP tensorflow.keras implementáció, közvetlenül exportálható `.h5`-ként az STM32Cube.AI számára
