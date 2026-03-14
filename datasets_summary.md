# Project Datasets Summary

This document provides a concise overview of all datasets currently integrated into the project for instrument recognition and environmental sound classification.

## 1. ESC-50 (Environmental Sound Classification)
- **Content:** 2000 recordings, 5-second clips, 50 classes (animals, urban noise, etc.).
- **Strengths:** Diverse and balanced; the academic standard for environmental sound benchmarking.

## 2. IRMAS (Instrument Recognition in Musical Audio Signals)
- **Content:** 6705 training excerpts (3s each) and 2874 testing excerpts (5-20s). Includes 11 instruments.
- **Strengths:** Focused on predominant instruments in real music; high-quality expert annotations.

## 3. NSynth (Neural Synth)
- **Content:** ~290,000 musical notes (4s clips). Includes acoustic, electronic, and synthetic sources.
- **Strengths:** Massive scale and perfect for generative modeling and deep learning instrument classification.

## 4. Medley-solos-DB
- **Content:** >20,000 solo instrument clips (3s each), 8 instrument classes.
- **Strengths:** Clean solo recordings from professional stems; ideal for isolated instrument recognition.

## 5. TinySOL
- **Content:** 2913 isolated notes across 12+ instrument families.
- **Strengths:** Studio-quality isolated notes; excellent for timbre analysis and clean feature extraction.

## 6. Good-sounds
- **Content:** 16,310 recordings categorized by sound quality (good/bad/rich etc.).
- **Strengths:** Unique focus on performance quality and sound characteristics beyond just instrument identity.

---
*For more detailed information, please refer to the `dataset_info.md` file within each dataset's respective directory.*
