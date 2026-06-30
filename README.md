# Automatikus hangszerfelismerés zenefelvételek alapján

Ez a repository egy BSc államvizsga-projekt forráskódját és LaTeX
dokumentációját tartalmazza. A rendszer rövid, 1 másodperces hangmintákat sorol
be hét előre definiált audioosztályba:

`guitar`, `piano`, `vocal`, `string`, `reed`, `brass`, `noise`.

A fő megvalósítás egy saját Log-Mel + 2D CNN pipeline. A valós idejű felismerés
a laptop mikrofonjáról érkező hangot dolgozza fel, Log-Mel spektrogrammá
alakítja, majd a betanított saját CNN modellel predikciót készít. A YAMNet-alapú
megoldás transzfer tanulási referencia, nem a fő valós idejű alkalmazás.

## Projektstruktúra

- `scripts/` - az adatfeldolgozó, feature extraction, realtime és tanító
  szkriptek.
- `scripts/utils/` - közös segédfüggvények augmentációhoz és feature
  extractionhöz.
- `scripts/06_train_model_colab/` - Google Colabban futtatott tanítószkriptek.
- `owndataset/` - helyi, nem publikált nyers és mikrofonos adatgyűjtés helye.
- `dataset_clean/` - tiszta, 1 másodperces mintákra bontott adathalmaz.
- `experiment_datasets/` - a négy kísérleti fázis adathalmazai.
- `processed_data/` - NumPy tömbökbe mentett feature-ök és címkék.
- `models/` - tanított modellek, checkpointok és riportok.
- `thesis/` - a dolgozat LaTeX forrása és a végleges PDF.

A nagy méretű adathalmazok, feature tömbök és modellek a `.gitignore` miatt nem
részei a nyilvános GitHub repositorynak.

## Pipeline futtatása

A szkripteket a projekt gyökérkönyvtárából érdemes futtatni.

```bash
python3 scripts/01_prepare_clean_dataset.py
python3 scripts/02_acquire_mic_data.py
python3 scripts/03a_merge_mic_to_train.py
python3 scripts/03b_check_dataset_counts.py
python3 scripts/04a_extract_features.py
python3 scripts/04b_check_extracted_counts.py
```

Röviden:

1. A clean adathalmaz 5 másodperces blokkokból 1 másodperces klipekre bontása.
2. Mikrofonos visszajátszás előkészítése, felvétele és szeletelése.
3. A négy kísérleti adathalmaz összeállítása.
4. Darabszámok ellenőrzése.
5. Log-Mel, STFT, MFCC és szükség esetén nyers audio tömbök kinyerése.
6. A feldolgozott NumPy tömbök és címkék ellenőrzése.

## Tanítás

A modellek tanítása Google Colab környezetben történt:

- `scripts/06_train_model_colab/06_train_custom_cnn.py`
- `scripts/06_train_model_colab/06_train_yamnet.py`

A saját CNN validációs futások alapján választ checkpointot, a test halmazt
csak a kiválasztott végső modellek ellenőrzésére használja. A YAMNet szkript
nyers audio bemenetből 1024 dimenziós embeddingeket készít, majd ezekre egy kis
saját osztályozófejet tanít.

## Valós idejű felismerés

Lokális futtatáshoz:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/05a_realtime_recognition.py
```

A realtime script az `exp_final` Log-Mel checkpointot tölti be, 1 másodperces
csúszó ablakot használ, 0,25 másodpercenként frissít, és 6 predikció átlagával
simítja a kijelzett eredményt.

## Dokumentáció

A dolgozat forrása a `thesis/` mappában található. Fordítás:

```bash
cd thesis
latexmk -pdf -interaction=nonstopmode main.tex
```

A végleges dolgozat PDF-je:

```text
thesis/main.pdf
```

## Megjegyzés az adatokhoz

A nyers internetes hangminták és a mikrofonos újrafelvételek oktatási-kutatási
demonstrációs célra használt helyi forrásanyagok. Ezek nagy méretük és jogi
státuszuk miatt nem kerülnek nyilvános terjesztésre. A repository a kódot,
a dokumentációt és a futtatási struktúrát dokumentálja.
