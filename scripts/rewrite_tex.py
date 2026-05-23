import re

file_path = "/Volumes/Kingston XS1000 Media/project/thesis/chapters/05_tervezes.tex"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update overall intro regarding models
content = content.replace(
    "és a 2D konvolúciós neurális hálózatokat, végül",
    "és a 2D konvolúciós neurális hálózatokat, a YAMNet transzfer tanulási architektúrát, végül"
)

# 2. Update Az adathalmaz tervezése counts (Clean)
content = re.sub(
    r"Ez osztályonként 500~darab, többnyire monofón, stúdió- és szobaminőségű mintát eredményezett.",
    r"Ez hangszertípusonként nagyjából 500~darab, többnyire monofón, stúdió- és szobaminőségű mintát eredményezett. A teljes tiszta adathalmaz (beleértve a kiterjedt zaj osztályt is) így 4428 mintából áll.",
    content
)

# 3. Update Mikrofonos újrafelvétel (Explain train-only leakage prevention)
mic_old = r"osztályonként összefűzöm a minták 50\%-át \(ez az arány biztosítja a megfelelő egyensúlyt a tiszta stúdióminőségű és a mikrofonnal rögzített zajos minták között, megakadályozva a laptop saját torzításaira való túltanulást\)"
mic_new = r"kizárólag a \texttt{train} halmazba szánt mintákat fűzöm össze (ez a lépés kulcsfontosságú, hiszen így a mikrofonos felvételek fizikailag semmilyen formában nem tartalmazzák a validációs vagy teszt halmaz hangjait, azaz az adatszivárgás, más néven data leakage, elméletileg és gyakorlatilag is ki van zárva)"
content = content.replace(mic_old, mic_new)

sync_old = r"a \texttt{02\_mic\_\\allowbreak prep\_\\allowbreak concatenate.py} scripttel"
sync_new = r"a \texttt{02\_acquire\_mic\_data.py} scripttel"
content = content.replace(sync_old, sync_new)

sync_old2 = r"a laptopomon futó \texttt{03\_mic\_\\allowbreak record\_\\allowbreak audio.py} script rögzíti"
sync_new2 = r"a laptopomon futó azonos script (\texttt{02\_acquire\_mic\_data.py}) rögzíti"
content = content.replace(sync_old2, sync_new2)

sync_old3 = r"Végül a \texttt{04\_mic\_\\allowbreak slice\_\\allowbreak audio.py} script"
sync_new3 = r"Végül ugyanez a script a szeletelő fázisban"
content = content.replace(sync_old3, sync_new3)

# 4. Update Végső adathalmaz összeállítása
dataset_old = r"Egy konzisztens és pontos adathalmaz.*?(?=\\section\{Jellemzőkinyerés tervezése\})"
dataset_new = r"""Egy konzisztens és pontos adathalmaz nagyban segíti a neurális hálózat tanulási folyamatát. A megtervezett adatfolyam két fő könyvtárra támaszkodik: a \texttt{dataset\_clean} (amely a tiszta internetes mintákat tartalmazza, összesen 4428 darabot) és a \texttt{dataset\_mic} (amely a \texttt{train} halmaz esetében már tartalmazza az újrafelvett, mikrofonos mintákat is, így összesen 6492 darabot számol).

Fontos kiemelni, hogy az adatok szétosztása a tanító (\texttt{train}), validációs (\texttt{val}) és tesztelő (\texttt{test}) halmazokra már a forrásblokkok szintjén megtörtént. Ezt a \texttt{01\_prepare\_clean\_dataset.py} script végezte, teljesen szeparált csoportszámokat (group ID) osztva ki. Emiatt az adatszivárgás (data leakage) jelensége – azaz, hogy ugyanannak a hangfelvételnek a részletei a tanító és a teszthalmazban is megjelenjenek – maradéktalanul kiküszöbölésre került.

A \texttt{noise} (zaj) osztály esetében közvetlenül az ESC-50 környezeti zajadatbázisból válogattam be több mint 1400 mintát (például szél, eső, utcai zajok), mivel ezek a felvételek a természetükből adódóan már eleve tartalmaznak háttérzajokat, így itt újrafelvételre nem volt szükség.

Amikor a jellemzőkinyerő script lefut, az adataugmentációs lépéseket (reverb, zajkeverés, torzítás) ,,on-the-fly'', azaz dinamikusan végzi el a memóriában. Ennek eredményeképpen a feldolgozott (kinyert) jellemzők száma a tanítási fázishoz kibővül: a tiszta adathalmazban 5927-re, míg a mikrofonos (augmented) adathalmazban 8335-re. Ez a megközelítés rendkívül stabil alapokat biztosít a hálózat számára.

"""
content = re.sub(dataset_old, dataset_new, content, flags=re.DOTALL)

# 5. Jellemzőkinyerés (Add Raw)
feat_old = r"a rövid idejű Fourier-transzformációt \(STFT\), a Mel-frekvenciás kepsztrális együtthatókat \(MFCC\), valamint a Log-Mel spektrogramot."
feat_new = r"a rövid idejű Fourier-transzformációt (STFT), a Mel-frekvenciás kepsztrális együtthatókat (MFCC), a Log-Mel spektrogramot, valamint a nyers hanghullámot (raw waveform)."
content = content.replace(feat_old, feat_new)

raw_section = r"""\subsection{Nyers hanghullám (Raw waveform)}

A frekvenciatartománybeli jellemzők mellett közvetlenül kinyerjük és elmentjük a nyers, időtartománybeli audiojelet is. 1 másodperc esetén ez egy 16\,000 elemű, 1D-s tömb. Bár a hagyományos 2D CNN hálózatok számára ez az ábrázolás nem ideális, a mélyebb transzfer tanulási eljárások (például a YAMNet) pontosan ezt a nyers formátumot várják bemenetként.

\subsection{A megfelelő jellemzőkinyerési eljárás kiválasztása}"""
content = content.replace(r"\subsection{A megfelelő jellemzőkinyerési eljárás kiválasztása}", raw_section)

# 6. Adataugmentáció (on-the-fly)
aug_old = r"Az augmentáció eredményeként az alapadathalmaz mérete megduplázódik a 6 hangszerosztály esetén \(a \\texttt\{noise\} osztályt nem augmentálom\), így alakítva ki a végső adathalmazt: 6 \\times 750 \\times 2 \+ 1500 = 10\\,500 minta."
aug_new = r"Ezek az augmentációs lépések közvetlenül a memóriában, ,,on-the-fly'' kerülnek alkalmazásra a \texttt{04a\_extract\_features.py} szkript futásakor. Ez azt jelenti, hogy az adatbővítés nem foglal felesleges helyet a lemezen WAV fájlok formájában, hanem azonnal a kinyert jellemzőmátrixokká alakul. Ennek eredményeképpen a tanításra kész, mikrofonos jellemzőhalmaz (processed\_data\_mic) összesen 8335 mintára nőtt."
content = content.replace(aug_old, aug_new)

# 7. Hálózat architektúra (Add YAMNet)
cnn_title_old = r"\section{A 2D CNN hálózat architektúrájának tervezése}"
cnn_title_new = r"\section{Modell architektúrák tervezése}"
content = content.replace(cnn_title_old, cnn_title_new)

cnn_intro_old = r"A tervezett hangszerfelismerő rendszer magját egy mély konvolúciós neurális hálózat \(CNN\) alkotja"
cnn_intro_new = r"A hangszerfelismerő rendszer magját kétféle megközelítéssel is megterveztem: egy saját fejlesztésű, kifejezetten erre a feladatra optimalizált 2D konvolúciós hálózattal (mint alapeset vagy baseline), valamint egy lényegesen mélyebb, YAMNet alapú transzfer tanulásos (transfer learning) architektúrával. Először a saját 2D CNN"
content = content.replace(cnn_intro_old, cnn_intro_new)

yamnet_section = r"""
\subsection{YAMNet Transfer Learning architektúra}
\label{subsec:yamnet_architecture}

Bár a saját fejlesztésű 2D CNN hálózat stabil teljesítményt nyújt (78\%-os baseline pontosság), a valós környezet sokrétű zajai és a felvételi torzítások kihívásai miatt szükség volt egy még robusztusabb, jobban általánosító modellre. Ennek eléréséhez a Google által fejlesztett és a TensorFlow Hub-on publikált YAMNet modellt választottam ki. 

A YAMNet egy mély, MobileNetV1 alapú architektúra, amelyet a hatalmas AudioSet adatbázison tanítottak be, így kiváló akusztikai reprezentációs képességekkel rendelkezik. A YAMNet bemenete közvetlenül a normalizált nyers hanghullám ($16\,000$ elemű vektor).

A rendszeremben a YAMNet modellt nem a nulláról tanítom újra, hanem jellemzőkinyerőként (feature extractor) használom. A folyamat során az 1 másodperces audiojel végighalad a YAMNet befagyasztott konvolúciós rétegein, aminek az eredménye egy 1024 dimenziós sűrű reprezentáció (beágyazás, azaz \textit{embedding}). 

Ezt a mély beágyazást egy saját tervezésű, teljesen összekapcsolt (fully connected) osztályozó hálózatba vezetem be. Az általam tervezett osztályozó felépítése a következő:
\begin{enumerate}
  \item \textbf{Bemenet:} 1024 dimenziós YAMNet embedding
  \item \textbf{Rejtett réteg 1:} Sűrű réteg (Dense), 256 neuron, ReLU aktiváció
  \item \textbf{Normalizáció 1:} Batch Normalization és 40\% Dropout
  \item \textbf{Rejtett réteg 2:} Sűrű réteg (Dense), 128 neuron, ReLU aktiváció
  \item \textbf{Normalizáció 2:} Batch Normalization és 40\% Dropout
  \item \textbf{Kimeneti réteg:} Sűrű réteg (Dense), 7 neuron, Softmax aktiváció (hangszerosztályok valószínűségei)
\end{enumerate}

Ez az architektúra nemcsak, hogy drasztikus pontosságnövekedést (akár 90\% körüli F1-score) és sokkal stabilabb működést eredményezett a saját CNN baseline-hoz képest, de a tanítási időt is jelentősen lecsökkentette.

\subsection{Tanítási konfiguráció}
"""
content = content.replace(r"\subsection{Tanítási konfiguráció}", yamnet_section)

# 8. Valós idejű
rt_old = r"átadom a betöltött 2D CNN modellnek."
rt_new = r"A YAMNet esetében a teljes, 16 000 mintás nyers puffert átalakítjuk a TensorFlow formátumába, amiből a YAMNet elkészíti az 1024-es beágyazást, majd ezt adjuk át a betanított osztályozónak (Classifier)."
content = content.replace(rt_old, rt_new)

rt_old2 = r"és a 2D CNN osztályozó futtatásával"
rt_new2 = r"és az osztályozó futtatásával"
content = content.replace(rt_old2, rt_new2)

rt_old3 = r"A simított valószínűség-vektor alapján"
rt_new3 = r"A \texttt{05b\_realtime\_yamnet.py} szerint a $W = 6$ elemű (kb. 1,5 másodperces historikus ablak) FIFO sorban tárolt vektorok alapján"
content = content.replace(rt_old3, rt_new3)

rt_old4 = r"egy $H = 0,05$ \($5\%$\) értékű hiszterézis bónuszt"
rt_new4 = r"egy $H = 0,05$ ($5\%$) értékű hiszterézis bónuszt"
content = content.replace(rt_old4, rt_new4)

rt_old5 = r"\tau = 0,3$ \($30\%$\) minimális"
rt_new5 = r"\tau = 0,45$ ($45\%$, illetve zaj esetén $20\%$) minimális"
content = content.replace(rt_old5, rt_new5)

rt_old6 = r"konvolúciós hálózat predikcióján"
rt_new6 = r"neurális hálózat predikcióján"
content = content.replace(rt_old6, rt_new6)

pipeline_old = r"(\texttt{10\_train\_*} szkriptek)"
pipeline_new = r"(\texttt{06\_train\_*} szkriptek)"
content = content.replace(pipeline_old, pipeline_new)

pipeline_old2 = r"(\texttt{07\_realtime\_"
pipeline_new2 = r"(\texttt{05a\_realtime\_} vagy \texttt{05b\_realtime\_"
content = content.replace(pipeline_old2, pipeline_new2)


with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Rewritten successfully.")
