import urllib.request
import os
import ssl

usecase_puml = """
@startuml
scale 3
left to right direction
skinparam packageStyle rectangle
skinparam handwritten false
skinparam monochrome false
skinparam shadowing true

actor "Felhasználó / Kutató" as User
actor "Mikrofon / Bemenet" as Mic

rectangle "Hangszerfelismerő Rendszer" {
  usecase "Adathalmaz letöltése és tisztítása" as UC1
  usecase "Spektrogramok legenerálása (Jellemzőkinyerés)" as UC2
  usecase "Modell tanítása és validálása" as UC3
  usecase "Valós idejű hangfelvétel" as UC4
  usecase "Hangszer predikciója" as UC5
}

User --> UC1
User --> UC2
User --> UC3
User --> UC4

Mic --> UC4
UC4 .> UC5 : <<include>>
@enduml
"""

architecture_puml = """
@startuml
scale 3
skinparam componentStyle uml2
skinparam handwritten false
top to bottom direction

package "Környezet és Bemenet" {
  [Mikrofon] as Mic
  [WAV / MP3 fájlok] as Files
}

package "Adatelőkészítés (Jellemzőkinyerés)" {
  [Keretezés és Ablakozás] as Framing
  [STFT / Log-Mel Konverzió] as STFT
}

package "Mélytanulási Modell (Keras)" {
  [2D CNN Jellemzőkinyerő] as CNN
  [Sűrű (Dense) Osztályozó] as Dense
}

package "Kimenet" {
  [Terminál UI] as UI
  [Konfúziós Mátrix] as Matrix
}

Mic --> Framing : "Valós idejű audio stream"
Files --> Framing : "Adathalmaz (Tanítás)"
Framing --> STFT : "2.27 ms - 20 ms keretek"
STFT --> CNN : "2D Spektrogram mátrix"
CNN --> Dense : "Laposított jellemzővektor"
Dense --> UI : "Valós idejű predikció"
Dense --> Matrix : "Modell validálása"

@enduml
"""

def generate_uml(uml_text, out_file):
    print(f"Generating {out_file}...")
    url = "https://kroki.io/plantuml/png"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    headers = {
        'Content-Type': 'text/plain',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    req = urllib.request.Request(url, data=uml_text.encode('utf-8'), headers=headers)
    with urllib.request.urlopen(req, context=ctx) as response:
        with open(out_file, 'wb') as f:
            f.write(response.read())
    print(f"Saved {out_file}.")

if __name__ == "__main__":
    out_dir = "/Volumes/Kingston XS1000 Media/project/thesis"
    generate_uml(usecase_puml, os.path.join(out_dir, "fig_usecase.png"))
    generate_uml(architecture_puml, os.path.join(out_dir, "fig_architecture.png"))
