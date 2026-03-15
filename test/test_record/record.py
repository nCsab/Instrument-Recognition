import serial
import wave
import sys

PORT = '/dev/cu.usbmodem1303' 
BAUD = 460800
SAMPLE_RATE = 16000
DURATION_SEC = 10
BYTES_TO_READ = SAMPLE_RATE * DURATION_SEC * 2

try:
    ser = serial.Serial(PORT, BAUD, timeout=10)
    print(f"CSATLAKOZVA: {PORT} ({BAUD} baud)")
    print(f"KÉSZENLÉT - Nyomd meg a kék gombot a Nucleo-n!")

    while True:
        header = ser.read(4)
        if header == b'REC!':
            print(">>> Felvétel elindult...")
            break

    raw_data = bytearray()
    while len(raw_data) < BYTES_TO_READ:
        chunk = ser.read(min(4096, BYTES_TO_READ - len(raw_data)))
        if not chunk:
            break
        raw_data.extend(chunk)
        progress = int(len(raw_data) / BYTES_TO_READ * 100)
        sys.stdout.write(f"\rFogadás: {progress}% [{'#' * (progress // 5)}{'.' * (20 - progress // 5)}]")
        sys.stdout.flush()

    print("\n" + "=" * 50)
    if len(raw_data) == BYTES_TO_READ:
        with wave.open("teszt.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(raw_data)
        print("KÉSZ! A felvétel mentve: teszt.wav")
    else:
        print(f"HIBA: Adatvesztés! ({len(raw_data)}/{BYTES_TO_READ} bájt)")
    print("=" * 50)

    ser.close()

except Exception as e:
    print(f"Hiba: {e}")
