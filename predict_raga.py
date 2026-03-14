import numpy as np
import librosa
import joblib
from raga_database import RAGA_DATABASE

MODEL_FILE = "swaratma_raga_model.pkl"

model = joblib.load(MODEL_FILE)

SWARA_POSITIONS = [
    0,100,200,300,400,500,600,700,800,900,1000,1100
]

SWARAS = [
    "Sa","Ri1","Ri2","Ga2","Ga3","Ma1","Ma2",
    "Pa","Da1","Da2","Ni2","Ni3"
]


def extract_pitch(audio):

    y, sr = librosa.load(audio, sr=22050, mono=True)

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )

    pitch = f0[~np.isnan(f0)]

    return pitch


def detect_sa(pitch):

    hist, bins = np.histogram(pitch, bins=100)

    peak = np.argmax(hist)

    sa = (bins[peak] + bins[peak+1]) / 2

    median_pitch = np.median(pitch)

    while sa * 2 < median_pitch:
        sa *= 2

    while sa > median_pitch:
        sa /= 2

    return sa


def pitch_to_swara_index(p):

    if p <= 0:
        return 0

    cents = 1200 * np.log2(p)
    cents = cents % 1200

    distances = [abs(cents - s) for s in SWARA_POSITIONS]

    return np.argmin(distances)


def generate_features(pitch, sa):

    histogram = np.zeros(12)

    for p in pitch:

        if p <= 0:
            continue

        ratio = p / sa

        index = pitch_to_swara_index(ratio)

        histogram[index] += 1

    if histogram.sum() > 0:
        histogram = histogram / histogram.sum()

    return histogram


def predict_raga(audio):

    print("Loading audio:", audio)

    pitch = extract_pitch(audio)

    if len(pitch) == 0:
        return {
            "status": "error",
            "message": "No pitch detected"
        }

    sa = detect_sa(pitch)

    print("Detected Sa:", round(sa,2),"Hz")

    features = generate_features(pitch, sa)

    raga = model.predict([features])[0]

    return {
        "raga": raga,
        "arohanam": RAGA_DATABASE[raga]["arohanam"],
        "avarohanam": RAGA_DATABASE[raga]["avarohanam"]
    }

    probs = model.predict_proba([features])[0]

    print("\nConfidence:")

    for r, p in zip(model.classes_, probs):
        print(r, ":", round(p * 100, 2), "%")


    if raga in RAGA_DATABASE:
        print("\nArohanam:")
        print(RAGA_DATABASE[raga]["arohanam"])

        print("\nAvarohanam:")
        print(RAGA_DATABASE[raga]["avarohanam"])


if __name__ == "__main__":

    audio_file = input("Enter audio file path: ")
    predict_raga(audio_file)