import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression

class ConfidenceDetector:
    def __init__(self, model_path):
        self.model = LogisticRegression.load_model(model_path)

    def predict(self, audio_features):
        return self.model.predict_proba(audio_features)[0, 1]

def extract_audio_features(audio_file):
    """Extracts audio features from an audio file."""
    y, sr = librosa.load(audio_file)

    # Compute Mel-frequency cepstral coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y, sr=sr)

    # Compute pitch
    pitch = librosa.core.pitch_extractor(y, sr=sr)

    return mfccs, pitch

def main():
    # Here Load the confidence detector model
    confidence_detector = ConfidenceDetector('confidence_detector.model')

    # Here Upload the audio file
    audio_file = input('Enter the path to the audio file: ')

    # Here Extract audio features
    mfccs, pitch = extract_audio_features(audio_file)

    # Here Predict the confidence score
    confidence_score = confidence_detector.predict([mfccs, pitch])

    # Here Print the results
    print('Predicted confidence score:', confidence_score)

if __name__ == '__main__':
    main()
    