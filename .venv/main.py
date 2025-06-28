import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

# Tabela de notas para MIDI
NOTA_TO_MIDI = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3,
    'E': 4, 'F': 5, 'F#': 6, 'G': 7,
    'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

# Faixa vocal por voz principal
FAIXA_VOCAL = {
    "soprano": (60, 84),  # C4 – C6
    "contralto": (55, 76),  # G3 – E5
    "tenor": (48, 69)  # C3 – A4
}


def detectar_notas(audio_path):
    y, sr = librosa.load(audio_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    notas = []

    for i in range(pitches.shape[1]):
        index = pitches[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            nota_midi = librosa.hz_to_midi(pitch)
            notas.append(int(round(nota_midi)))

    return notas


def calcular_intervalos(voz_principal):
    # Tríade maior para simplificação
    tercas = 4  # terça maior
    quintas = 7  # quinta justa
    return tercas, quintas


def aplicar_pitch_shift(original_path, semitons, output_path):
    y, sr = librosa.load(original_path)
    y_shifted = librosa.effects.pitch_shift(y, n_steps=semitons, sr=sr)
    sf.write(output_path, y_shifted, sr)


def gerar_vozes(audio_path, voz_melodia, direcao='up'):
    tercas, quintas = calcular_intervalos(voz_melodia)

    # Direção de pitch
    if direcao == 'up':
        semitons1 = tercas
        semitons2 = quintas
    elif direcao == 'down':
        semitons1 = -tercas
        semitons2 = -quintas
    else:
        semitons1 = -tercas
        semitons2 = quintas

    aplicar_pitch_shift(audio_path, semitons1, "voz1.wav")
    aplicar_pitch_shift(audio_path, semitons2, "voz2.wav")

    # Mixando
    original = AudioSegment.from_file(audio_path)
    voz1 = AudioSegment.from_file("voz1.wav")
    voz2 = AudioSegment.from_file("voz2.wav")
    mix = original.overlay(voz1).overlay(voz2)
    mix.export("mixagem_final.wav", format="wav")

    print("Vozes geradas com sucesso!")


# Executar
gerar_vozes("voz_melodia.wav", voz_melodia="tenor", direcao="up")
