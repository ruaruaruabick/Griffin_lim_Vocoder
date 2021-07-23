import librosa
import numpy as np
import scipy.signal as signal
import copy
class griffin_lim:
    def __init__(self,args) -> None:
        self.sr = args['sr'] # Sample rate.
        self.n_fft =  args['n_fft'] # fft points (samples)
        self.frame_shift = args['frame_shift'] # seconds
        self.frame_length = args['frame_length'] # seconds
        self.hop_length = 256 # samples.
        self.win_length = 1024 # samples.
        self.n_mels = args['n_mels'] # Number of Mel banks to generate
        self.power = args['power'] # Exponent for amplifying the predicted magnitude
        self.n_iter = args['n_iter'] # Number of inversion iterations
        self.preemphasis = args['preemphasis'] # or None
        self.max_db = args['max_db']
        self.ref_db = args['ref_db']
        self.top_db = args['top_db']
    def get_spectrograms(self, fpath):
        '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
        Args:
        sound_file: A string. The full path of a sound file.

        Returns:
        mel: A 2d array of shape (T, n_mels) <- Transposed
        mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
        '''
            # Loading sound file
        y, sr = librosa.load(fpath, sr=self.sr)

        # Trimming
        y, _ = librosa.effects.trim(y, top_db=self.top_db)

        # Preemphasis
        y = np.append(y[0], y[1:] - self.preemphasis * y[:-1])

        # stft
        linear = librosa.stft(y=y,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length)

        # magnitude spectrogram
        mag = np.abs(linear)  # (1+n_fft//2, T)

        # mel spectrogram
        mel_basis = librosa.filters.mel(sr, self.n_fft, self.n_mels)  # (n_mels, 1+n_fft//2)
        mel = np.dot(mel_basis, mag)  # (n_mels, t)

        # to decibel
        mel = 20 * np.log10(np.maximum(1e-5, mel))
        mag = 20 * np.log10(np.maximum(1e-5, mag))

        # normalize
        mel = np.clip((mel - self.ref_db + self.max_db) / self.max_db, 1e-8, 1)
        mag = np.clip((mag - self.ref_db + self.max_db) / self.max_db, 1e-8, 1)

        # Transpose
        mel = mel.T.astype(np.float32)  # (T, n_mels)
        mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

        return mel, mag
    def melspectrogram2wav(self,mel):
        ''' Generate wave file from mel-spectrogram'''
        # transpose
        mel = mel.T

        # de-noramlize
        mel = (np.clip(mel, 0, 1) * self.max_db) - self.max_db + self.ref_db

        # to amplitude
        mel = np.power(10.0, mel * 0.05)
        m = self.__mel_to_linear_matrix()
        mag = np.dot(m, mel)

        # wav reconstruction
        wav = self.__griffin_lim(mag)

        # de-preemphasis
        wav = signal.lfilter([1], [1, -self.preemphasis], wav)

        # trim
        wav, _ = librosa.effects.trim(wav)

        return wav.astype(np.float32)       
    def spectrogram2wav(self,mag):
        ''' Generate wave file from spectrogram'''
        # transpose
        mag = mag.T

        # de-noramlize
        mag = (np.clip(mag, 0, 1) * self.max_db) - self.max_db + self.ref_db

        # to amplitude
        mag = np.power(10.0, mag * 0.05)

        # wav reconstruction
        wav = self.__griffin_lim(mag)

        # de-preemphasis
        wav = signal.lfilter([1], [1, -self.preemphasis], wav)

        # trim
        wav, _ = librosa.effects.trim(wav)
        return wav.astype(np.float32)
    def __mel_to_linear_matrix(self):
        m = librosa.filters.mel(self.sr, self.n_fft, self.n_mels)
        m_t = np.transpose(m)
        p = np.matmul(m, m_t)
        d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
        return np.matmul(m_t, np.diag(d))
    def __griffin_lim(self,spectrogram):
        '''Applies Griffin-Lim's raw.
        '''
        X_best = copy.deepcopy(spectrogram)
        for i in range(self.n_iter):
            X_t = self.__invert_spectrogram(X_best)
            est = librosa.stft(X_t, self.n_fft, self.hop_length, win_length=self.win_length)
            phase = est / np.maximum(1e-8, np.abs(est))
            X_best = spectrogram * phase
        X_t = self.__invert_spectrogram(X_best)
        y = np.real(X_t)
        return y
    def __invert_spectrogram(self,spectrogram):
        '''
        spectrogram: [f, t]
        '''
        return librosa.istft(spectrogram, self.hop_length, win_length=self.win_length, window="hann")