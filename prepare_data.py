"""
Summary:  Prepare data & util functions. 
Author:   Qiuqiang Kong
Created:  2017.12.12
Modified: - 
"""
import numpy as np
import argparse
from scipy import signal
from midiutil.MidiFile import MIDIFile
import matplotlib.pyplot as plt
import soundfile
import librosa
import csv
import time
import h5py
import pickle
import cPickle
import os
from sklearn import preprocessing

import config as cfg


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na

### Audio & feature related. 
def read_audio(path, target_fs=None):
    """Read 1 dimension audio sequence from given path. 
    
    Args:
      path: string, path of audio. 
      target_fs: int, resampling rate. 
      
    Returns:
      audio: 1 dimension audio sequence. 
      fs: sampling rate of audio. 
    """
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs
    
def write_audio(path, audio, sample_rate):
    """Write audio sequence to .wav file. 
    
    Args:
      path: string, path to write out .wav file. 
      data: ndarray, audio sequence to write out. 
      sample_rate: int, sample rate to write out. 
      
    Returns: 
      None. 
    """
    soundfile.write(file=path, data=audio, samplerate=sample_rate)
    
def spectrogram(audio):
    """Calculate magnitude spectrogram of an audio sequence. 
    
    Args: 
      audio: 1darray, audio sequence. 
      
    Returns:
      x: ndarray, spectrogram (n_time, n_freq)
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    
    ham_win = np.hamming(n_window)
    [f, t, x] = signal.spectral.spectrogram(
                    audio, 
                    window=ham_win,
                    nperseg=n_window, 
                    noverlap=n_overlap, 
                    detrend=False, 
                    return_onesided=True, 
                    mode='magnitude') 
    x = x.T
    x = x.astype(np.float32)
    return x
    
def logmel(audio):
    """Calculate log Mel spectrogram of an audio sequence. 
    
    Args: 
      audio: 1darray, audio sequence. 
      
    Returns:
      x: ndarray, log Mel spectrogram (n_time, n_freq)
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    
    ham_win = np.hamming(n_window)
    [f, t, x] = signal.spectral.spectrogram(
                    audio, 
                    window=ham_win,
                    nperseg=n_window, 
                    noverlap=n_overlap, 
                    detrend=False, 
                    return_onesided=True, 
                    mode='magnitude') 
    x = x.T
                    
    if globals().get('melW') is None:
        global melW
        melW = librosa.filters.mel(sr=fs, 
                                n_fft=n_window, 
                                n_mels=229, 
                                fmin=0, 
                                fmax=fs / 2.)
    x = np.dot(x, melW.T)
    x = np.log(x + 1e-8)
    x = x.astype(np.float32)
    return x

def calculate_features(args): 
    """Calculate and write out features & ground truth notes of all songs in MUS 
    directory of all pianos. 
    """
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    feat_type = args.feat_type
    fs = cfg.sample_rate
    tr_pianos = cfg.tr_pianos
    te_pianos = cfg.te_pianos
    pitch_bgn = cfg.pitch_bgn
    pitch_fin = cfg.pitch_fin
    
    out_dir = os.path.join(workspace, "features", feat_type)
    create_folder(out_dir)
    
    # Calculate features for all 9 pianos. 
    cnt = 0
    for piano in tr_pianos + te_pianos:
        audio_dir = os.path.join(dataset_dir, piano, "MUS")
        wav_names = [na for na in os.listdir(audio_dir) if na.endswith('.wav')]
        
        for wav_na in wav_names:
            # Read audio. 
            bare_na = os.path.splitext(wav_na)[0]
            wav_path = os.path.join(audio_dir, wav_na)
            (audio, _) = read_audio(wav_path, target_fs=fs)
            
            # Calculate feature. 
            if feat_type == "spectrogram":
                x = spectrogram(audio)
            elif feat_type == "logmel":
                x = logmel(audio)
            else:
                raise Exception("Error!")
            
            # Read piano roll from txt file. 
            (n_time, n_freq) = x.shape
            txt_path = os.path.join(audio_dir, "%s.txt" % bare_na)
            roll = txt_to_midi_roll(txt_path, max_fr_len=n_time)    # (n_time, 128)
            y = roll[:, pitch_bgn : pitch_fin]      # (n_time, 88)
            
            # Write out data. 
            data = [x, y]
            out_path = os.path.join(out_dir, "%s.p" % bare_na)
            print(cnt, out_path, x.shape, y.shape)
            cPickle.dump(data, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
            cnt += 1
        
### Pack features. 
def is_in_pianos(na, list_of_piano):
    """E.g., na="MAPS_MUS-alb_esp2_SptkBGCl.wav", list_of_piano=['SptkBGCl', ...]
    then return True. 
    """
    for piano in list_of_piano:
        if piano in na:
            return True
    return False
            
def pack_features(args):
    """Pack already calculated features and write out to a big file, for 
    speeding up later loading. 
    """
    workspace = args.workspace
    feat_type = args.feat_type
    tr_pianos = cfg.tr_pianos
    te_pianos = cfg.te_pianos
    
    fe_dir = os.path.join(workspace, "features", feat_type)
    fe_names = os.listdir(fe_dir)
    
    # Load all single feature files and append to list. 
    tr_x_list, tr_y_list, tr_na_list = [], [], []
    te_x_list, te_y_list, te_na_list = [], [], []
    t1 = time.time()
    cnt = 0
    for fe_na in fe_names:
        print(cnt)
        bare_na = os.path.splitext(fe_na)[0]
        fe_path = os.path.join(fe_dir, fe_na)
        [x, y] = cPickle.load(open(fe_path, 'rb'))
        
        if is_in_pianos(fe_na, tr_pianos):
            tr_x_list.append(x)
            tr_y_list.append(y)
            tr_na_list.append("%s.wav" % bare_na)
        elif is_in_pianos(fe_na, te_pianos):
            te_x_list.append(x)
            te_y_list.append(y)
            te_na_list.append("%s.wav" % bare_na)
        else:
            raise Exception("File not in tr_pianos or te_pianos!")
        cnt += 1
    
    # Write out the big file. 
    out_dir = os.path.join(workspace, "packed_features", feat_type)
    create_folder(out_dir)
    tr_packed_feat_path = os.path.join(out_dir, "train.p")
    te_packed_feat_path = os.path.join(out_dir, "test.p")
    
    cPickle.dump([tr_x_list, tr_y_list, tr_na_list], open(tr_packed_feat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump([te_x_list, te_y_list, te_na_list], open(te_packed_feat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    print("Packing time: %s s" % (time.time() - t1,))
    
### Scaler related. 
def compute_scaler(args):
    """Compute and write out scaler from already packed feature file. Using 
    scaler in training neural network can speed up training. 
    """
    workspace = args.workspace
    feat_type = args.feat_type
    
    # Load packed features. 
    t1 = time.time()
    packed_feat_path = os.path.join(workspace, "packed_features", feat_type, "train.p")
    [x_list, _, _] = cPickle.load(open(packed_feat_path, 'rb'))
    
    # Compute scaler. 
    x_all = np.concatenate(x_list)
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(x_all)
    print(scaler.mean_)
    print(scaler.scale_)
    
    # Save out scaler. 
    out_path = os.path.join(workspace, "scalers", feat_type, "scaler.p")
    create_folder(os.path.dirname(out_path))
    pickle.dump(scaler, open(out_path, 'wb'))
    print("Compute scaler finished! %s s" % (time.time() - t1,))
    
def scale_on_x_list(x_list, scaler): 
    """Scale list of ndarray. 
    """
    return [scaler.transform(e) for e in x_list]
    
### Data pre-processing. 
def data_to_3d(x_list, y_list, n_concat, n_hop):
    """Convert data to 3d tensor. 
    
    Args: 
      x_list: list of ndarray, e.g., [(N1, n_freq), (N2, n_freq), ...]
      y_list: list of ndarray, e.g., [(N1, 88), (N2, 88), ...]
      n_concat: int, number of frames to concatenate. 
      n_hop: int, hop frames. 
      
    Returns:
      x_all: (n_samples, n_concat, n_freq)
      y_all: (n_samples, n_out)
    """
    x_all, y_all = [], []
    n_half = (n_concat - 1) / 2
    for e in x_list:
        x3d = mat_2d_to_3d(e, n_concat, n_hop)
        x_all.append(x3d)
        
    for e in y_list:
        y3d = mat_2d_to_3d(e, n_concat, n_hop)
        y_all.append(y3d)
        
    x_all = np.concatenate(x_all, axis=0)   # (n_samples, n_concat, n_freq)
    y_all = np.concatenate(y_all, axis=0)   # (n_samples, n_concat, n_out)
    y_all = y_all[:, n_half, :]     # (n_samples, n_out)
    return x_all, y_all
    
def mat_2d_to_3d(x, agg_num, hop):
    """Convert data to 3d tensor. 
    
    Args: 
      x: 2darray, e.g., (N, n_in)
      agg_num: int, number of frames to concatenate. 
      hop: int, hop frames. 
      
    Returns:
      x3d: 3darray, e.g., (n_samples, agg_num, n_in)
    """
    # pad to at least one block
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num-len_x, n_in))))
        
    # agg 2d to 3d
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1+agg_num <= len_x):
        x3d.append(x[i1:i1+agg_num])
        i1 += hop
    x3d = np.array(x3d)
    return x3d
    
### I/O. 
def txt_to_midi_roll(txt_path, max_fr_len):
    """Read txt to piano roll. 
    
    Args: 
      txt_path: string, path of note info txt. 
      max_fr_len: int, should be the same as the number of frames of calculated 
          feature. 
          
    Returns:
      midi_roll: (n_time, 108)
    """
    step_sec = cfg.step_sec
    
    with open(txt_path, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)

    midi_roll = np.zeros((max_fr_len, 128))
    for i1 in xrange(1, len(lis)):
        # Read a note info from a line. 
        [onset_time, offset_time, midi_pitch] = lis[i1]
        onset_time = float(onset_time)
        offset_time = float(offset_time)
        midi_pitch = int(midi_pitch)
        
        # Write a note info to midi roll. 
        onset_fr = int(np.floor(onset_time / step_sec))
        offset_fr = int(np.ceil(offset_time / step_sec)) + 1
        midi_roll[onset_fr : offset_fr, midi_pitch] = 1
        
    return midi_roll

def prob_to_midi_roll(x, thres):
    """Threshold input probability to binary, then convert piano roll (n_time, 88) 
    to midi roll (n_time, 108). 
    
    Args:
      x: (n_time, n_pitch)    
    """
    pitch_bgn = cfg.pitch_bgn
    x_bin = np.zeros_like(x)
    x_bin[np.where(x >= thres)] = 1
    n_time = x.shape[0]
    out = np.zeros((n_time, 128))
    out[:, pitch_bgn : pitch_bgn + 88] = x_bin
    return out    

def write_midi_roll_to_midi(x, out_path):
    """Write out midi_roll to midi file. 
    
    Args: 
      x: (n_time, n_pitch), midi roll. 
      out_path: string, path to write out the midi. 
    """
    step_sec = cfg.step_sec
    
    def _get_bgn_fin_pairs(ary):
        pairs = []
        bgn_fr, fin_fr = -1, -1
        for i2 in xrange(1, len(ary)):
            if ary[i2-1] == 0 and ary[i2] == 0:
                pass
            elif ary[i2-1] == 0 and ary[i2] == 1:
                bgn_fr = i2
            elif ary[i2-1] == 1 and ary[i2] == 0:
                fin_fr = i2
                if fin_fr > bgn_fr:
                    pairs.append((bgn_fr, fin_fr))
            elif ary[i2-1] == 1 and ary[i2] == 1:
                pass
            else:
                raise Exception("Input must be binary matrix!")
            
        return pairs
    
    # Get (pitch, bgn_frame, fin_frame) triple. 
    triples = []
    (n_time, n_pitch) = x.shape
    for i1 in xrange(n_pitch):
        ary = x[:, i1]
        pairs_per_pitch = _get_bgn_fin_pairs(ary)
        if pairs_per_pitch:
            triples_per_pitch = [(i1,) + pair for pair in pairs_per_pitch]
            triples += triples_per_pitch
    
    # Sort by begin frame. 
    triples = sorted(triples, key=lambda x: x[1])
    
    # Write out midi. 
    MyMIDI = MIDIFile(1)    # Create the MIDIFile Object with 1 track
    track = 0   
    time = 0
    tempo = 120
    beat_per_sec = 60. / float(tempo)
    MyMIDI.addTrackName(track, time, "Sample Track")  # Add track name 
    MyMIDI.addTempo(track, time, tempo)   # Add track tempo
    
    for triple in triples:
        (midi_pitch, bgn_fr, fin_fr) = triple
        bgn_beat = bgn_fr * step_sec / float(beat_per_sec)
        fin_beat = fin_fr * step_sec / float(beat_per_sec)
        dur_beat = fin_beat - bgn_beat
        MyMIDI.addNote(track=0,     # The track to which the note is added.
                    channel=0,   # the MIDI channel to assign to the note. [Integer, 0-15]
                    pitch=midi_pitch,    # the MIDI pitch number [Integer, 0-127].
                    time=bgn_beat,      # the time (in beats) at which the note sounds [Float].
                    duration=dur_beat,  # the duration of the note (in beats) [Float].
                    volume=100)  # the volume (velocity) of the note. [Integer, 0-127].
    out_file = open(out_path, 'wb')
    MyMIDI.writeFile(out_file)
    out_file.close()
    
### Evaluation. 
def tp_fn_fp_tn(p_y_pred, y_gt, thres, average):
    """
    Args:
      p_y_pred: shape = (n_samples,) or (n_samples, n_classes)
      y_gt: shape = (n_samples,) or (n_samples, n_classes)
      thres: float between 0 and 1. 
      average: None (element wise) | 'micro' (calculate metrics globally) 
        | 'macro' (calculate metrics for each label then average). 
      
    Returns:
      tp, fn, fp, tn or list of tp, fn, fp, tn. 
    """
    if p_y_pred.ndim == 1:
        y_pred = np.zeros_like(p_y_pred)
        y_pred[np.where(p_y_pred > thres)] = 1.
        tp = np.sum(y_pred + y_gt > 1.5)
        fn = np.sum(y_gt - y_pred > 0.5)
        fp = np.sum(y_pred - y_gt > 0.5)
        tn = np.sum(y_pred + y_gt < 0.5)
        return tp, fn, fp, tn
    elif p_y_pred.ndim == 2:
        tps, fns, fps, tns = [], [], [], []
        n_classes = p_y_pred.shape[1]
        for j1 in xrange(n_classes):
            (tp, fn, fp, tn) = tp_fn_fp_tn(p_y_pred[:, j1], y_gt[:, j1], thres, None)
            tps.append(tp)
            fns.append(fn)
            fps.append(fp)
            tns.append(tn)
        if average is None:
            return tps, fns, fps, tns
        elif average == 'micro' or average == 'macro':
            return np.sum(tps), np.sum(fns), np.sum(fps), np.sum(tns)
        else: 
            raise Exception("Incorrect average arg!")
    else:
        raise Exception("Incorrect dimension!")
        
def prec_recall_fvalue(p_y_pred, y_gt, thres, average):
    """
    Args:
      p_y_pred: shape = (n_samples,) or (n_samples, n_classes)
      y_gt: shape = (n_samples,) or (n_samples, n_classes)
      thres: float between 0 and 1. 
      average: None (element wise) | 'micro' (calculate metrics globally) 
        | 'macro' (calculate metrics for each label then average). 
      
    Returns:
      prec, recall, fvalue | list or prec, recall, fvalue. 
    """
    eps = 1e-10
    if p_y_pred.ndim == 1:
        (tp, fn, fp, tn) = tp_fn_fp_tn(p_y_pred, y_gt, thres, average=None)
        prec = tp / max(float(tp + fp), eps)
        recall = tp / max(float(tp + fn), eps)
        fvalue = 2 * (prec * recall) / max(float(prec + recall), eps)
        return prec, recall, fvalue
    elif p_y_pred.ndim == 2:
        n_classes = p_y_pred.shape[1]
        if average is None or average == 'macro':
            precs, recalls, fvalues = [], [], []
            for j1 in xrange(n_classes):
                (prec, recall, fvalue) = prec_recall_fvalue(p_y_pred[:, j1], y_gt[:, j1], thres, average=None)
                precs.append(prec)
                recalls.append(recall)
                fvalues.append(fvalue)
            if average is None:
                return precs, recalls, fvalues
            elif average == 'macro':
                return np.mean(precs), np.mean(recalls), np.mean(fvalues)
        elif average == 'micro':
            (prec, recall, fvalue) = prec_recall_fvalue(p_y_pred.flatten(), y_gt.flatten(), thres, average=None)
            return prec, recall, fvalue
        else:
            raise Exception("Incorrect average arg!")
    else:
        raise Exception("Incorrect dimension!")
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')

    parser_a = subparsers.add_parser('calculate_features')
    parser_a.add_argument('--dataset_dir', type=str)
    parser_a.add_argument('--workspace', type=str)
    parser_a.add_argument('--feat_type', type=str, choices=['logmel'])
    
    parser_pack_features = subparsers.add_parser('pack_features')
    parser_pack_features.add_argument('--workspace', type=str)
    parser_pack_features.add_argument('--feat_type', type=str, choices=['logmel'])
    
    parser_compute_scaler = subparsers.add_parser('compute_scaler')
    parser_compute_scaler.add_argument('--workspace', type=str)
    parser_compute_scaler.add_argument('--feat_type', type=str, choices=['logmel'])
    
    args = parser.parse_args()
    if args.mode == 'calculate_features':
        calculate_features(args)
    elif args.mode == 'pack_features':
        pack_features(args)
    elif args.mode == 'compute_scaler':
        compute_scaler(args)
    else:
        raise Exception("Incorrect argument!")