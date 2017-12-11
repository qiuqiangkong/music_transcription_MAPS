# dataset_dir = "/vol/vssp/AP_datasets/audio/MAPS"
# 
# workspace = "/vol/vssp/msos/qk/workspaces/music_transcription_MAPS"

sample_rate = 16000
n_window = 2048
n_step = 512
n_overlap = n_window - n_step
step_sec = float(n_step) / sample_rate

# 88 piano notes range from midi index [21, 109)
pitch_bgn = 21  
piano_notes = 88
pitch_fin = pitch_bgn + piano_notes

tr_pianos = ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGCl']
# tr_pianos = ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']
te_pianos = ['ENSTDkAm', 'ENSTDkCl']