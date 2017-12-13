"""
Summary:  Configuration file. 
Author:   Qiuqiang Kong
Created:  2017.12.12
Modified: - 
"""
sample_rate = 16000
n_window = 2048
n_step = 512
n_overlap = n_window - n_step
step_sec = float(n_step) / sample_rate

# 88 piano notes range from midi index [21, 109)
pitch_bgn = 21  
piano_notes = 88
pitch_fin = pitch_bgn + piano_notes

tr_pianos = ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']
te_pianos = ['ENSTDkAm', 'ENSTDkCl']