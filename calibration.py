import numpy as np

def compute_ece(confidences, correct_flags, n_bins=10):
    conf = np.array(confidences, dtype=float)
    corr = np.array(correct_flags, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(conf)
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i+1])
        if mask.sum() == 0: continue
        ece += (mask.sum()/n) * abs(conf[mask].mean() - corr[mask].mean())
    return round(float(ece), 4)

def compute_owa(confidences, correct_flags):
    conf = np.array(confidences, dtype=float)
    wrong = ~np.array(correct_flags, dtype=bool)
    if wrong.sum() == 0: return 0.0
    return round(float(conf[wrong].mean()), 4)

def full_report(confidences, correct_flags, label=''):
    conf = np.array(confidences, dtype=float)
    corr = np.array(correct_flags, dtype=float)
    r = {'label':label, 'ECE':compute_ece(conf,corr),
         'OWA':compute_owa(conf,corr), 'Accuracy':round(corr.mean(),4), 'N':len(conf)}
    print(f'[{label}] ECE={r["ECE"]}  OWA={r["OWA"]}  Acc={r["Accuracy"]}')
    return r
