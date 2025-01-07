import numpy as np
import scipy.signal as signal

def asl_P56(x, fs, nbits):
    eps = np.finfo(float).eps
    x = x[:]  # make sure x is column vector
    if len(x.shape) < 2:
        x = x[:, np.newaxis]
    T = 0.03  # time constant of smoothing, in seconds
    H = 0.2  # hangover time in seconds
    M = 15.9  # margin in dB of the difference between threshold and ASL
    thres_no = nbits-1  # number of thresholds, for 16 bit, it's 15

    I = np.ceil(fs*H)  # hangover in samples
    g = np.exp(-1/(fs*T))  # smoothing factor in envelop detection
    c = 2**np.arange(-15, thres_no-15, dtype=float)

    # vector with thresholds from one quantizing level up to half the maximum
    # code, at a step of 2, in the case of 16bit samples, from 2^-15 to 0.5
    a = np.full(thres_no, -1)  # activity counter for each level threshold
    hang = np.full(thres_no, I)  # hangover counter for each level threshold

    sq = x.T@x  # long-term level square energy of x
    x_len = x.shape[0] # length of x

    # use a 2nd order IIR filter to detect the envelope q
    x_abs = np.abs( x)
    p = signal.lfilter([1-g, 0], [1, -g], np.squeeze(x_abs))
    q = signal.lfilter([1-g, 0], [1, -g], np.squeeze(p))  # q is the envelope, obtained from moving average of abs(x) (with slight "hangover").

    for k in range(x_len):
        for j in range(thres_no):
            if q[k] >= c[j]:
                a[j] = a[j]+1
                hang[j] = 0
            elif hang[j] < I:
                a[j] = a[j]+1
                hang[j] = hang[j]+1
            else:
                break
    actfact = 0
    asl_msq = 0
    c0 = 0
    if a[0] == -1:
        return asl_msq, actfact, c0
    else:
        a+=2
        AdB1=10*np.log10(sq/a[0]+eps)

    CdB1 = 20*np.log10(c[0]+eps)
    if (AdB1-CdB1 < M):
        return asl_msq, actfact, c0

    AdB = np.zeros(thres_no)
    CdB = np.zeros(thres_no)
    Delta = np.zeros(thres_no)
    AdB[0] = AdB1
    CdB[0] = CdB1
    Delta[0] = AdB1-CdB1

    for j in range(1, thres_no):
        AdB[j] = 10*np.log10(sq/(a[j]+eps)+eps)
        CdB[j] = 20*np.log10(c[j]+eps)

    for j in range(1, thres_no):
        if a[j] != 0:
            Delta[j]= AdB[j]- CdB[j]
            if Delta[j] <= M:
                # interpolate to find the actfact
                asl_ms_log, cl0 = bin_interp(AdB[j],
                    AdB[j-1], CdB[j], CdB[j-1], M, 0.5)
                asl_msq = 10**(asl_ms_log/10) # this is the mean square value NOT the rms
                actfact = (sq/x_len)/asl_msq # this is the proportion of the time that the speech is deemed "active"
                c0= 10**(cl0/20) # this is the threshold above which the speech is deemed "active".
                break
    return asl_msq, actfact, c0

# --------------------------------------------------------------------------

def bin_interp(upcount, lwcount, upthr, lwthr, Margin, tol):

    if tol < 0:
        tol = -tol

    # Check if extreme counts are not already the true active value
    iterno = 1
    if np.abs(upcount - upthr - Margin) < tol:
        asl_ms_log = upcount
        cc = upthr
        return asl_ms_log, cc

    if np.abs(lwcount - lwthr - Margin) < tol:
        asl_ms_log= lwcount
        cc= lwthr
        return asl_ms_log, cc

    # Initialize first middle for given (initial) bounds
    midcount = (upcount + lwcount) / 2.0
    midthr = (upthr + lwthr) / 2.0

    # Repeats loop until `diff' falls inside the tolerance (-tol<=diff<=tol)
    while 1:
        diff = midcount-midthr-Margin
        if abs(diff) <= tol:
            break

        # if tolerance is not met up to 20 iteractions, then relax the
        # tolerance by 10#
        iterno += 1

        if iterno>20:
            tol = tol*1.1

        if diff > tol:   # then new bounds are...
            midcount = (upcount+midcount)/2.0
            # upper and middle activities
            midthr = (upthr+midthr)/2.0
            # ... and thresholds
        elif diff < -tol:  # then new bounds are...
            midcount = (midcount+lwcount)/2.0
            # middle and lower activities
            midthr = (midthr+lwthr)/2.0
            # ... and thresholds

    #   Since the tolerance has been satisfied, midcount is selected
    #   as the interpolated value with a tol [dB] tolerance.

    asl_ms_log = midcount
    cc = midthr
    return asl_ms_log, cc
