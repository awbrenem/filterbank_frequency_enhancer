# Calculate frequency for a narrow band signal from filter bank data. Based on
# IDL code written by A. Brenneman (2013).
#
# TRANSLATION FROM ORIGINAL IDL CODE IN "InterpFilterBank" IS NOT
# FINISHED YET AND NEEDS MORE WORK. MODIFIED VERSION IN "InterpFilterBank2d"
# WORKS BUT STILL NEEDS TO BE COMPARED TO THE ORIGINAL IDL CODE.
#
# 02-04-2021 M. T. Johnson: Conversion from IDL to Python. Separated code into
#   three groups of functions: loading gain curves, finding filter bank data
#   that meet criteria for processing, and predicting frequency of a narrow
#   signal.
# 03-09-2021 M. T. Johnson: Collected the function for method1 and method2c and
#   method2d into classes. Looks like method2d works ok. Need to check the
#   other two.

import pathlib

import cdflib
import numpy as np
import scipy.io
from scipy import interpolate

# import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt


def get_gain_curves(filename=None, debug_plot=False):
    """
    Load the measured gain curves for the Van Allen Probes (RBSP)
    FBK frequency bins.

    :param filename: The name of the file to load. None=Use default file in
        the current directory.
    :param debug_plot: True=Make a plot of the gain curves for the frequency
        bins.
    :return: (freq, gain_curve) where freq is an array of frequencies and
        gain_curve is a 2D array of measured gain curves. Gain curves for
        each frequency bin are held in rows. First row has the lowest frequency
        bin.
    """

    if filename is None:
        filename = 'RBSP_FilterBank_Theoretical_Rsponse_wE12ACmeasuredResponse_DMM_20130305.sav'

    # Grab the gain curves from the IDL save file.
    path = pathlib.Path(filename)
    gain_curve_data = scipy.io.readsav(path.absolute())
    freq = gain_curve_data['freq'].copy()
    gaincurve = gain_curve_data['fb_theoretical_gainresponse_unitygain10khz'].copy()

    # Interpolate over a bad point in the one of the curves.
    gaincurve[7, 130] = np.interp(freq[130], freq[[129, 131]],
                                  gaincurve[7, [129, 131]])

    # Data is loaded with lowest index having the highest frequency. Want
    # lowest index to have the lowest frequency. Reverse the order of the rows.
    gaincurve = gaincurve[::-1, :]

    # Quick plot to look at them.
    if debug_plot:
        fig, ax = plt.subplots()
        ax.grid(True)
        for n, g in enumerate(gaincurve):
            ax.semilogx(freq, g, '.-', label=f'index={n}')
        ax.legend()
        ax.set_title('as selected')

    return freq, gaincurve


def normalize_gaincurves(freq, gaincurve, fbk_mode='7', debug_plot=False, all_same=False):

    # Select the channels for seven channels mode, else use all 13 channels.
    if fbk_mode == '7':
        gaincurve = gaincurve[[0, 2, 4, 6, 8, 10, 12]]

    # Normalize the gain curves. I'll use these b/c I'm assuming that
    # the FBK data is already calibrated. However, note that
    # each gain curve is somewhat offset from the others (most
    # notably the highest one which is about 2.1x larger than the
    # lowest). If I don't take this into account than the
    # interpolation doesn't work as well. However, my testing shows that
    # the results are more accurate if I only modify the last bin.

    # The normalization factor (ranges from 1 to 2.1). gaincurve_normfactor
    # gives the scale of each bin relative to the lowest amplitude bin.
    gaincurve_maxy = gaincurve.max(axis=1)
    gaincurve_normfactor = gaincurve_maxy / gaincurve_maxy.min()

    # Scale each gain curve so that its maximum is one.
    gaincurve_norm = gaincurve.T / gaincurve.T.max(0)
    gaincurve_norm = gaincurve_norm.T

    # MODIFY ALL BINS? NO, RESULTS NOT
    # AS ACCURATE AS ONLY MODIFYING
    # THE LAST ONE (SEE RBSP_EFW_FBK_FREQ_INTERPOLATE_TEST.PRO)
    # gain_curve_norm = gain_curve_norm.T * gain_curve_normfactor
    # gain_curve_norm = gain_curve_norm.T

    # The the gain curves normalized to one except for the the highest
    # frequency bin that is scale to have a higher (2.1x higher) relative
    # ampitude.
    if not all_same:
        gaincurve_norm[-1] = gaincurve[-1] * gaincurve_normfactor[-1]

    # Quick plot to look at them.
    if debug_plot:
        fig, ax = plt.subplots()
        ax.grid(True)
        for n, g in enumerate(gaincurve_norm):
            ax.semilogx(freq, g, '.-', label=str(n))
        ax.legend()
        ax.set_title('normalized')

    return freq, gaincurve_norm


def interpolate_gain_curves(freq, gaincurve, num_per_seg=500, debug_plot=False):
    """
    Interpolate the given gain curves to a different frequency step. RBSP gain
    curves are broken into five segments with increasing steps sizes staring at
    0.018 Hz and going up by a factor of 10 for each segment. Interpolate
    using a spline.

    :param freq: Array of frequencies for the bin calibration.
    :param gaincurve: 2D array with gain curves to interpolate. One gain_curve
        per row.
    :param num_per_seg: Number of points to use for each segment. Original curves
        had 50.
    :param debug_plot: True=Make plot for debugging.
    :return: (freq2, gc2) where freq2 is the new array of frequencies and gc2
        is the interpolated gain_curve values.
    """

    # -----------------------------------------------------------------------
    # Get the filter bank frequency bin definitions.

    # TODO REMOVED DUPLICATE POINTS AT THE ENDS OF EACH SEGMENT.
    # Increased frequency resolution. Original calibration has 5 segments
    # with linear frequency steps. Each increases by a factor of 10 for each
    # segment.
    range1 = np.array([0, 50, 100, 150, 200])
    range2 = np.array([50, 100, 150, 200, 249])
    freq2 = np.array([])
    for index1, index2 in zip(range1, range2):
        rng = freq[index2] - freq[index1]
        values = np.arange(num_per_seg) * rng / num_per_seg + freq[index1]
        freq2 = np.concatenate([freq2, values])
    freq2 = np.concatenate([freq2, np.array([freq[-1]])])

    # Interpolate the gain curves to the finer frequency step.
    gc2 = np.zeros((len(gaincurve), len(freq2)))
    for gc, curve in zip(gc2, gaincurve):
        tck = interpolate.splrep(freq, curve, s=0)
        gc[:] = interpolate.splev(freq2, tck, der=0)

    # TODO Interpolating after normalizing causes interpolated max to go above 1.0
    #  Do we want this? Maybe normalize after interpolating?

    # Some plots to make sure the interpolation looks good.
    if debug_plot:

        # # Make a plot for each bin.
        # for n in range(len(gain_curve)):
        #     fig, ax = plt.subplots()
        #     ax.semilogx(freq, gain_curve[n], 'o-', label='Original')
        #     ax.semilogx(freq2, gc2[n], '.-', label='Fine step')
        #     ax.grid(True)
        #     ax.legend()

        # Quick plot to look at all of them at once.
        fig, ax = plt.subplots()
        ax.grid(True)
        for n, g in enumerate(gaincurve):
            ax.semilogx(freq, g, 'x-', label=str(n))
        ax.set_prop_cycle(None)
        for n, g in enumerate(gc2):
            ax.semilogx(freq2, g, '.-', label=str(n))
        ax.legend()
        ax.set_title('interpolated')

    return freq2, gc2


# TODO NEEDS FIXING.
# Original filter bank processing.
class InterpFilterBank(object):
    """
    Python implementation of original, IDL frequency calculation code.
    """

    def __init__(self):
        """
        Calculation of frequencies using frequency bin peak amplitudes and
        averages.
        """

        self.bins = None
        self.check = None

        # Load the gain curves for RBSP
        self.freq_raw, self.gain_raw = get_gain_curves()

        # Normalize and refine frequency step.
        _, gain_norm = normalize_gaincurves(self.freq_raw, self.gain_raw,
                                            all_same=True)
        self.freq, self.gain_norm = interpolate_gain_curves(self.freq_raw,
                                                            gain_norm)

    def fit(self, peak, avg=None, keep_edges=True, minamp=2.0,
            noise_level=0.1, scale_factor_limit=2.0, maxamp_limit=2.0):
        """
        Find the frequency of a narrow band wave in the frequency bank.
        THIS ONE IS NOT COMPLETE. NEEDS MORE WORK TO MATCH ORIGINAL IDL CODE.

        If average values are given then also check to see if the the shape of
        the wave in the banks is also similar when it was passed through the
        filters.

        :param peak: Peak values from the the filter bank.
        :param avg: Average values from the filer bank.
        :param keep_edges: If True, include the lowest and highest bins when
            finding the maximum measurement in the bank.
        :param minamp: the minimum allowed amplitude for the maximum peak
             found in the filter bank.
        :param noise_level: The noise floor of the measurement in the filter
            bank bins.
        :param scale_factor_limit: The maximum scaling factor allowed on the
            maximum peak value than was found.
        :param maxamp_limit: The lowest value that is allow on the value
             will be scaled to be the real amplitde of the wave.
        :return: A numpy structures array with columns as the output of the
            frequency and amplitude calcluation.
        """

        self.bins = self.find_peaks(peak, avg=avg, keep_edges=keep_edges)
        self.check = self.find_valid_bins(self.bins, minamp=minamp,
                                          noise_level=noise_level,
                                          scale_factor_limit=scale_factor_limit)

        self.out = self.calc_freq_method1(self.bins, valid=self.check,
                                          maxamp_lim=maxamp_limit)

        return self.out

    def fitb(self, peak, avg=None, keep_edges=True, minamp=2.0,
             noise_level=0.1, scale_factor_limit=2.0, maxamp_limit=2.0):
        """
        Find the frequency of a narrow band wave in the frequency bank.

        If average values are given then also check to see if the the shape of
        the wave in the banks is also similar when it was passed through the
        filters.

        Note: This is the same as the method "fit" except that it removes
        the the assumption that the gain of the bin with the maximum peak is
        one.

        :param peak: Peak values from the the filter bank.
        :param avg: Average values from the filer bank.
        :param keep_edges: If True, include the lowest and highest bins when
            finding the maximum measurement in the bank.
        :param minamp: the minimum allowed amplitude for the maximum peak
             found in the filter bank.
        :param noise_level: The noise floor of the measurement in the filter
            bank bins.
        :param scale_factor_limit: The maximum scaling factor allowed on the
            maximum peak value than was found.
        :param maxamp_limit: The lowest value that is allow on the value
             will be scaled to be the real amplitde of the wave.
        :return: A numpy structures array with columns as the output of the
            frequency and amplitude calcluation.
        """

        self.bins = self.find_peaks(peak, avg=avg, keep_edges=keep_edges)
        self.check = self.find_valid_bins(self.bins, minamp=minamp,
                                          noise_level=noise_level,
                                          scale_factor_limit=scale_factor_limit)

        self.out = self.calc_freq_method1b(self.bins, valid=self.check,
                                           maxamp_lim=maxamp_limit)

        return self.out

    @staticmethod
    def find_peaks(peak, avg=None, keep_edges=True):
        """
        Find the largest bins and it's adjacent bin the larger amplitude.

        Exclude bins that have the maximum and are highest of lowest
        frequency bin. If the average values are also given, then return the
        corresponding average values for the identified bins.

        :param peak: Array of samples that have the maximum in each frequency bin.
            One sample per row.
        :param avg: Array of average values in the bins for each sample. Has the
            same shape as parameter peak.
        :return: A structured array with the identified indices and maximum values
            of the maximum and adjacent bins. Index values for adjacent bins that
            are not value are set to the index for the maximum bin.
        """
        res = np.zeros(len(peak), dtype=[('i_ctr', np.int), ('m_ctr', np.float),
                                         ('i_adj', np.int), ('m_adj', np.float),
                                         ('i_other', np.int), ('m_other', np.float),
                                         ('adj_lower', '?'), ('adj_higher', '?'),
                                         ('a_ctr', np.float), ('a_adj', np.float),
                                         ('a_other', np.float)])

        # Find the index of the bin with the maximum.
        res['i_ctr'] = np.argmax(peak, axis=1)
        n_bins = peak.shape[1]

        # Find the indices and values in the two neighboring bins.
        i_low = np.where(res['i_ctr']-1 > -1, res['i_ctr']-1, res['i_ctr'])
        i_high = np.where(res['i_ctr']+1 < n_bins, res['i_ctr']+1, res['i_ctr'])
        m_low = np.choose(i_low, peak.T)
        m_high = np.choose(i_high, peak.T)

        # Pick the larger on to be the adjacent one.
        res['i_adj'] = np.where(m_low > m_high, i_low, i_high)
        res['i_other'] = np.where(m_low <= m_high, i_low, i_high)

        # Special case, check for edge bins.
        if keep_edges:
            res['i_adj'] = np.where(res['i_ctr'] == 0, 1, res['i_adj'])
            res['i_other'] = np.where(res['i_ctr'] == 0, 0, res['i_other'])
            res['i_adj'] = np.where(res['i_ctr'] == n_bins-1, n_bins-2, res['i_adj'])
            res['i_other'] = np.where(res['i_ctr'] == n_bins-1, n_bins-1, res['i_other'])

        # Find which side the adjacent bin is on.
        res['adj_higher'] = res['i_adj'] > res['i_ctr']
        res['adj_lower'] = res['i_adj'] < res['i_ctr']

        # Get values at center and neighboring bins.
        np.choose(res['i_ctr'], peak.T, out=res['m_ctr'])
        np.choose(res['i_adj'], peak.T, out=res['m_adj'])
        np.choose(res['i_other'], peak.T, out=res['m_other'])

        # Get average values if given.
        if avg is not None:
            np.choose(res['i_ctr'], avg.T, out=res['a_ctr'], mode='clip')
            np.choose(res['i_adj'], avg.T, out=res['a_adj'], mode='clip')
            np.choose(res['i_other'], avg.T, out=res['a_other'], mode='clip')
        else:
            res['a_ctr'] = np.nan
            res['a_adj'] = np.nan
            res['a_other'] = np.nan

        # Possible useful metrics
        amp_ratio = np.full_like(res['m_ctr'], np.inf)
        denom_good = res['m_ctr'] > 0.0
        amp_ratio[denom_good] = res['m_adj'][denom_good] / res['m_ctr'][denom_good]

        return res

    @staticmethod
    def find_valid_bins(bins, minamp=2.0, noise_level=0.1,
                        scale_factor_limit=2.0):
        """
        Find the bins with peaks that are OK to interpolate.

        Will not interpolate if: 1) Either of the adjacent bins is zero. 2)
        The max peak is in the lowest or highest bin. 3) The peak value is
        too low. 4) The adjacent bin is below the noise floor. Or, 5) The gain
        to scale the center value is unreasonably high.

        :param bins: The bins loaded using the method "find_peaks"
        :param minamp: The minimum value to allow for the maximum peak.
        :param noise_level: The noise level in the bins.
        :param scale_factor_limit: The maximum scaling factor allowed for
            the center bin.
        :return: A structured array of boolean values indicating if the
            bins have passed the checks.
        """
        # TODO UPDATE INPUT DEFAULTS.

        # Results storage.
        v = np.zeros(len(bins), dtype=[('edgebin', np.bool),
                                       ('smallamp', np.bool),
                                       ('adj_toosmall', np.bool),
                                       ('sf_exceeded', np.bool),
                                       ('no_interp', np.bool),
                                       ('valid', np.bool)])

        # 1) If either of the adjacent bins is zero then we won't interpolate.
        # 2) If max value is in lowest or highest bin then no interpolation.
        v['edgebin'] = ((bins['m_adj'] == 0.0) | (bins['m_other'] == 0.0) |
                        (bins['i_ctr'] == bins['i_adj']) |
                        (bins['i_ctr'] == bins['i_other']))

        # 3) If center value is too low then no interpolation
        v['smallamp'] = (bins['m_ctr'] < minamp)

        # 4) If neighboring bin value is at noise level then no interpolation.
        v['adj_toosmall'] = (bins['m_adj'] <= noise_level)

        # 5) If amplitude scaling is too large then no interpolation.
        v['sf_exceeded'] = np.zeros_like(bins['m_adj'], np.bool)
        if scale_factor_limit is not None:
            amp_ratio = (bins['m_adj'][~v['smallamp']] /
                            bins['m_ctr'][~v['smallamp']])
            v['sf_exceeded'][~v['smallamp']] = amp_ratio > scale_factor_limit

        # No interpolation if any of the above conditions are true.
        v['no_interp'] = (v['edgebin'] | v['smallamp'] |
                          v['adj_toosmall'] | v['sf_exceeded'])
        v['valid'] = ~v['no_interp']

        return v

    # Narrow-band frequency calculation from original IDL code.
    def calc_freq_method1(self, bins, valid=None, maxamp_lim=2.0):
        """
        Find the narrow-band frequency in the wave found in the bin with the
        largest peak.

        This is the algorithm from the original IDL code.

        :param bins: A structured array with the bin data.
        :param valid: A structured array ingicating if the bins are valid to
            fit.
        :param maxamp_lim: The maximum allowed amplitude allowed for the
            corrected center bin amplitude.
        :return: A structured array with the results of the interpolation.
        """

        # Results storage.
        out = np.full(len(bins), np.nan, dtype=[('real_freq', 'f8'), ('real_amp', 'f8'),
                                                ('i_freq', 'i4'), ('scaling', 'f8'),
                                                ('distance_ctr', 'f8'), ('gain', 'f8')])
        out['real_amp'] = -1.0
        limited_to_maxamp_limit = np.zeros_like(bins['m_ctr'], dtype=np.bool)

        # Gain curves in dB relative to the peak at 1.0
        gain_dB = 20.0 * np.log10(self.gain_norm)

        # Number of dB in reduction that adj FBK bins see.
        amp_ratio = bins['m_adj'] / bins['m_ctr']
        dB_all = 20 * np.log10(amp_ratio)

        # For all valid points
        i_good = np.nonzero(valid['valid'])[0]
        for i in i_good:

            # Shorthand for rows
            b = bins[i]
            dB = dB_all[i]
            out1 = out[i]

            # Find the corrected frequency
            goo = np.nonzero(gain_dB[b['i_adj']] <= dB)[0]
            loc2 = np.nanargmax(gain_dB[b['i_adj']])  # index of max in adj gain curve
            if b['adj_higher']:
                locf = goo[np.nonzero(goo <= loc2)[0]]  # For adj higher
                out1['real_freq'] = self.freq[max(locf)]
            elif b['adj_lower']:
                locf = goo[np.nonzero(goo >= loc2)[0]]  # For adj lower
                out1['real_freq'] = self.freq[min(locf)]
            else:
                out1['real_freq'] = np.nan  # TODO update this us use freq at i_peak bin max.

            # Find the gain for the corrected amplitude.
            if b['adj_higher']:
                goo2 = np.nonzero(self.freq <= out1['real_freq'])[0]
                dB_new = gain_dB[b['i_ctr'], goo2[len(goo2) - 1]]
            elif b['adj_lower']:
                goo2 = np.nonzero(self.freq >= out1['real_freq'])[0]
                dB_new = gain_dB[b['i_ctr'], goo2[0]]
            else:
                dB_new = np.nan  # TODO update to use o_peak bin maximum

            # Calculate the corrected amplitude.
            out1['gain'] = 10**(dB_new / 20.0)
            out1['scaling'] = 1.0 / 10**(dB_new / 20.0)
            limited_to_maxamp_limit[i] = out1['scaling'] > maxamp_lim
            if limited_to_maxamp_limit[i]:
                out1['scaling'] = maxamp_lim
            out1['real_amp'] = b['m_ctr'] * out1['scaling']

        return out


    # Similar to original code except removes the assumption that the center
    # gain at the frequency in the center bin is one.
    def calc_freq_method1b(self, bins, valid=None, maxamp_lim=2.0):
        """
        Find the narrow-band frequency in the wave found in the bin with the
        largest peak. Same as method "calc_freq_method1" except removes the
        assumption that the gain for the center bin is near zero.

        :param bins: A structured array with the bin data.
        :param valid: A structured array ingicating if the bins are valid to
            fit.
        :param maxamp_lim: The maximum allowed amplitude allowed for the
            corrected center bin amplitude.
        :return: A structured array with the results of the interpolation.
        """

        # Results storage.
        out = np.full(len(bins), np.nan, dtype=[('real_freq', 'f8'), ('real_amp', 'f8'),
                                                ('i_freq', 'i4'), ('scaling', 'f8'),
                                                ('distance_ctr', 'f8'), ('gain', 'f8')])
        out['real_amp'] = -1.0
        limited_to_maxamp_limit = np.zeros_like(bins['m_ctr'], dtype=np.bool)

        # Gain curves in dB relative to the peak at 1.0
        gain_dB = 20.0 * np.log10(self.gain_norm)

        # Number of dB in reduction that adj FBK bins see.
        amp_ratio = bins['m_adj'] / bins['m_ctr']
        dB_all = 20 * np.log10(amp_ratio)

        # For all valid points
        i_good = np.nonzero(valid['valid'])[0]
        for i in i_good:

            # Shorthand for rows
            b = bins[i]
            dB = dB_all[i]
            out1 = out[i]

            # Find the corrected frequency
            goo = np.nonzero(gain_dB[b['i_adj']] <= (dB - gain_dB[b['i_ctr']]))[0]
            loc2 = np.nanargmax(gain_dB[b['i_adj']])  # index of max in adj gain curve
            if b['adj_higher']:
                locf = goo[np.nonzero(goo <= loc2)[0]]  # For adj higher
                out1['real_freq'] = self.freq[max(locf)]
            elif b['adj_lower']:
                locf = goo[np.nonzero(goo >= loc2)[0]]  # For adj lower
                out1['real_freq'] = self.freq[min(locf)]
            else:
                out1['real_freq'] = np.nan  # TODO update this us use freq at i_peak bin max.

            # Find the gain for the corrected amplitude.
            if b['adj_higher']:
                goo2 = np.nonzero(self.freq <= out1['real_freq'])[0]
                dB_new = gain_dB[b['i_ctr'], goo2[len(goo2) - 1]]
            elif b['adj_lower']:
                goo2 = np.nonzero(self.freq >= out1['real_freq'])[0]
                dB_new = gain_dB[b['i_ctr'], goo2[0]]
            else:
                dB_new = np.nan  # TODO update to use o_peak bin maximum

            # Calculate the corrected amplitude.
            out1['gain'] = 10**(dB_new / 20.0)
            out1['scaling'] = 1.0 / 10**(dB_new / 20.0)
            limited_to_maxamp_limit[i] = out1['scaling'] > maxamp_lim
            if limited_to_maxamp_limit[i]:
                out1['scaling'] = maxamp_lim
            out1['real_amp'] = b['m_ctr'] * out1['scaling']

        return out


# Modified to directly solve gain balance for the two bins. Added check using
# bin average values.
class InterpFilterBank2c(object):
    """
    Find frequencies and amplitudes of the largest wave in the filter
    bank assuming that it is a narrow-band wave.

    Do this by solving the equation that assumes the the relative amplitudes of
    the peaks in the bins is the same as for the gain curves at the
    narrow-band wave frequency.

    Note: Currently the "other" bin is not used. It should be check for
    consistency with the equation solving the frequency for the max peak and
    the adjacent peak.
    """

    def __init__(self):
        """
        Calculation of frequencies using frequency bin peak amplitudes and
        averages.
        """

        self.bins = None
        self.check = None
        self.out = None

        # Load the gain curves for RBSP
        self.freq_raw, self.gain_raw = get_gain_curves()

        # Normalize and refine frequency step.
        _, gain_norm = normalize_gaincurves(self.freq_raw, self.gain_raw,
                                            all_same=True)
        self.freq, self.gain_norm = interpolate_gain_curves(self.freq_raw,
                                                           gain_norm)

    def fit(self, peak, avg=None, keep_edges=True, minamp=2.0,
            noise_level=0.1, bin_limit=3, debug_plots=False):
        """
        Find the frequency of a narrow band wave in the frequency bank. If
        average values are given then also check to see if the the shape of
        the wave in the banks is also similar when it was passed through the
        filters.

        :param peak: Peak values from the the filter bank.
        :param avg: Average values from the filer bank.
        :param keep_edges: If True, include the lowest and highest bins when
            finding the maximum measurement in the bank.
        :param minamp: the minimum allowed amplitude for the maximum peak
             found in the filter bank.
        :param noise_level: The noise floor of the measurement in the filter
            bank bins.
        :param bin_limit: The index of the lowest freuqnecy bin allow in the
            caluclation.
        :return: A numpy structures array with columns as the output of the
            frequency and amplitude calculation.
        """

        self.bins = self.find_peaks2c(peak, avg=avg, keep_edges=keep_edges)
        self.check = self.find_valid_bins2c(self.bins, minamp=minamp,
                                            noise_level=noise_level,
                                            scale_factor_limit=2.0,
                                            bin_limit=bin_limit)

        self.out = self.calc_freq_method2c(self.bins, valid=self.check,
                                           debug_plots=debug_plots,
                                           peak=peak, noise_level=noise_level)

        return self.out

    # TODO Might not need the "other" bin. Because is is always == 0 when
    #  the adjacent bin == 0.
    @staticmethod
    def find_peaks2c(peak, avg=None):
        """
        For each sample, find the bin with the largest peak and it's neighboring
        bins. If either of the neighboring bins is out of the index range, then
        return a value of zero. The larger of the neighbors is the "adjacent" bin,
        the smaller neighbor is the "other" bin.

        :param peak: Array of samples that have the maximum in each frequency bin.
            One sample per row.
        :param avg: Array of average values in the bins for each sample. Has the
            same shape as parameter peak.
        :return: A structured array with the identified indices and maximum values
            of the maximum, adjacent, and other bins.
        """

        # Allocate storage for the results of the peak selection.
        res = np.zeros(len(peak),
                       dtype=[('i_ctr', np.int), ('m_ctr', np.float),
                              ('i_adj', np.int), ('m_adj', np.float),
                              ('i_other', np.int), ('m_other', np.float),
                              ('adj_lower', np.bool), ('adj_higher', np.bool),
                              ('a_ctr', np.float), ('a_adj', np.float),
                              ('a_other', np.float), ('arv_ctr', np.float),
                              ('arv_adj', np.float), ('arv_ratio', 'f8'),
                              ('arv_metric', 'f8')])
        n_bins = peak.shape[1]

        # Find the index and value in the the bin with the maximum.
        res['i_ctr'] = np.argmax(peak, axis=1)
        res['m_ctr'] = np.choose(res['i_ctr'], peak.T)

        # Find freq bin one lower than the maximum. Out of range value is 0.0
        i_low = res['i_ctr'] - 1
        m_low = np.choose(i_low, peak.T, mode='clip')
        m_low[i_low == -1] = 0.0

        # Find freq bin one higher than the maximum. Out of range value is 0.0
        i_high = res['i_ctr'] + 1
        m_high = np.choose(i_high, peak.T, mode='clip')
        m_high[i_high == n_bins] = 0.0

        # Find the higher of the two neighboring bins as the adjacent one.
        res['i_adj'] = (m_low >= m_high) * i_low + (m_low < m_high) * i_high
        res['m_adj'] = (m_low >= m_high) * m_low + (m_low < m_high) * m_high

        # The other neighboring bin.
        res['i_other'] = (m_low < m_high) * i_low + (m_low >= m_high) * i_high
        res['m_other'] = (m_low < m_high) * m_low + (m_low >= m_high) * m_high

        # Find which side the adjacent bin is on.
        res['adj_higher'] = res['i_adj'] > res['i_ctr']
        res['adj_lower'] = res['i_adj'] < res['i_ctr']

        # Get average bin values if given. Out of range value = 0.0
        if avg is not None:
            res['a_ctr'] = np.choose(res['i_ctr'], avg.T)
            res['a_adj'] = np.choose(res['i_adj'], avg.T, mode='clip')
            res['a_adj'][(res['i_adj'] == n_bins) | (res['i_adj'] == -1)] = 0.0
            res['a_other'] = np.choose(res['i_other'], avg.T, mode='clip')
            res['a_other'][
                (res['i_other'] == n_bins) | (res['i_other'] == -1)] = 0.0
            ii = (res['m_adj'] != 0.0)
            res['arv_adj'][ii] = res['a_adj'][ii] / res['m_adj'][ii]
            ii = (res['m_ctr'] != 0.0)
            res['arv_ctr'][ii] = res['a_ctr'][ii] / res['m_ctr'][ii]
            ii = (res['arv_ctr'] != 0.0)
            res['arv_ratio'][ii] = res['arv_adj'][ii] / res['arv_ctr'][ii]
            res['arv_metric'] = res['arv_ratio'] - 1.0

        else:
            res['a_ctr'] = np.nan
            res['a_adj'] = np.nan
            res['a_other'] = np.nan
            res['arv_ctr'] = np.nan
            res['arv_adj'] = np.nan
            res['arv_ratio'] = np.nan
            res['arv_metric'] = np.nan

        return res

    @staticmethod
    def find_valid_bins2c(bins, minamp=2.0, noise_level=0.1,
                          scale_factor_limit=2.0, bin_limit=3,
                          use_avg=None):
        """
        Find the bins with peaks that are OK to interpolate.

        Will not interpolate if: 1) The peak value is too low. 2) The
        adjacent bin is below the noise floor. Or, 3) The gain
        to scale the center value is unreasonably high.

        :param bins: A structured array of the bin peak data from method
            "find_peaks2c"
        :param minamp: The minimum value to allow for the maximum peak.
        :param noise_level: The noise level in the bins.
        :param scale_factor_limit: The maximum scaling factor allowed for
            the center bin.
        :param bin_limit: The minimum bin index to allow search for the
            maximum peaks.
        :param use_avg: If True, try to use the average data to check for
            similar wave shapes in peak and adjacent bins. EXPERIMENTAL.
        :return: A structured array of boolean values indicating if the
            bins have passed the checks.
        """
        # TODO UPDATE INPUT DEFAULTS.

        # Results storage.
        v = np.zeros(len(bins), dtype=[
            ('centerbin', np.bool),
            ('edgebin', np.bool),
            ('ctr_too_small', np.bool),
            ('adj_too_small', np.bool),
            ('sf_exceeded', np.bool),
            ('no_interp', np.bool),
            ('valid', np.bool),
            ('bad_arv', np.bool),
            ('bin_limit', np.bool)])

        # TODO Edge bins are now accepted. If m_adj is zero, then m_other must
        #   also be zero. A check for a possible frequency calculation can be
        #   done by checking if m_adj ==0.0. If m_adj is zero, then no frequency
        #   can be found. This needs to be checked during frequency caluclation.

        # 3) If center value is too low then no interpolation
        if minamp is not None:
            v['ctr_too_small'] = (bins['m_ctr'] < minamp)

        # TODO It might be possible to remove the lower limit on the noise in the
        #   adjacent bin. Low values of m_adj relative to m_ctr will push the
        #   calculated frequency to the frequency at the center of the i_ctr bin.

        # 4) If neighboring bin value is at noise level then no interpolation. Still allow
        # if the center value is big enough, Means is near center of bin.
        if noise_level is not None:
            v['adj_too_small'] = (bins['m_adj'] <= noise_level)

        # 5) If amplitude scaling is too large then no interpolation.
        if scale_factor_limit is not None:
            amp_ratio = (bins['m_adj'][~v['smallamp']] /
                         bins['m_ctr'][~v['smallamp']])
            v['sf_exceeded'][~v['smallamp']] = amp_ratio > scale_factor_limit

        # If the bin has too low of a frequency then can not determine if is narrow band
        if bin_limit is not None:
            v['bin_limit'] = bins['i_ctr'] <= bin_limit

        # Check if avg/peak is the same for the bin. Is an indicator that the
        # waves are narrow band in both bins. Might be to restrictive?
        if use_avg:
            arv_ctr = bins['a_ctr'] / bins['m_ctr']
            arv_adj = bins['a_adj'] / bins['m_adj']
            metric = arv_adj / arv_ctr
            v['bad_arv'] = np.abs(metric - 1.0) > 0.1
        else:
            v['bad_arv'] = False

        # No interpolation if any of the above conditions are true.
        v['no_interp'] = (v['edgebin'] | v['ctr_too_small'] |
                          v['adj_too_small'] | v['sf_exceeded'] |
                          v['bad_arv']) | v['bin_limit']
        v['valid'] = ~v['no_interp']

        return v

    # Find solution using zero searching by using zero finding.
    def calc_freq_method2c(self, bins, valid=None, debug_plots=True,
                           peak=None, noise_level=0.1):
        """
        Find frequencies and amplitudes of the largest wave in the filter
        bank assuming that it is a narrow-band wave.

        Do this by solving the equation that assumes the the relative amplitudes
        of the peaks in the bins is the same as for the gain curves at the
        narrow-band wave frequency.

        :param bins: A structured array with the bin data.
        :param valid: A structured array indicating if the bins are valid to
            fit.
        :param debug_plots: If True, generate debug plots.
        :param peak: The measured peak values, only needed for debug plots.
        :param noise_level: Relative of center and adjacent bins. Could be a
            quality metric.
        :return: A structured array with the results of the interpolation.
        """

        # All points are valid if not given selection.
        if valid is None:
            valid = np.ones(len(bins), dtype=[('valid', '?')])

        if debug_plots:
            fig, axs = plt.subplots(2, 1)
            fig.set_tight_layout({'h_pad': 0.05})
            fig.set_size_inches([13.24, 4.8])
            for ax in axs:
                ax.grid(True)

        # Results storage.
        out = np.full(len(bins), np.nan, dtype=[('real_freq', 'f8'),
                                                ('real_amp', 'f8'),
                                                ('i_freq', 'i4'),
                                                ('scaling', 'f8'),
                                                ('distance_ctr', 'f8'),
                                                ('gain', 'f8'),
                                                ('near_limit', 'f8'),
                                                ('i_freq_noise', 'i4'),
                                                ('freq_noise', 'f8')])
        out['real_amp'] = -1.0

        # Temporary storage for gain curve calculations.
        cut = np.empty_like(self.freq, dtype=np.bool)
        metric = np.zeros_like(self.gain_norm[0], dtype=np.float)

        # Find the frequency for the maximum values of the gain curves.
        i_gain_max = np.nanargmax(self.gain_norm, axis=1)
        freq_gain_max = self.freq[i_gain_max]

        # # TODO limits are hard coded.
        # # Special case for bins with both neighbors having zero peak value.
        # # Assume that the peak is at the center of the gain curve.
        # alone = (bins['m_ctr'] > 2.0) & (bins['m_adj'] < 0.1) & (bins['i_ctr'] >= 3)
        # out['real_freq'][alone] = freq_gain_max[bins['i_ctr'][alone]]
        # out['real_amp'][alone] = bins['m_ctr'][alone]
        # out['i_freq'][alone] = -1  # TODO find closest to bin max?
        # out['scaling'][alone] = 1.0
        # out['gain'][alone] = 1.0  # TODO Normalized makes off a little.

        # print('points alone', np.count_nonzero(alone))

        # Calculate the frequencies for points that have meet the valid criteria.
        i_good = np.nonzero(valid['valid'] & (bins['m_adj'] != 0.0))[0]
        print('Points to interpolate', len(i_good))
        for i in i_good:

            # Shorthand for each row.
            out1 = out[i]
            b = bins[i]
            v = valid[i]

            #  Can only find a frequency between the two gain curve maxima.
            if b['adj_higher']:
                cut[:] = ((self.freq > freq_gain_max[b['i_ctr']]) &
                          (self.freq < freq_gain_max[b['i_adj']]))
            if b['adj_lower']:
                cut[:] = ((self.freq > freq_gain_max[b['i_adj']]) &
                          (self.freq < freq_gain_max[b['i_ctr']]))

            # Find the index of the frequency where m_peak/gain_peak = m_adj/gain_adj
            np.abs(self.gain_norm[b['i_adj']] * b['m_ctr'] -
                   self.gain_norm[b['i_ctr']] * b['m_adj'], out=metric)
            i_at_min = np.argmin(metric[cut])
            i_freq_corr = np.nonzero(self.freq == self.freq[cut][i_at_min])[0]
            real_freq = self.freq[i_freq_corr]

            # Find the frequency that is allowed at the adjacent bin noise floor
            metric2 = np.abs(self.gain_norm[b['i_adj']] * b['m_ctr'] -
                             self.gain_norm[b['i_ctr']] * noise_level)
            i_at_min2 = np.argmin(metric2[cut])
            i_freq_corr2 = np.nonzero(self.freq == self.freq[cut][
                i_at_min2])[0]
            real_freq_noise = self.freq[i_freq_corr2]
            out1['i_freq_noise'] = i_freq_corr2
            out1['freq_noise'] = real_freq_noise

            # Find the corrected amplitude. Calculations for both peak and adjacent
            # gaincurves should give the same amplitude.
            out1['gain'] = self.gain_norm[b['i_ctr'], i_freq_corr]
            out1['scaling'] = 1.0 / out1['gain']
            out1['real_amp'] = b['m_ctr'] * out1['scaling']
            out1['real_freq'] = real_freq
            out1['i_freq'] = i_freq_corr

            # Some extra calculations that might be useful.
            out1['distance_ctr'] = out1['real_freq'] - freq_gain_max[
                b['i_ctr']]

            # Calculate how far away the calculated amplitude is from the
            # maximum allowed by the noise floor of the adjacent bin.
            if noise_level is not None:
                # out1['near_limit'] = (b['m_ctr'] * gain_norm[b['i_adj'], i_freq_corr] /
                #                       noise_level / gain_norm[b['i_ctr'], i_freq_corr])
                # out1['near_limit'] = (noise_level * gain_norm[b['i_ctr'], i_freq_corr] /
                #                       gain_norm[b['i_adj'], i_freq_corr])
                out1['near_limit'] = 1.0 / (
                            self.gain_norm[b['i_ctr'], i_freq_corr] /
                            self.gain_norm[b['i_adj'], i_freq_corr])

            if debug_plots:

                if 'edgebin' in valid.dtype.names:
                    print(f'{i + 1}/{len(bins)}  edgebins={v["edgebin"]}, '
                          f'smallamp={v["smallamp"]}, adj_toosmall={v["adj_toosmall"]},'
                          f' sf_exceeded={v["sf_exceeded"]}, no_interp={v["no_interp"]},'
                          f' valid={v["valid"]}')

                # Some values for plotting.
                peak1 = peak[i] / b['m_ctr']
                m_min = peak1.min()

                for ax in axs:
                    for line in ax.get_lines():
                        line.remove()
                    ax.relim()
                    ax.set_prop_cycle(None)

                axs[0].set_title(
                    f'{i + 1}/{len(bins)}  meas_center={b["m_ctr"]:0.2f}, meas_adj={b["m_adj"]:0.2f}, '
                    f'meas_min={m_min:0.2f}\nfreq={out1["real_freq"]:0.2f}Hz, '
                    f'amp={out1["real_amp"]:0.2f}, gain={out1["gain"]:0.2f}, '
                    f'scaling={out1["scaling"]:0.2f}')

                ax = axs[0]
                ax.semilogx(self.freq, self.gain_norm[b['i_ctr']], linewidth=5)
                ax.semilogx(self.freq, self.gain_norm[b['i_adj']], linewidth=5)
                ax.semilogx([out1['real_freq']] * 2, ax.get_ylim(), 'k')
                if peak is not None:
                    for f, p in zip(freq_gain_max, peak1):
                        ax.semilogx(f, p, 'ro')
                        ax.semilogx([f, f], [0, p], 'r')
                    ax.plot(self.freq, self.gain_norm.T, 'y--')
                ax.legend(['Gain Center', 'Gain Adjacent', 'Calculated Freq',
                           'Amp relative to Max'])

                ax = axs[1]
                ax.semilogx(self.freq, metric)
                ax.semilogx(self.freq[cut], metric[cut], '.-')
                ax.semilogx(out1['real_freq'], metric[out1['i_freq']], 'ok')
                ax.semilogx([out1['real_freq']] * 2, ax.get_ylim(), 'k')

                fig.waitforbuttonpress()

        return out


# Modified 2c to have valid bins for processing to be True. Renamed variables
# so they make more sense.
class InterpFilterBank2d(object):
    """
    Find frequencies and amplitudes of the largest wave in the filter
    bank assuming that it is a narrow-band wave. Do this by solving the
    equation that assumes the the relative amplitudes of the peaks in the
    bins is the same as for the gain curves at the narrow-band wave
    frequency.

    Note: Try to use the "other" bin to check for value frequency
    calculation. Updated the names of the variable from the InterpFilterBank2c.
    In particular, changed the checking for valid bins to to True for values
    that should be processed instead of True for values not to process.
    """

    def __init__(self):
        """
        Calculation of frequencies using frequency bin peak amplitudes and
        averages.
        """

        self.bins = None
        self.check = None
        self.out = None

        # Load the gain curves for RBSP
        self.freq_raw, self.gain_raw = get_gain_curves()

        # Normalize and refine frequency step.
        _, gain_norm = normalize_gaincurves(self.freq_raw, self.gain_raw,
                                            all_same=True)
        self.freq, self.gain_norm = interpolate_gain_curves(self.freq_raw,
                                                           gain_norm)

    def fit(self, peak, avg=None, m_ctr_min=2.0, noise_level=0.1,
            bin_minimum=3, use_avg=0.2, include_alone=True,
            debug_plots=False):
        """
        Find the frequency of a narrow band wave in the frequency bank.

        If average values are given then also check to see if the the shape of
        the wave in the banks is also similar when it was passed through the
        filters.

        :param peak: Peak values from the the filter bank.
        :param avg: Average values from the filer bank.
        :param m_ctr_min: the minimum allowed amplitude for the maximum peak
             found in the filter bank.
        :param noise_level: The noise floor of the measurement in the filter
            bank bins.
        :param bin_minimum: Minimum bin to use inteh filter bank.
        :param use_avg: If average given use the target value to check for
            similarity of wave shape in the bins.
        :param include_alone: If True, include bins that have zeros on both
            sides.
        :param debug_plots: If True, make debug plots.
        :return: A numpy structures array with columns as the output of the
            frequency and amplitude calculation.
        """

        self.bins = self.find_peaks2c(peak, avg=avg)

        self.check = self.find_valid_bins2d(self.bins, m_ctr_min=m_ctr_min,
                                            noise_level=noise_level,
                                            bin_minimum=bin_minimum,
                                            use_avg=use_avg)

        self.out = self.calc_freq_method2d(self.bins, check=self.check,
                                  debug_plots=debug_plots,
                                  peak=peak, noise_level=noise_level,
                                  include_alone=include_alone)

        return self.out

    # TODO Might not need the "other" bin. Because is is always == 0 when
    #  the adjacent bin == 0.
    @staticmethod
    def find_peaks2c(peak, avg=None):
        """
        For each sample, find the bin with the largest peak and it's neighboring
        bins. If either of the neighboring bins is out of the index range, then
        return a value of zero. The larger of the neighbors is the "adjacent" bin,
        the smaller neighbor is the "other" bin.

        :param peak: Array of samples that have the maximum in each frequency bin.
            One sample per row.
        :param avg: Array of average values in the bins for each sample. Has the
            same shape as parameter peak.
        :return: A structured array with the identified indices and maximum values
            of the maximum, adjacent, and other bins.
        """

        # Allocate storage for the results of the peak selection.
        res = np.zeros(len(peak),
                       dtype=[('i_ctr', np.int), ('m_ctr', np.float),
                              ('i_adj', np.int), ('m_adj', np.float),
                              ('i_other', np.int), ('m_other', np.float),
                              ('adj_lower', np.bool), ('adj_higher', np.bool),
                              ('a_ctr', np.float), ('a_adj', np.float),
                              ('a_other', np.float), ('arv_ctr', np.float),
                              ('arv_adj', np.float), ('arv_ratio', 'f8'),
                              ('arv_metric', 'f8')])
        n_bins = peak.shape[1]

        # Find the index and value in the the bin with the maximum.
        res['i_ctr'] = np.argmax(peak, axis=1)
        res['m_ctr'] = np.choose(res['i_ctr'], peak.T)

        # Find freq bin one lower than the maximum. Out of range value is 0.0
        i_low = res['i_ctr'] - 1
        m_low = np.choose(i_low, peak.T, mode='clip')
        m_low[i_low == -1] = 0.0

        # Find freq bin one higher than the maximum. Out of range value is 0.0
        i_high = res['i_ctr'] + 1
        m_high = np.choose(i_high, peak.T, mode='clip')
        m_high[i_high == n_bins] = 0.0

        # Find the higher of the two neighboring bins as the adjacent one.
        res['i_adj'] = (m_low >= m_high) * i_low + (m_low < m_high) * i_high
        res['m_adj'] = (m_low >= m_high) * m_low + (m_low < m_high) * m_high

        # The other neighboring bin.
        res['i_other'] = (m_low < m_high) * i_low + (m_low >= m_high) * i_high
        res['m_other'] = (m_low < m_high) * m_low + (m_low >= m_high) * m_high

        # Find which side the adjacent bin is on.
        res['adj_higher'] = res['i_adj'] > res['i_ctr']
        res['adj_lower'] = res['i_adj'] < res['i_ctr']

        # Get average bin values if given. Out of range value = 0.0
        if avg is not None:
            res['a_ctr'] = np.choose(res['i_ctr'], avg.T)
            res['a_adj'] = np.choose(res['i_adj'], avg.T, mode='clip')
            res['a_adj'][(res['i_adj'] == n_bins) | (res['i_adj'] == -1)] = 0.0
            res['a_other'] = np.choose(res['i_other'], avg.T, mode='clip')
            res['a_other'][
                (res['i_other'] == n_bins) | (res['i_other'] == -1)] = 0.0
            ii = (res['m_adj'] != 0.0)
            res['arv_adj'][ii] = res['a_adj'][ii] / res['m_adj'][ii]
            ii = (res['m_ctr'] != 0.0)
            res['arv_ctr'][ii] = res['a_ctr'][ii] / res['m_ctr'][ii]
            ii = (res['arv_ctr'] != 0.0)
            res['arv_ratio'][ii] = res['arv_adj'][ii] / res['arv_ctr'][ii]
            res['arv_metric'] = res['arv_ratio'] - 1.0

        else:
            res['a_ctr'] = np.nan
            res['a_adj'] = np.nan
            res['a_other'] = np.nan
            res['arv_ctr'] = np.nan
            res['arv_adj'] = np.nan
            res['arv_ratio'] = np.nan
            res['arv_metric'] = np.nan

        return res

    @staticmethod
    def find_valid_bins2d(bins, m_ctr_min=2.0, noise_level=0.1, bin_minimum=3,
                          use_avg=0.1):
        """
        Find the valid bins to find the frequency.

        :param bins: A structured array with the bin data.
        :param m_ctr_min: the minimum allowed amplitude for the maximum peak
             found in the filter bank.
        :param noise_level: The noise floor of the measurement in the filter
            bank bins.
        :param bin_minimum: Minimum bin to use inteh filter bank.
        :param use_avg: If average given use the target value to check for
            similarity of wave shape in the bins.
        :return: A structured array of boolean values indicating if the
            bins have passed the checks.
        """

        # Results storage.
        v = np.zeros(len(bins), dtype=[
            ('ctr_large_enough', '?'),
            ('adj_large_enough', '?'),
            ('bin_index_good', '?'),
            ('arv_match', '?'),
            ('alone', '?'),
            ('good_amplitude', '?'),
            ('good', '?')])

        # Bins that have the m_ctr large enough.
        if m_ctr_min is not None:
            v['ctr_large_enough'] = (bins['m_ctr'] > m_ctr_min)

        # Bins that have m_adj above the noise floor.
        if noise_level is not None:
            v['adj_large_enough'] = (bins['m_adj'] > noise_level)

        # Bins low frequency bins have a low sampling rate. Makes sens to
        # pick bin index values that have ~ 10 points per sample (1/8 sec)
        # or more.
        if bin_minimum is not None:
            v['bin_index_good'] = bins['i_ctr'] >= bin_minimum

        # Check if avg/peak is the same for ctr and adj bins. This is a check
        # to see if the waveform shape is the same in both bins.
        if use_avg is not None:
            v['arv_match'] = np.abs(bins['arv_metric']) < use_avg
        else:
            v['arv_match'] = False

        # Bins that have a large enough amplitude but the adjacent is not large
        # enough for a frequency calculation or determining the gain.
        v['alone'] = (v['bin_index_good'] &
                      v['ctr_large_enough'] &
                      ~v['adj_large_enough'])

        # Bins that have large enough center peak and adjacent.
        v['good_amplitude'] = (v['bin_index_good'] &
                               v['ctr_large_enough'] &
                               v['adj_large_enough'])

        # Bins where amplitudes are large enough, and the arv (shape) of the
        # waves is similar for both bins.
        v['good'] = (v['bin_index_good'] &
                     v['ctr_large_enough'] &
                     v['adj_large_enough'] &
                     v['arv_match'])
        return v

    # Find solution using zero searching by using zero finding.
    def calc_freq_method2d(self, bins, check=None, debug_plots=True,
                           peak=None, noise_level=0.1, include_alone=True):
        """

        :param bins: A structured array with the bin data.
        :param check: A structured array of boolean values indicating if the
            bins have passed the checks.
        :param debug_plots: IF True, make debug plots.
        :param peak: The original ppeak values, only for debug plots.
        :param noise_level: The noise level for the bins, only needed for
            debug plots.
        :param include_alone: If True, include the center bins that are
            large enough but both adjacent bins are below the noise floor.
        :return: A numpy structures array with columns as the output of the
            frequency and amplitude calculation.
        """

        # All points are valid if not given selection.
        if check is None:
            check = np.ones(len(bins), dtype=[('valid', '?')])

        if debug_plots:
            fig, axs = plt.subplots(2, 1)
            fig.set_tight_layout({'h_pad': 0.05})
            fig.set_size_inches([13.24, 4.8])
            for ax in axs:
                ax.grid(True)

        # Results storage.
        out = np.full(len(bins), np.nan, dtype=[('real_freq', 'f8'),
                                                ('real_amp', 'f8'),
                                                ('i_freq', 'i4'),
                                                ('scaling', 'f8'),
                                                ('distance_ctr', 'f8'),
                                                ('gain', 'f8'),
                                                ('near_limit', 'f8'),
                                                ('i_freq_noise', 'i4'),
                                                ('freq_noise', 'f8'),
                                                ('gain_at_noise', 'f8')])

        # Temporary storage for gain curve calculations.
        cut = np.empty_like(self.freq, dtype=np.bool)
        metric = np.zeros_like(self.gain_norm[0], dtype=np.float)

        # Find the frequency for the maximum values of the gain curves.
        i_gain_max = np.nanargmax(self.gain_norm, axis=1)
        freq_gain_max = self.freq[i_gain_max]

        # Special case if m_ctr is large enough but m_adj is below the noise floor.
        if include_alone:
            alone = check['alone']
            out['real_freq'][alone] = freq_gain_max[bins['i_ctr'][alone]]
            out['real_amp'][alone] = bins['m_ctr'][alone]
            out['i_freq'][alone] = -1  # TODO find closest to bin max?
            out['scaling'][alone] = 1.0
            out['gain'][alone] = 1.0  # TODO Normalized makes off a little.
            print('points alone', np.count_nonzero(alone))

        # Calculate the frequencies for points that have meet the valid criteria.
        i_good = np.nonzero(check['good_amplitude'])[0]
        print('Points to interpolate', len(i_good))
        for i in i_good:

            # Shorthand for each row.
            out1 = out[i]
            b = bins[i]

            #  Can only find a frequency between the two gain curve maxima.
            if b['adj_higher']:
                cut[:] = ((self.freq > freq_gain_max[b['i_ctr']]) &
                          (self.freq < freq_gain_max[b['i_adj']]))
            if b['adj_lower']:
                cut[:] = ((self.freq > freq_gain_max[b['i_adj']]) &
                          (self.freq < freq_gain_max[b['i_ctr']]))

            # Find the index of the frequency where m_peak/gain_peak = m_adj/gain_adj
            np.abs(self.gain_norm[b['i_adj']] * b['m_ctr'] -
                   self.gain_norm[b['i_ctr']] * b['m_adj'], out=metric)
            i_at_min = np.argmin(metric[cut])
            i_freq_corr = np.nonzero(self.freq == self.freq[cut][i_at_min])[0]
            real_freq = self.freq[i_freq_corr]

            # Find the frequency that is allowed at the adjacent bin noise floor
            metric2 = np.abs(self.gain_norm[b['i_adj']] * b['m_ctr'] -
                             self.gain_norm[b['i_ctr']] * noise_level)
            i_at_min2 = np.argmin(metric2[cut])
            i_freq_corr2 = np.nonzero(self.freq == self.freq[cut][
                i_at_min2])[0]
            real_freq_noise = self.freq[i_freq_corr2]
            out1['i_freq_noise'] = i_freq_corr2
            out1['freq_noise'] = real_freq_noise
            out1['gain_at_noise'] = self.gain_norm[b['i_ctr'], i_freq_corr2]

            # Find the corrected amplitude. Calculations for both peak and adjacent
            # gaincurves should give the same amplitude.
            out1['gain'] = self.gain_norm[b['i_ctr'], i_freq_corr]
            out1['scaling'] = 1.0 / out1['gain']
            out1['real_amp'] = b['m_ctr'] * out1['scaling']
            out1['real_freq'] = real_freq
            out1['i_freq'] = i_freq_corr

            # Some extra calculations that might be useful.
            out1['distance_ctr'] = out1['real_freq'] - freq_gain_max[
                b['i_ctr']]

            # Calculate have far away the calculated amplitude is from the maximum
            # allowed by the noise floor of the adjacent bin.
            if noise_level is not None:
                # out1['near_limit'] = (b['m_ctr'] * gain_norm[b['i_adj'], i_freq_corr] /
                #                       noise_level / gain_norm[b['i_ctr'], i_freq_corr])
                # out1['near_limit'] = (noise_level * gain_norm[b['i_ctr'], i_freq_corr] /
                #                       gain_norm[b['i_adj'], i_freq_corr])
                out1['near_limit'] = 1.0 / (
                            self.gain_norm[b['i_ctr'], i_freq_corr] /
                            self.gain_norm[b['i_adj'], i_freq_corr])

            if debug_plots:

                # Some values for plotting.
                peak1 = peak[i] / b['m_ctr']
                m_min = peak1.min()

                for ax in axs:
                    for line in ax.get_lines():
                        line.remove()
                    ax.relim()
                    ax.set_prop_cycle(None)

                axs[0].set_title(
                    f'{i + 1}/{len(bins)}  meas_center={b["m_ctr"]:0.2f}, meas_adj={b["m_adj"]:0.2f}, '
                    f'meas_min={m_min:0.2f}\nfreq={out1["real_freq"]:0.2f}Hz, '
                    f'amp={out1["real_amp"]:0.2f}, gain={out1["gain"]:0.2f}, '
                    f'scaling={out1["scaling"]:0.2f}')

                ax = axs[0]
                ax.semilogx(self.freq, self.gain_norm[b['i_ctr']], linewidth=5)
                ax.semilogx(self.freq, self.gain_norm[b['i_adj']], linewidth=5)
                ax.semilogx([out1['real_freq']] * 2, ax.get_ylim(), 'k')
                if peak is not None:
                    for f, p in zip(freq_gain_max, peak1):
                        ax.semilogx(f, p, 'ro')
                        ax.semilogx([f, f], [0, p], 'r')
                    ax.plot(self.freq, self.gain_norm.T, 'y--')
                ax.legend(['Gain Center', 'Gain Adjacent', 'Calculated Freq',
                           'Amp relative to Max'])

                ax = axs[1]
                ax.semilogx(self.freq, metric)
                ax.semilogx(self.freq[cut], metric[cut], '.-')
                ax.semilogx(out1['real_freq'], metric[out1['i_freq']], 'ok')
                ax.semilogx([out1['real_freq']] * 2, ax.get_ylim(), 'k')

                fig.waitforbuttonpress()

        return out


if __name__ == '__main__':

    filename = 'rbspb_efw-l2_fbk_20150416_v01.cdf'

    # filename = 'rbspb_efw-l2_fbk_20150416_v01.cdf'
    # filename = 'rbspb_efw-l2_fbk_20160101_v01.cdf'

    # -----------------------------------------------------------
    # FOR TESTING: Load the data from a cdf file.

    # The the bins values from the CDF file.
    with cdflib.CDF(filename) as f:
        peak = f.varget('fbk7_e12dc_pk')
        avg = f.varget('fbk7_e12dc_av')
        freq_center = f.varget('fbk7_fcenter')
        freq_labels = f.varget('fbk7_labl_fcenter')

    # Time in hours used for plotting.
    t = np.arange(avg.shape[0]) * 1./8. / 3600.0

    # Cut out range shown in Tyler 2019.
    icut = (t >= 0.0) & (t <= 4.0)
    t = t[icut]
    peak = peak[icut]
    avg = avg[icut]

    # icut = (t >= 0.0) & (t <= 4.0)
    # t = t[icut]
    # peak = peak[icut]
    # avg = avg[icut]

    # -------------------------------------------------------------
    # Interpolate to find the frequencies.

    test_method2 = True
    if test_method2:
        noise_level = 0.5

        print('Finding frequencies method2')
        f_calc = InterpFilterBank2d()
        res2 = f_calc.fit(peak, avg=avg, m_ctr_min=3.0, noise_level=noise_level,
                          bin_minimum=3, use_avg=0.2, include_alone=True,
                          debug_plots=False)

        # -----------------------------------------------------------------------
        # Make a plot of the frequencies.

        print('Making a plot of the frequencies and amplitudes.')

        # Find the frequency for the maximum values of the gain curves.
        i_gain_max = np.nanargmax(f_calc.gain_norm, axis=1)
        freq_gain_max = f_calc.freq[i_gain_max]

        fig, axs = plt.subplots(6, 1, sharex=True)
        fig.set_tight_layout({'h_pad': 0.05})
        fig.set_size_inches([6.4, 10.43])

        axs[-1].set_xlabel('Time (Hours)')
        axs[0].set_title(filename)

        ax = axs[0]
        ax.semilogy(t, res2['real_freq'], '.', label='method2')
        if 'freq_noise' in res2.dtype.names:
            ax.semilogy(t, res2['freq_noise'], '.', label='freq_noise')
        if 'alone' in f_calc.check.dtype.names:
            ax.semilogy(t[f_calc.check['alone']], res2['real_freq'][f_calc.check[
                'alone']], '.', label='alone')
        for freq_gain_max1 in freq_gain_max:
            ax.semilogy([t.min(), t.max()], [freq_gain_max1]*2, 'k')
        ax.grid(True)
        ax.set_ylabel('Frequency (Hz)')
        ax.legend()

        ax = axs[1]
        if 'alone' in f_calc.check.dtype.names:
            ax.plot(t[f_calc.check['alone']], f_calc.bins['m_ctr'][f_calc.check[
                'alone']], '.', label='ctr alone')
        ax.plot(t[f_calc.check['good_amplitude']], f_calc.bins['m_ctr'][
            f_calc.check['good_amplitude']], '.', label='ctr good_amplitude')
        ax.plot(t[f_calc.check['good']], f_calc.bins['m_ctr'][f_calc.check[
            'good']], '.', label='ctr good')

        ax.plot(t, f_calc.bins['m_adj'], '.', label='adj')
        ax.plot(t, res2['real_amp'], '.', label='method2 good_amplitude')
        ax.plot(t[f_calc.check['good']], res2['real_amp'][f_calc.check['good']],
                '.k', label='method2 good')
        ax.plot([t.min(), t.max()], [noise_level]*2, label='noise level')
        ax.grid(True)
        ax.set_ylabel('Amplitude (mV/m)')
        ax.legend(fontsize=8)

        ax = axs[2]
        ax.plot(t, res2['near_limit'], '.')
        ax.grid(True)
        ax.set_ylabel('gain_adj/gain_ctr')

        ax = axs[3]
        ax.plot(t[f_calc.check['good']], f_calc.bins['i_ctr'][f_calc.check[
            'good']], '.', label='i_ctr')
        ax.plot(t[f_calc.check['good']], f_calc.bins['i_adj'][f_calc.check[
            'good']], '.', label='i_adj')
        ax.grid(True)
        ax.legend()
        ax.set_ylabel('index')

        ax = axs[4]
        ax.plot(t[f_calc.check['good_amplitude']], f_calc.bins['arv_ctr'][
            f_calc.check['good_amplitude']], '.', label='ctr good amplitudes')
        ax.plot(t[f_calc.check['good']], f_calc.bins['arv_ctr'][f_calc.check[
            'good']], '.', label='ctr good shape')
        ax.grid(True)
        ax.legend()
        ax.set_ylabel('ARV')

        ax = axs[5]
        ax.plot(t[f_calc.check['good_amplitude']], f_calc.bins['arv_metric'][
            f_calc.check['good_amplitude']], '.', label='ctr good amplitudes')
        ax.plot(t[f_calc.check['good']], f_calc.bins['arv_metric'][f_calc.check[
            'good']], '.', label='ctr good shape')
        ax.grid(True)
        ax.legend()
        ax.set_ylabel('ARV metric')

    # ------------------------------------------------------------------
    # Plot debug

    plot_avg = False
    if plot_avg:

        print('Plots using avarage.')
        n1 = 5
        n2 = 6

        fig, axs = plt.subplots(7, 1, sharex=True)
        axs[0].set_title(f'{n1}')
        ax = axs[0]
        ax.plot(t, peak[:, n1], '.-')
        ax.set_ylabel(f'peak {n1}')
        ax.grid(True)
        ax = axs[2]
        ax.plot(t, avg[:, n1], '.-')
        ax.set_ylabel(f'avg {n1}')
        ax.grid(True)
        ax = axs[4]
        ax.plot(t, avg[:, n1] / peak[:, n1], '.-')
        ax.set_ylabel(f'ARV {n1}')
        ax.grid(True)


        axs[0].set_title(f'{n2}')
        ax = axs[1]
        ax.plot(t, peak[:, n2], '.-')
        ax.set_ylabel(f'peak {n2}')
        ax.grid(True)
        ax = axs[3]
        ax.plot(t, avg[:, n2], '.-')
        ax.set_ylabel(f'avg {n2}')
        ax.grid(True)
        ax = axs[5]
        ax.plot(t, avg[:, n2] / peak[:, n2], '.-')
        ax.set_ylabel(f'ARV {n2}')
        ax.grid(True)

        axs[-1].plot(t, (avg[:, n1] / peak[:, n1]) / (avg[:, n2] / peak[:, n2]))
        axs[-1].grid(True)


