;+
;*****************************************************************************************
;
;  PROCEDURE :  rbsp_efw_fbk_freq_interpolate  (see rbsp_efw_fbk_freq_interpolate_test.pro)
;
;  PURPOSE  : Takes advantage of the EFW filterbank overlapping gain curve characteristics
;			  to correct the frequency and amplitude of
;			  the RBSP filterbank data
;                         assuming narrowband waves. Returns
;			  tplot variables with the corrected
;			  frequencies and amplitudes. Assumes that the
;			  FBK data have already been calibrated
;			  frequency by frequency. Thus, each gain
;			  curve is normalized to unity.
;
;
; KEYWORDS:               testing -> makes various plots for each data point

;                         scale_fac_lim -> Set to ensure that the
;			  correction is applied only to narrowband
;			  waves. This is defined as val_adjacent/mvpk, the
;			  amplitude of the narrowband signal in the
;			  adjacent FBK bin divided by the amplitude in
;			  the peak FBK bin. Assuming a perfectly
;			  narrowband signal of a freq that matches the
;			  gaincurve peak in FBK bin "i", this will be
;			  seen with ~30 dB of attenuation in the
;			  adjacent bin. This corresponds to a signal
;			  that is 0.215 times the amplitude in the
;			  peak bin. This represents the extreme low
;			  value of scale_fac_lim. The default is set
;			  to 0.6 for FBK13 and 0.1 for FBK7. **WARNING**: setting this factor too
;			  high results in occasional unrealistic FBK
;			  amplitudes b/c the amplitude adjustment can
;			  be very large. This can often happen when
;			  observed chorus has both upper and lower
;			  bands. Say the peak FBK bin at time "t"
;			  corresponds to lower band chorus. The
;			  adjacent (higher) FBK bin can then show the
;			  upper band chorus, which can be nearly the
;			  same amplitude. This will cause the program
;			  (under the narrowband assumption) to think
;                         that the freq and amplitude need to be
;                         significantly adjusted. Large amplitude
;                         adjustments can be avoided by setting
;                         maxamp_lim, which defaults to 2.
;
;                         minamp -> FBK amps below this value
;                         won't be interpolated. I find that
;                         interpolation doesn't work for amps < 2 mV/m.
;
;                         maxamp_lim -> the freq interpolation does
;                         really well with larger values of
;                         scale_fac_lim (up to 0.99). However, the amplitude
;                         adjustment is often a bit more sketchy. Set
;                         this value to limit how much amplitude boost
;                         the program gives. Defaults to 2.
;
;		 	  Ex: info = {probe:'a',fbk_mode:'7',fbk_type:'Ew'}
;           rbsp_efw_fbk_freq_interpolate,'rbspb_efw_fbk_13_fb1_pk',info,/noplot
;					where the tplot variable is the [n,13] filterbank data. Can also be
;					used with FBK7
;
;
;			  Returns the following 1-D tplot variables
;				fbk_maxamp_orig -> the max FBK amp of all 13(7) input freq bins for each time for input data
;				fbk_binnumber_of_max_orig -> the bin number [0-12] for FBK 13 or [0-6] for FBK7 that this max value occurs in
;				fbk_maxamp_adj -> the adjusted max FBK amp. This is an attempt to correct
;						the amp b/c of its
;						undervaluation due to
;						the gain curve of each
;						FBK channel.
;						i.e. assume that the FBK data are previously calibrated. In this case the
;						gain curve for each FBK bin is normalized. The input amplitude is then scaled
;						by  amp_new = amp_input * (1/G(f)), where G(f) is the value of the gain
;						at the interpolated freq.
;				fbk_freq_of_max_orig -> the freq of
;				the max value of the input data. When
;				the values aren't adjusted by
;				comparing gain curves (for e.g., if
;				the input amplitude is too small or
;				the peak occurs in the first or last
;				FBK bin, then the frequency is set to
;				the frequency of the peak of the
;				gaincurve in that particular channel ("fbins_maxgain")
;				fbk_freq_of_max_adj -> the adjusted freq of the max value
;
;
;
;
;			  Note: this routine assumes a narrowband wave. Say the max value of a
;			  chorus wave is seen in bin 10 from FBK13 (400-800 Hz).
;			  The innate freq resolution is poor for the FBK product. However, the gain
;			  curves for adjacent bins overlap. So, if you
;			  know the amplitudes in the adjacent bins (bin 9 and bin 11) then you can
;			  use the gain curves to determine how far in frequency you need to slide
;			  to make the amplitude in bin 9 (say) equal that in bin 10. This is the
;			  actual
;			  wave
;			  frequency. To see how well the interpolation works see rbsp_efw_fbk_freq_interpolate_test.pro
;
;  REQUIRES:  FBK gain curves (supplied by David Malaspina from LASP)
;
;

;Here I plot the low and high freq bin limits for each FBK channel as
;well as the location in frequency of the peak of each
;channel's gain curve. Note that the last of freqmax has been set
;to the channel's center value.


;; for i=0,12 do print,i+1, fcals.cal_fbk.FREQ_FBK13L[i], '  ',fcals.cal_fbk.freq_fbk13C[i],'   ',fbins_maxgain[i], '  ',fcals.cal_fbk.FREQ_FBK13H[i]

;; FBK bin  freqlow         center      freq_peakofgaincurve  freqhigh
;;  1     0.800000        1.15000         1.36000        1.50000
;;  2      1.50000        2.25000         2.62000        3.00000
;;  3      3.00000        4.50000         5.14000        6.00000
;;  4      6.00000        9.00000         10.0000        12.0000
;;  5      12.0000        18.5000         20.8000        25.0000
;;  6      25.0000        37.5000         40.6000        50.0000
;;  7      50.0000        75.0000         83.8000        100.000
;;  8      100.000        150.000         172.000        200.000
;;  9      200.000        300.000         334.000        400.000
;; 10      400.000        600.000         658.000        800.000
;; 11      800.000        1200.00         1360.00        1600.00
;; 12      1600.00        2400.00         2800.00        3200.00
;; 13      3200.00        4850.00         4850.00        6500.00


;
; 			  bin -> the bin number of the peak FBK value for FBK 7
;					bin        0     0.800000      1.50000 Hz
;					bin        1      3.00000      6.00000 Hz
;					bin        2      12.0000      25.0000 Hz
;					bin        3      50.0000      100.000 Hz
;					bin        4      200.000      400.000 Hz
;					bin        5      800.000      1600.00 Hz
;					bin        6      3200.00      6500.00 Hz
;
;
;
;
;   NOTES:    Use with caution. The accuracy of the final frequency value depends on the
;			  nature of the signal being measured. For ex, very broadband noise won't lead
;			  to a very good freq determination.
;
;			  At the moment the program only uses the unity gain curves.
;			  These are applicable for DC-coupled Ew channels like E12DC which is the default
;			  FBK channel - but this may change.
;
;
;
;   CREATED:  03/07/2013
;   CREATED BY:  Aaron W. Breneman
;    LAST MODIFIED:  10/26/2013   v1.0.1
;						Major modifications
;    MODIFIED BY: AWB
;
;*****************************************************************************************
;-
;**************************************************************************


pro rbsp_efw_fbk_freq_interpolate,tname,info,$
                                  scale_fac_lim=scale_factor_limit,minamp=minamp,$
                                  maxamp_lim=maxamp_lim,$
                                  testing=testing

  rbspx = 'rbsp' + info.probe

  get_data,tname,data=pk

  if info.fbk_mode eq '13' then maxbin = 12 else maxbin = 6

  amp_original = fltarr(n_elements(pk.x))             ;;peak amplitude for each time
  amp_corrected = fltarr(n_elements(pk.x))            ;;corrected amplitudes for each time

  freqbin_orig = fltarr(n_elements(pk.x))	        ;;original frequencies for each time
  freq_corrected = fltarr(n_elements(pk.x))             ;;corrected frequencies for each time



  ;;Grab the gain curves
;  restore,'~/Desktop/code/Aaron/datafiles/rbsp/FBK_gain_curves/RBSP_FilterBank_Theoretical_Rsponse_wE12ACmeasuredResponse_DMM_20130305.sav'
  restore,'~/Desktop/code/Aaron/github.umn.edu/filterbank_frequency_enhancer/RBSP_FilterBank_Theoretical_Rsponse_wE12ACmeasuredResponse_DMM_20130305.sav
;~/Desktop/code/Aaron/github.umn.edu/filterbank_frequency_enhancer/RBSP_FilterBank_Theoretical_Rsponse_SCMw_measuredResponse_DMM_20131030.sav

  freqs_for_gaincurves = freq



  ;;Set default values
  if ~keyword_set(scale_factor_limit) then begin
     if info.fbk_mode eq '13' then scale_factor_limit = 0.6
     if info.fbk_mode eq '7'  then scale_factor_limit = 0.1
  endif
  if ~keyword_set(minamp) then begin
     if info.fbk_type eq 'Ew' then minamp = 2       ;mV/m
     if info.fbk_type eq 'Bw' then minamp = 5/1000. ;nT
  endif
  if ~keyword_set(maxamp_lim) then maxamp_lim = 2.


  ;;These data are for the FBK Ew data channels which have unity gain
  ;;in all bins except for the last one.
  gaincurve = FB_THEORETICAL_GAINRESPONSE_UNITYGAIN10KHZ
  if info.fbk_mode eq '13' then gaincurve = [[gaincurve[*,12]],[gaincurve[*,11]],[gaincurve[*,10]],[gaincurve[*,9]],$
                                             [gaincurve[*,8]],[gaincurve[*,7]],[gaincurve[*,6]],[gaincurve[*,5]],[gaincurve[*,4]],$
                                             [gaincurve[*,3]],[gaincurve[*,2]],[gaincurve[*,1]],[gaincurve[*,0]]]
  if info.fbk_mode eq '7'  then gaincurve = [[gaincurve[*,12]],[gaincurve[*,10]],[gaincurve[*,8]],[gaincurve[*,6]],$
                                             [gaincurve[*,4]],[gaincurve[*,2]],[gaincurve[*,0]]]




  ;;Normalize the gain curves. I'll use these b/c I'm assuming that
  ;;the FBK data is already
  ;;calibrated. However, note that
  ;;each gain curve is somewhat
  ;;offset from the others (most
  ;;notably the highest one which is
  ;;about 2.1x larger than the
  ;;lowest). If I don't take
  ;;this into account than the
  ;;interpolation doesn't
  ;;work as well. However, my
  ;;testing shows that the results
  ;;are more accurate if I only
  ;;modify the last bin.
  gaincurve_norm = gaincurve

  ;;normalization factor (ranges
  ;;from 1 to 2.1)
  if info.fbk_mode eq '13' then gaincurve_maxv = fltarr(13) else gaincurve_maxv = fltarr(7)
  for i=0,n_elements(gaincurve_maxv)-1 do gaincurve_maxv[i] = max(gaincurve[*,i])
  gaincurve_normfactor = gaincurve_maxv/min(gaincurve_maxv)

  ;;normalize each curve to unity
  for i=0,n_elements(gaincurve[0,*])-1 do gaincurve_norm[*,i] = gaincurve[*,i]/max(gaincurve[*,i])

  ;;MODIFY ALL BINS? NO, RESULTS NOT
  ;;AS ACCURATE AS ONLY MODIFYING
  ;;THE LAST ONE (SEE RBSP_EFW_FBK_FREQ_INTERPOLATE_TEST.PRO)
  ;; for i=0,n_elements(gaincurve[0,*])-1 do gaincurve_norm[*,i] = gaincurve[*,i]*gaincurve_normfactor[i]

  ;;Only modify the last bin
  if info.fbk_mode eq '13' then gaincurve_norm[*,12] = gaincurve[*,12]*gaincurve_normfactor[12]
  if info.fbk_mode eq '7' then gaincurve_norm[*,6] = gaincurve[*,6]*gaincurve_normfactor[6]






  ;;Get the filterbank freq bin definitions
  fcals = rbsp_efw_get_gain_results()


  ;; ;Get the gain vs freq (for center of freq bins)
  ;; fgain_vs_freq = fcals.cal_fbk.e12dc_a_fbk13.gain_vs_freq


  ;;Find the freq corresponding to the peak of each gain curve
  if info.fbk_mode eq '13' then freq_peak_for_each_gaincurve = replicate(0.,n_elements(fcals.cal_fbk.FREQ_FBK13H))
  if info.fbk_mode eq '7'  then freq_peak_for_each_gaincurve = replicate(0.,n_elements(fcals.cal_fbk.FREQ_FBK7H))




;--------------------------------------------------
;increase the resolution of the gain curves
;--------------------------------------------------

  range1 = [0,50,100,150,200]
  range2 = [50,100,150,200,249]

  nelem = 500.
  freq2 = 0.
  gc2 = 0.
  for bb=0,n_elements(range1)-1 do begin                         ;$
     freqsub = freq[range1[bb]:range2[bb]]                       ;& $
     slope = (freqsub[n_elements(freqsub)-1]-freqsub[0])         ;& $
     freqsub2 = slope*indgen(nelem)/(nelem-1) + freq[range1[bb]] ;& $
     freq2 = [freq2,freqsub2]
  endfor


  freq2 = freq2[1:n_elements(freq2)-1]
  ;; plot,freq2,psym=4,/xstyle



  ;; Now interpolate the gain curves to the new frequencies
  gc2 = fltarr(nelem*n_elements(range1),n_elements(gaincurve_norm[0,*]))
  for bb=0,n_elements(gaincurve_norm[0,*])-1 do gc2[*,bb] = interpol(gaincurve_norm[*,bb],freq,freq2,/spline)


  ;; plot,  freq2 ,gc2[*,0],/xlog,/ylog,/ys,yrange=[0.001,10], /xs,xrange = [0.1, 10000], $
  ;;        background = 255, color = 254 , thick = 2, xtitle = 'Hz', ytitle = 'dB', $
  ;;        title = 'Filter Bank Unity response curves...these are the ones used in this program',$
  ;;        psym=4
  ;; oplot,  freq ,gaincurve_norm[*,0] ,color=0,psym=5


  ;; for kk = 1, n_elements(freq_peak_for_each_gaincurve)-1, 1 do oplot, freq2, gc2[*,kk],$
  ;;    color = kk*20 + 20, thick = 2,psym=4

  ;; for kk = 1, n_elements(freq_peak_for_each_gaincurve)-1, 1 do oplot, freq, gaincurve_norm[*,kk],$
  ;;    color = 0, thick = 1,psym=5

  ;; stop

  gaincurve_norm = gc2
  gaincurve_dB = 20d*alog10(gaincurve_norm)
  freqs_for_gaincurves = freq2

  for i=0,n_elements(gaincurve_dB[0,*])-1 do begin
     goo = max(gaincurve_dB[*,i],wh)
     freq_peak_for_each_gaincurve[i] = freqs_for_gaincurves[wh]
  endfor




  ;;--------------------------------------------------
  ;;For each filterbank data point...
  ;;--------------------------------------------------

  ;;Array that indicates whether any amplitude/freq adjustment
  ;;has occurred
  bin_shift = replicate(3,n_elements(pk.x))

  ;;Various arrays to indicate when no interpolation has occurred due
  ;;to various conditions
  nointerp_smallamp = replicate(0,n_elements(pk.x))
  nointerp_edgebins = replicate(0,n_elements(pk.x))
  nointerp_sf_exceeded = replicate(0,n_elements(pk.x))
  nointerp_neighbor_toosmall = replicate(0,n_elements(pk.x))
  amp_interp_limited_to_maxamp_lim = replicate(0,n_elements(pk.x))


  for i=0L,n_elements(pk.x)-1 do begin
     if i/1000 - floor(i mod 1000) eq 1 then print,'i=',i

     ;;Find what freq bin the max value occurs in
     mvpk = max(pk.y[i,*],whpk,/nan)


     ;;Find amplitude values in adjacent bins (L=low, H=high)
     if (whpk-1) ge 0      then mvLpk = pk.y[i,whpk-1] else mvLpk = 0
     if (whpk+1) le maxbin then mvHpk = pk.y[i,whpk+1] else mvHpk = 0


     ;;--------------------------------------------------
     ;;Conditions under which we won't do the interpolation
     ;;--------------------------------------------------

     ;;***1) If either of the adjacent bins is zero then we won't
     ;;do the interpolation. This condition can occur, for example,
     ;;when values below 0.1*fce_eq are set to zero. In this case the
     ;;freq shift can only happen upwards, which can give the wrong
     ;;answer. This is the same as having an edge bin
     if mvLpk eq 0 or mvHpk eq 0 then bin_shift[i] = 0
     if mvLpk eq 0 or mvHpk eq 0 then nointerp_edgebins[i] = 1
     ;;***2) If max value is in lowest or highest bin then I won't do the freq and amp interpolation
     if whpk eq maxbin then bin_shift[i] = 0
     if whpk eq 0 then bin_shift[i] = 0
     if whpk eq maxbin then nointerp_edgebins[i] = 1
     if whpk eq 0 then nointerp_edgebins[i] = 1
     ;;***3) If max value is too low then
     ;;I won't do the interpolation b/c it doesn't work
     ;;well (you're too far down in the gain curve
     ;;for meaningful interpolation)
     if mvpk le minamp then bin_shift[i] = 0
     if mvpk le minamp then nointerp_smallamp[i] = 1


     ;;--------------------------------------------------


     ;;If max value isn't in lowest or highest bin then I'll find out which adjacent bin has
     ;;the larger amplitude
     if bin_shift[i] ne 0 then begin
        if mvLpk ge mvHpk then bin_shift[i] = -1 else bin_shift[i] = 1
     endif

     if bin_shift[i] eq -1 then val_adjacent = mvLpk
     if bin_shift[i] eq 1  then val_adjacent = mvHpk
     if bin_shift[i] eq 0  then val_adjacent = mvpk



     ;;***4) if the neighboring bin value is at noise level then we won't try to interpolate freq or amp
     if info.fbk_type eq 'Ew' then noiselevel = 0.1     ;mV/m
     if info.fbk_type eq 'Bw' then noiselevel = 1/1000. ;nT

     if val_adjacent le noiselevel then begin
        val_adjacent = pk.y[i,whpk]
        bin_shift[i] = 0
        nointerp_neighbor_toosmall[i] = 1
     endif

     amp_ratio = val_adjacent/mvpk


     ;;***5) The amplitude scaling is too large. This is a violation
     ;;of the narrowband assumption.
     if amp_ratio gt float(scale_factor_limit) then begin
        scale_factor = 1.
        bin_shift[i] = 0
        nointerp_sf_exceeded[i] = 1
     endif


     ;;Find accurate center freq of the max FBK bin
     boo = max(gaincurve_dB[*,whpk],loc1)


     ;;Number of dB of reduction that adjacent FBK bin sees the signal
     dB = 20*alog10(val_adjacent/mvpk)



     ;;Find the actual freq that a narrowband signal would be to
     ;;appear in the adjacent FBK bin with the amplitude that it does
     goo = where(gaincurve_dB[*,whpk+bin_shift[i]] le dB)
     boo = max(gaincurve_dB[*,whpk+bin_shift[i]],loc2)




     if goo[0] ne -1 then begin
        ;;Adjust frequency upwards
        if bin_shift[i] eq 1 then begin
           locf = goo[where(goo le loc2)]
           freq_corrected[i] = freqs_for_gaincurves[max(locf)]
        endif
        ;;Adjust frequency downwards
        if bin_shift[i] eq -1 then begin
           locf = goo[where(goo ge loc2)]
           freq_corrected[i] = freqs_for_gaincurves[locf[0]]
        endif
     endif
     ;;Don't adjust frequency
     if bin_shift[i] eq 0 or goo[0] eq -1 then begin
        freq_corrected[i] = freq_peak_for_each_gaincurve[whpk]
     endif


     freqbin_orig[i] = whpk



     ;;-----------------------------------------------------
     ;;Now we'll adjust the amp based on the corrected freq.
     ;;--------------------------------------------------

     ;;Find the dB difference in the main gain curve (whpk)
     ;;b/t the center location and the location of the adjusted peak

     ;; freq_peak_for_each_gaincurve[whpk+bin_shift[i]]


     if bin_shift[i] eq 1 then begin
        goo = where(freqs_for_gaincurves le freq_corrected[i])
        dB_new = gaincurve_dB[goo[n_elements(goo)-1],whpk]
     endif
     if bin_shift[i] eq -1 then begin
        goo = where(freqs_for_gaincurves ge freq_corrected[i])
        dB_new = gaincurve_dB[goo[0],whpk]
     endif
     if bin_shift[i] eq 0 then dB_new = 0


     scale_factor = 1/10^(dB_new/20.)


     if scale_factor gt maxamp_lim then begin
        scale_factor = maxamp_lim
        amp_interp_limited_to_maxamp_lim[i] = 1
     endif


     amp_corrected[i] = scale_factor*mvpk
     amp_original[i] = mvpk







;--------------------------------------------------
;MASTER PLOT
;--------------------------------------------------



     ;; if 1/10^(dB/20.) ge 5 and freq_corrected[i] ge 500. then begin


     if keyword_set(testing) and bin_shift[i] ne 0 then begin
        !p.multi = [0,0,2]


                                ;First panel is in dB space

        ;;main gain curve
        plot,freqs_for_gaincurves,gaincurve_dB[*,whpk],$
             ytitle='amplitude in dB',$
             xtitle='freq (Hz)',$
             title='FBK freq and amplitude enhancement process',$
             yrange = [-60,20], /xlog,/xs,xrange = [100, 1d4],/ystyle,thick=2
        ;;adjacent gain curve
        oplot,freqs_for_gaincurves,gaincurve_dB[*,whpk+bin_shift[i]],color=50,thick=2

        ;;original frequency and amplitude
        oplot,[freq_peak_for_each_gaincurve[whpk],freq_peak_for_each_gaincurve[whpk]],$
              [gaincurve_dB[loc1,whpk],gaincurve_dB[loc1,whpk]],psym=4

        ;;amplitude as observed on adjacent bin
        oplot,[freq_peak_for_each_gaincurve[whpk+bin_shift[i]],freq_peak_for_each_gaincurve[whpk+bin_shift[i]]],$
              [dB,dB],psym=4,color=50

        ;;horizontal line from adjacent amplitude to gain curve
        oplot,[freq_peak_for_each_gaincurve[whpk+bin_shift[i]],freq_corrected[i]],$
              [dB,dB],color=50

        ;;horizontal line from peak of main curve to adjusted freq
        oplot,[freq_peak_for_each_gaincurve[whpk],freq_corrected[i]],$
              [0,0],color=50



        ;;vertical line showing actual frequency
        oplot,[freq_corrected[i],freq_corrected[i]],$
              [-1000,30],color=50,linestyle=2

        ;;vertical line from adjacent amplitude to peak of adjacent
        ;;bin. This shows how I determine the corrected amplitude
        oplot,[freq_corrected[i],freq_corrected[i]],$
              [0,-dB_new],color=254
        oplot,[freq_corrected[i],freq_corrected[i]],$
              [0,dB_new],color=200

        ;;horizontal line showing actual amplitude
        oplot,[0.0001,1d5],$
              -1*[dB_new,dB_new],color=50,linestyle=2

        ;;Adjusted frequency and amplitude
        oplot,[freq_corrected[i],freq_corrected[i]],$
              -1*[dB_new,dB_new],color=50,linestyle=2,psym=4,symsize=2



;--------------------------------------------------

        ;;Second panel shows fractional value change of amplitude

        ;;main gain curve
        plot,freqs_for_gaincurves,gaincurve_norm[*,whpk],$
             ytitle='amplitude (normalized to unity)',$
             xtitle='freq (Hz)',$
             title='from rbsp_efw_fbk_freq_interpolate.pro',$
             yrange = [0.001,20], /xlog,/xs,xrange = [100, 1d4],/ystyle,thick=2,/ylog
        ;;adjacent gain curve
        oplot,freqs_for_gaincurves,gaincurve_norm[*,whpk+bin_shift[i]],color=50,thick=2

        ;;original frequency and amplitude
        oplot,[freq_peak_for_each_gaincurve[whpk],freq_peak_for_each_gaincurve[whpk]],$
              [gaincurve_norm[loc1,whpk],gaincurve_norm[loc1,whpk]],psym=4

        ;;amplitude as observed on adjacent bin
        oplot,[freq_peak_for_each_gaincurve[whpk+bin_shift[i]],freq_peak_for_each_gaincurve[whpk+bin_shift[i]]],$
              [10^(dB/20.),10^(dB/20.)],psym=4,color=50

        ;;horizontal line from adjacent amplitude to gain curve
        oplot,[freq_peak_for_each_gaincurve[whpk+bin_shift[i]],freq_corrected[i]],$
              [10^(dB/20.),10^(dB/20.)],color=50

        ;;horizontal line from peak of main curve to adjusted freq
        oplot,[freq_peak_for_each_gaincurve[whpk],freq_corrected[i]],$
              [1,1],color=50

        ;;vertical line showing actual frequency
        oplot,[freq_corrected[i],freq_corrected[i]],$
              [0.0001,1000],color=50,linestyle=2

        ;;vertical line from adjacent amplitude to peak of adjacent bin
        oplot,[freq_corrected[i],freq_corrected[i]],$
              [1,10^(dB_new/20.)],color=200
        oplot,[freq_corrected[i],freq_corrected[i]],$
              [1,1/10^(dB_new/20.)],color=254

        ;;horizontal line showing actual amplitude
        oplot,[0.0001,1d5],$
              [1/10^(dB_new/20.),1/10^(dB_new/20.)],color=50,linestyle=2

        ;;Adjusted frequency and amplitude
        oplot,[freq_corrected[i],freq_corrected[i]],$
              [1/10^(dB_new/20.),1/10^(dB_new/20.)],color=50,linestyle=2,psym=4,symsize=2


        !p.charsize = 1.4

        xyouts,0.1,0.9,'Black curve = original gain curve',/normal
        xyouts,0.1,0.88,'Blue curve =  adjacent gain curve',/normal,color=50
        xyouts,0.1,0.86,'Black diamond = amplitude in peak FBK bin',/normal
        xyouts,0.1,0.84,'Blue diamond = amplitude in adjacent FBK bin',/normal,color=50

        xyouts,0.1,0.5,time_string(pk.x[i],tformat="YYYY-MM-DD/hh:mm:ss.fff"),/normal,color=250
        xyouts,0.1,0.4,'Observed amplitude (' + $
               strtrim(val_adjacent,2) + ') in adjacent FBK bin is ' + $
               strtrim(floor(100*val_adjacent/mvpk),2) + $
               '% !Cof observed amplitude in peak FBK bin ('+strtrim(mvpk,2)+')!C'+ $
               'Adjusted amplitude is '+strtrim(amp_corrected[i],2)+', !C'+$
               strtrim(amp_corrected[i]/mvpk,2)+' times larger',/normal
        xyouts,0.1,0.3,'Observed freq is ' + $
               strtrim(floor(freq_peak_for_each_gaincurve[whpk]),2)+' Hz' + $
               '!Cand adjusted freq is '+strtrim(floor(freq_corrected[i]),2) + ' Hz',/normal


        stop
     endif

  endfor





;create the array with the actual original frequencies
  freqs_orig = replicate(0.,n_elements(pk.x))
  for bb=0L,n_elements(pk.x)-1 do freqs_orig[bb] = freq_peak_for_each_gaincurve[freqbin_orig[bb]]


  store_data,rbspx+'_flag_freq_amp_adjusted',data={x:pk.x,y:bin_shift}
  store_data,rbspx+'_flag_nointerp_smallamp',data={x:pk.x,y:nointerp_smallamp}
  store_data,rbspx+'_flag_nointerp_edgebins',data={x:pk.x,y:nointerp_edgebins}
  store_data,rbspx+'_flag_nointerp_scalefactor_toolarge',data={x:pk.x,y:nointerp_sf_exceeded}
  store_data,rbspx+'_flag_nointerp_neighbor_toosmall',data={x:pk.x,y:nointerp_neighbor_toosmall}
  store_data,rbspx+'_flag_amp_interp_limited_to_maxamp_lim',data={x:pk.x,y:amp_interp_limited_to_maxamp_lim}


  store_data,rbspx+'_fbk_maxamp_orig',data={x:pk.x,y:amp_original}
  store_data,rbspx+'_fbk_maxamp_adj',data={x:pk.x,y:amp_corrected}
  store_data,rbspx+'_fbk_binnumber_of_max_orig',data={x:pk.x,y:freqbin_orig}
  store_data,rbspx+'_fbk_freq_of_max_orig',data={x:pk.x,y:freqs_orig}
  store_data,rbspx+'_fbk_freq_of_max_adj',data={x:pk.x,y:freq_corrected}





;Plot the gain curves

  if keyword_set(testing) and n_elements(freq_peak_for_each_gaincurve) eq 13 then begin
     window, 0

     plot,  freq ,gaincurve_norm[*,0] , yrange = [0,1], /xlog, /xs,xrange = [0.1, 1d4], /ystyle, $
            background = 255, color = 0 , thick = 2, xtitle = 'Hz', ytitle = 'dB', $
            title = 'Filter Bank Unity response curves...these are the ones used in this program'

     for kk = 1, n_elements(freq_peak_for_each_gaincurve)-1, 1 do oplot, freq, gaincurve_norm[*,kk],$
        color = kk*20 + 20, thick = 2


     window, 1
     plot,  freq , 20d * alog10(FB_theoretical_Gainresponse_UnityGain10kHz[*,0] ), $
            yrange = [-70,1], /xlog, /xs,xrange = [0.1, 1d4], /ystyle, $
            background = 255, color = 0 , thick = 2, $
            xtitle = 'Hz', $
            ytitle = 'dB', $
            title = 'Filter Banks Theoretical Response (Unity gain at 10kHz)'

     for kk = 1, n_elements(freq_peak_for_each_gaincurve)-1, 1 do oplot, freq, 20d * alog10(FB_theoretical_Gainresponse_UnityGain10kHz[*,kk]),$
        color = kk*20 + 20, thick = 2

     stop

     window, 2
     plot,  freq , 20d * alog10(FB_theoretical_Gainresponse_E12ACmeasuredResponseConvolved[*,0]),$
            yrange = [-70,1], /xlog, /xs,xrange = [0.1, 1d4], /ystyle, $
            background = 255, color = 0 , thick = 2, $
            xtitle = 'Hz', $
            ytitle = 'dB', $
            title = 'Filter Banks Theoretical Response, E12AC measured response convovled '

     for kk = 1, n_elements(freq_peak_for_each_gaincurve)-1, 1 do $
        oplot, freq, 20d * alog10(FB_theoretical_Gainresponse_E12ACmeasuredResponseConvolved[*,kk]), $
               color = kk*20 + 20, thick = 2

     stop
  endif



end
