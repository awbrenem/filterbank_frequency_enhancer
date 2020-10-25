;Crib sheet for using the filterbank frequency interpolation routine and also testing
;the results against burst and spectral data

;      WARNING: don't detrend the interpolated freq data. If
;there are multiple bands of waves occurring then the detrended
;version will split the difference b/t them, leading to a false frequency


;Written by Aaron W Breneman


;Days with EB1 and chorus to test (**NEED TO FIND A DAY WITH SOME
;LOWER FREQ CHORUS SO THAT LOWER CADENCE BURST CAN RESOLVE IT***)
;2013-01-14  293MB  (***excellent low freq chorus***)
;2013-01-20  445MB
;2013-01-21  262MB
;2013-01-26  1.3GB
;2013-01-27  2.0GB


;probe can be 'a' or 'b'
;fbk_mode can be '13' or '7'
;fbk_type can be 'Ew' or 'Bw'
probe = 'b'
fbk_mode = '7'
fbk_type = 'Ew'
info = {probe:probe,$
		fbk_mode:fbk_mode,$
		fbk_type:fbk_type}


tplot_options,'title','from rbsp_efw_fbk_freq_interpolate_test.pro'
  tplot_options,'xmargin',[20.,16.]
  tplot_options,'ymargin',[3,9]
  tplot_options,'xticklen',0.08
  tplot_options,'yticklen',0.02
  tplot_options,'xthick',2
  tplot_options,'ythick',2
  tplot_options,'labflag',-1



include_burst = 1
rbspx = 'rbsp' + probe
sc = probe
currdate = '2014-04-30'
;;currdate = '2012-10-07'     --> works well
;currdate = '2013-02-05'  ;;Too small amplitude for FBK
;; currdate = '2013-01-26'
;; currdate = '2013-01-27'
;; currdate = '2013-01-14'   ;--> works well
;; currdate = '2012-10-13'
;; currdate = '2013-01-14'   --> works well
;; currdate = '2015-03-01'
;; currdate = '2014-03-13'

timespan,currdate
btype = 'vb2'

if fbk_type eq 'Ew' then stem = 'e12dc'
if fbk_type eq 'Bw' then stem = 'scmw'


rbsp_load_efw_fbk_l2,probe=probe
rbsp_load_efw_spec,probe=probe,type='calibrated'


if keyword_set(include_burst) then begin
   rbsp_load_efw_waveform,type='calibrated',datatype=btype,probe=probe

   tplot_save,'*',filename='/Users/aaronbreneman/Desktop/code/Aaron/RBSP/survey_programs/fbk_interpolate_test/b1_RBSPa_20130114'
endif





;--------------------------------------------------

get_data,'rbsp'+probe+'_efw_fbk'+fbk_mode+'_'+stem+'_pk',data=pk,dlim=dlim,lim=lim

rbsp_efw_position_velocity_crib
;; tinterpol_mxn,'rbspa_state_mlat',pk.x
;; get_data,'rbspa_state_mlat_interp',data=mlats
;; mlats = mlats.y



magcadence = '1sec'
rbsp_load_emfisis,probe=probe,coord='gse',cadence=magcadence,level='l3'


get_data,rbspx+'_emfisis_l3_'+magcadence+'_gse_Magnitude',data=mag
get_data,rbspx+'_emfisis_l3_'+magcadence+'_gse_Mag',data=bfield


;;Interpolate the Bfield data to the cadence of FBK data so that I
;;can sort by f/fce
mag = {x:pk.x,y:interpol(mag.y,mag.x,pk.x)}
bfield2 = [[interpol(bfield.y[*,0],bfield.x,pk.x)],$
           [interpol(bfield.y[*,1],bfield.x,pk.x)],$
           [interpol(bfield.y[*,2],bfield.x,pk.x)]]
fce = 28.*mag.y


;;Project the Bfield to the magnetic eq
;;tinterpol_mxn,rbspx+'_mlat',pk.x,newname='mlat_hires'
tinterpol_mxn,'rbsp'+probe+'_state_mlat',pk.x,newname='mlat_hires'
get_data,'mlat_hires',data=mlat


                                ;EMFISIS fce_eq product (should be more accurate than ECT OP77Q model)
fce_eq = fce*cos(2*mlat.y*!dtor)^3/sqrt(1+3*sin(mlat.y*!dtor)^2)

store_data,'fce',data={x:pk.x,y:fce}
store_data,'fce_eq',data={x:pk.x,y:fce_eq}
store_data,'fce_eq2',data={x:pk.x,y:0.5*fce_eq}
store_data,'fce_eq01',data={x:pk.x,y:0.1*fce_eq}
store_data,'fci',data={x:pk.x,y:fce_eq/1836.}
store_data,'flh',data={x:pk.x,y:sqrt(fce_eq*fce_eq/1836.)}


;--------------------------------------------------






if keyword_set(include_burst) then begin
   get_data,'rbsp'+probe + '_efw_'+btype,data=vb
   e12 = 10.*(vb.y[*,0] - vb.y[*,1])
   store_data,'e12',data={x:vb.x,y:e12}


;; tplot_save,'e12',filename='/Users/aaronbreneman/Desktop/code/Aaron/RBSP/survey_programs/fbk_interpolate_test/b1_RBSPa_20130127_e12'
;; tplot_save,'e12',filename='/Users/aaronbreneman/Desktop/code/Aaron/RBSP/survey_programs/fbk_interpolate_test/b1_RBSPa_20130114_e12'
;; tplot_restore,filenames='/Users/aaronbreneman/Desktop/code/Aaron/RBSP/survey_programs/fbk_interpolate_test/b1_RBSPa_20130126_e12.tplot'
;; tplot_restore,filenames='/Users/aaronbreneman/Desktop/code/Aaron/RBSP/survey_programs/fbk_interpolate_test/b1_RBSPa_20130127_e12.tplot'


;Detrend e12 to remove the DC offset that may be from ground bounce
   rbsp_detrend,'e12',60.*0.02
   copy_data,'e12_detrend','e12'

;; tplot_save,'e12',filename='/Users/aaronbreneman/Desktop/code/Aaron/RBSP/survey_programs/fbk_interpolate_test/b1_RBSPa_20130114_e12'
;; tplot_restore,filenames='/Users/aaronbreneman/Desktop/code/Aaron/RBSP/survey_programs/fbk_interpolate_test/b1_RBSPa_20130114_e12.tplot'


endif
;fn = '/Users/aaronbreneman/Desktop/code/Aaron/github.umn.edu/rbsp-efw-fbk-frequency-enhancer/info.idl'
;fn = '/Users/aaronbreneman/Desktop/code/Aaron/RBSP/survey_programs/runtest_tmp2/info.idl'
;restore,fn
;timesc = time_double(currdate)+info.timesb+info.dt/2.
;str_element,info,'scale_fac_lim',1000,/add_replace
;info.probe = probe

if fbk_mode eq '13' then freq_maxgain = [1.36,2.62,5.14,10,20.8,40.6,83.8,172,334,658,1360,2800,8020]
if fbk_mode eq '7' then freq_maxgain = [1.36,5.14,20.8,83.8,334,1360,8020]

;; ;;--------------------------------------------------
;; ;;eliminate FBK values that are below 0.1*fce_eq

;; ;values from rbsp_efw_fbk_freq_interpolate.pro

;; for qq=0L,n_elements(mlat.x)-1 do begin  $
;;    goo = where(pk.v lt 0.1*fce_eq[qq])  & $
;;    if goo[0] ne -1 then pk.y[qq,goo] = 0


;; store_data,'rbsp'+probe+'_efw_fbk'+fbk_mode+'_'+stem+'_pk',data=pk,dlim=dlim,lim=lim
;; ylim,'rbsp'+probe+'_efw_fbk'+fbk_mode+'_'+stem+'_pk',10,10000,1
;; ;;--------------------------------------------------


;info.fbk_mode = fbk_mode
;info.fbk_type = fbk_type

get_data,rbspx+'_efw_fbk'+fbk_mode+'_'+stem+'_pk',data=dd
if stem eq 'scmw' then store_data,rbspx+'_efw_fbk'+fbk_mode+'_'+stem+'_pk',data={x:dd.x,y:1000.*dd.y,v:dd.v}
if stem eq 'e12dc' then store_data,rbspx+'_efw_fbk'+fbk_mode+'_'+stem+'_pk',data={x:dd.x,y:dd.y,v:dd.v}

;;fbk13
if info.fbk_mode eq '13' then rbsp_efw_fbk_freq_interpolate,$
   rbspx+'_efw_fbk'+fbk_mode+'_'+stem+'_pk',info,$
   scale_fac_lim=0.99,$
   maxamp_lim=20.,$
   minamp=2;,/testing

;;fbk7,Ew
if info.fbk_mode eq '7' and fbk_type eq 'Ew' then rbsp_efw_fbk_freq_interpolate,$
   rbspx+'_efw_fbk'+fbk_mode+'_'+stem+'_pk',info,$
   scale_fac_lim=0.99,$
   maxamp_lim=20.,$
   minamp=2; ,/testing

;;fbk7,Bw
if info.fbk_mode eq '7' and fbk_type eq 'Bw' then rbsp_efw_fbk_freq_interpolate,$
   rbspx+'_efw_fbk'+fbk_mode+'_'+stem+'_pk',info,$
   scale_fac_lim=0.99,$
   maxamp_lim=20,$
   minamp=10.;,/testing



if keyword_set(include_burst) then store_data,'maxamp_comb',data=['e12',rbspx+'_fbk_maxamp_adj',rbspx+'_fbk_maxamp_orig']
if ~keyword_set(include_burst) then store_data,'maxamp_comb',data=[rbspx+'_fbk_maxamp_adj',rbspx+'_fbk_maxamp_orig']
;similar version but without burst data (for saving small eps files)
store_data,'maxamp_comb_v2',data=[rbspx+'_fbk_maxamp_adj',rbspx+'_fbk_maxamp_orig']
dif_data,rbspx+'_fbk_maxamp_adj',rbspx+'_fbk_maxamp_orig',newname='ampdiff'
div_data,rbspx+'_fbk_maxamp_adj',rbspx+'_fbk_maxamp_orig',newname='amp_scalefac'

;; ylim,'amp_scalefac',0,info.scale_fac_lim + 0.5
options,'maxamp_comb','colors',[0,250,50]
options,'maxamp_comb_v2','colors',[250,50]
options,'amp_scalefac','ytitle','FBK!CAdjusted amp/Original amp'
options,'maxamp_comb','ytitle','FBK!COriginal amp(blue)!CAdjusted amp(red)'
options,'maxamp_comb_v2','ytitle','FBK!COriginal amp(blue)!CAdjusted amp(red)'
options,'ampdiff','ytitle','FBK!CAdjusted-Original!Camp'
options,'rbsp'+probe+'_fbk_maxamp_*','thick',1.5
ylim,'rbsp'+probe+'_fbk_maxamp_orig',0,10

;;--------------------------------------------------
;;Check adjusted amplitudes
;;--------------------------------------------------

;; t0 = time_double(currdate + '/02:30:00')
;; t1 = time_double(currdate + '/03:20:00')


;; t0 = time_double(currdate + '/09:55:00')
;; t1 = time_double(currdate + '/10:20:00')

;; timespan,t0,(t1-t0),/seconds





ylim,'maxamp_comb',0,0
ylim,'maxamp_comb_v2',0,0
ylim,'ampdiff',0,0
ylim,'amp_scalefac',0,0
ylim,'rbsp'+probe+'_fbk_maxamp_orig',0,10  ;;zoomed-in view
ylim,'rbsp'+probe+'_adjust_flag',-2,2
ylim,'rbsp'+probe+'_efw_64_spec?',100,7000,0



;Adjust the FBK spec bins so that they are centered on the "center" of
;the freq bin. This allows comparison with the spectral data
get_data,'rbsp'+probe+'_efw_fbk'+fbk_mode+'_'+stem+'_pk',data=fbk
;; fcals = rbsp_efw_get_gain_results()
;; fbinsH = fcals.cal_fbk.FREQ_FBK13H
;; fbk.v = fbinsH
;; fbinsC = fcals.cal_fbk.FREQ_FBK13C
;; fbk.v = fbinsC
fbk.v = freq_maxgain

store_data,'rbsp'+probe+'_efw_fbk'+fbk_mode+'_'+stem+'_pk',data=fbk
ylim,'rbsp'+probe+'_efw_fbk'+fbk_mode+'_'+stem+'_pk',100,7000,0
zlim,'rbsp'+probe+'_efw_fbk'+fbk_mode+'_e12dc_pk',1d-4,100,1
zlim,'rbsp'+probe+'_efw_fbk'+fbk_mode+'_scmw_pk',1d-6,10000,1


ylim,'*flag*',0,1.2
options,'*flag*','panel_size',0.5
options,'*flag*','thick',1.5
options,'rbsp'+probe+'_adjust_flag','panel_size',0.5
options,'rbsp'+probe+'_adjust_flag','thick',1.5
options,'rbsp'+probe+'_adjust_flag','color',20
ylim,'rbsp'+probe+'_adjust_flag',-2,2


;;combine the spectral data with the adjusted freq data
;; options,'rbsp'+probe+'_fbk_freq_of_max_adj_smoothed','thick',2
;; store_data,'spec_comb_smoothed',data=['rbsp'+probe+'_efw_64_spec0','rbsp'+probe+'_fbk_freq_of_max_adj_smoothed']
if fbk_type eq 'Ew' then store_data,'spec_comb',data=['rbsp'+probe+'_efw_64_spec0','rbsp'+probe+'_fbk_freq_of_max_orig','rbsp'+probe+'_fbk_freq_of_max_adj','fce_eq','fce_eq01']
if fbk_type eq 'Bw' then store_data,'spec_comb',data=['rbsp'+probe+'_efw_64_spec4','rbsp'+probe+'_fbk_freq_of_max_orig','rbsp'+probe+'_fbk_freq_of_max_adj','fce_eq','fce_eq01']
ylim,['spec_comb'],10,8000,1
options,'spec_comb','colors',[0,50,250,50]
ylim,['spec_comb'],10,8000,1

;; store_data,'fbk_spec_comb_smoothed',data=['rbsp'+probe+'_efw_fbk'+fbk_mode+'_'+stem+'_pk','rbsp'+probe+'_fbk_freq_of_max_adj_smoothed']
store_data,'fbk_spec_comb',data=['rbsp'+probe+'_efw_fbk'+fbk_mode+'_'+stem+'_pk','rbsp'+probe+'_fbk_freq_of_max_orig','rbsp'+probe+'_fbk_freq_of_max_adj','fce_eq','fce_eq01']
ylim,['fbk_spec_comb'],10,8000,1
ylim,'rbsp'+probe+'_efw_64_spec?',10,8000,1
options,'fbk_spec_comb','colors',[0,50,250,50]

if fbk_type eq 'Ew' then store_data,'spec_comb2',data=['rbsp'+probe+'_efw_64_spec0','fce_eq','fce_eq01']
if fbk_type eq 'Bw' then store_data,'spec_comb2',data=['rbsp'+probe+'_efw_64_spec4','fce_eq','fce_eq01']
ylim,'spec_comb2',10,8000,1
options,'spec_comb2','colors',[50,50]



;; store_data,'fbk_spec_comb_smoothed',data=['rbsp'+probe+'_efw_fbk'+fbk_mode+'_'+stem+'_pk','rbsp'+probe+'_fbk_freq_of_max_adj_smoothed']
store_data,'fbk_spec_comb',data=['rbsp'+probe+'_efw_fbk'+fbk_mode+'_'+stem+'_pk','rbsp'+probe+'_fbk_freq_of_max_orig','rbsp'+probe+'_fbk_freq_of_max_adj']
ylim,['fbk_spec_comb'],10,8000,1
ylim,'rbsp'+probe+'_efw_64_spec?',10,8000,1


if fbk_type eq 'Ew' then store_data,'spec_comb2',data=['rbsp'+probe+'_efw_64_spec0','fce_eq','fce_eq01']
if fbk_type eq 'Bw' then store_data,'spec_comb2',data=['rbsp'+probe+'_efw_64_spec4','fce_eq','fce_eq01']
ylim,'spec_comb2',10,8000,1
ylim,'spec_comb',10,8000,1


options,'rbsp'+probe+'_fbk_freq_of_max_adj',psym=4
options,'rbsp'+probe+'_fbk_freq_of_max_orig',psym=5


tplot_options,'title','from rbsp_efw_fbk_freq_interpolate_test.pro'

;Version for plotting
;tplot,['spec_comb2',$
;       'spec_comb',$
;       'maxamp_comb_v2','amp_scalefac',$
;       'rbsp'+probe+'_flag_amp_interp_limited_to_maxamp_lim',$
;       'rbspa_lshell','e12']

;stop


;version for checking flags
;tplot,['spec_comb2',$
;       'spec_comb',$
;       'fbk_spec_comb',$
;       'maxamp_comb','amp_scalefac',$
;       'rbsp'+probe+'_fbk_maxamp_orig',$
;       'rbsp'+probe+'_adjust_flag','*flag*','e12']

;stop
;Version for comparing with burst data
tplot,['spec_comb2',$
       'spec_comb',$
       'fbk_spec_comb',$
       'maxamp_comb','amp_scalefac',$
       'rbsp'+probe+'_adjust_flag','e12']


stop

;;--------------------------------------------------


get_data,'rbsp'+probe+'_fbk_maxamp_orig',data=fbk
get_data,'amp_scalefac',data=sfc


plot,fbk.y,sfc.y,xtitle='Original FBK amp [mV/m]',ytitle='FBK adj/orig',psym=4,$
     xrange=[0,20],yrange=[0,50],title='Adjustment of FBK amplitudes by!Crbsp_efw_fbk_freq_interpolate.pro!Ccalled from rbsp_efw_fbk_freq_interpolate_test.pro'


stop



get_data,'rbsp'+probe+'_fbk_maxamp_orig',data=fbk2
get_data,'amp_scalefac',data=sfc2
goo = where(fbk2.y le 2.)
if goo[0] ne -1 then fbk2.y[goo] = !values.f_nan
if goo[0] ne -1 then sfc2.y[goo] = !values.f_nan
store_data,'rbsp'+probe+'_fbk_maxamp_orig_nolowamps',data=fbk2
store_data,'amp_scalefac_nolowamps',data=sfc2

plot,fbk2.y,sfc2.y,xtitle='Original FBK amp [mV/m]',ytitle='FBK adj/orig',psym=4,$
     xrange=[0,20]

;burst glitch
;2012-10-13T17:09:53.138512	2012-10-13T17:40:39.138242















end
