import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import glob
from tqdm import tqdm as tqdm
from tqdm.auto import tqdm
import scipy.integrate
from scipy.stats import chi2
from scipy.stats import sigmaclip
from astropy.io import fits
from astroquery.vizier import Vizier
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
from astropy.time import Time
from astropy.io import ascii
import os
import re
from sklearn.gaussian_process.kernels import RBF, Matern, \
RationalQuadratic, WhiteKernel, DotProduct, ConstantKernel as C
import fulu
from fulu import *
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')


path_zeros = '{}/zp_thresholds_quadID.txt'.format(os.getcwd())
'''Constants from 12 paragraph ZTF Forced Photometry Service manual'''
SNU = 5
SNT = 3
markers = Line2D.markers
passband2lam  = {'ZTF_r': 1, 'ZTF_g': 2}
models_dict = {'BNN': bnn_aug.BayesianNetAugmentation(passband2lam),
               'NF': nf_aug.NormalizingFlowAugmentation(passband2lam),
               'NN (pytorch)': single_layer_aug.SingleLayerNetAugmentation(passband2lam),
               'NN (sklearn)': mlp_reg_aug.MLPRegressionAugmentation(passband2lam),
               'GP': gp_aug.GaussianProcessesAugmentation(passband2lam),
               'GP C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel()': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel(),  False),\
               'GP C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel()': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel())}

def parse_file(file_path):
    f = open(file_path, 'r')
    ra_template = 'R.A.\s*=\s*(\d+.\d+)\s*degrees'
    dec_template = 'Dec.\s*=\s*([-]*\d+.\d+)\s*degrees'
    ra = np.nan
    dec = np.nan
    for l in f.readlines():
        ra_res = re.findall(ra_template,l)
        if len(ra_res) > 0:
            ra = np.float(ra_res[0])
        dec_res = re.findall(dec_template,l)
        if len(dec_res) > 0:
            dec = np.float(dec_res[0])
    return {'ra': ra, 'dec': dec}
    
def iau_object_name(file_path):
    parsed_pos = parse_file(file_path)
    c = SkyCoord(parsed_pos['ra']*u.degree, parsed_pos['dec']*u.degree)
    ra_p = c.ra.to_string(unit=u.hourangle, sep='', precision=2, pad=True)
    dec_p = c.dec.to_string(sep='', precision=1, alwayssign=True, pad=True)
    jname = 'J'+ra_p+dec_p
    return jname

def object_coord(file_path):
    parsed_pos = parse_file(file_path)
    c = SkyCoord(parsed_pos['ra']*u.degree, parsed_pos['dec']*u.degree)
    ra_p = c.ra
    dec_p = c.dec
    return ra_p.deg, dec_p.deg

def panstars_query(ra_deg, dec_deg, rad_deg, maxmag=27,
                    maxsources=10000):
    """
    Query PanSTARRS @ VizieR using astroquery.vizier
    :param ra_deg: RA in degrees
    :param dec_deg: Declination in degrees
    :param rad_deg: field radius in degrees
    :param maxmag: upper limit G magnitude (optional)
    :param maxsources: maximum number of sources
    :return: astropy.table object
    """
    vquery = Vizier(columns=['objID', 'RAJ2000', 'DEJ2000',
                             'e_RAJ2000', 'e_DEJ2000',
                             'gmag', 'e_gmag',
                             'rmag', 'e_rmag',
                             'imag', 'e_imag',
                             'zmag', 'e_zmag',
                             'ymag', 'e_ymag'],
                    column_filters={"gmag":
                                    ("<%f" % maxmag)},
                    row_limit=maxsources)

    field = coord.SkyCoord(ra=ra_deg, dec=dec_deg,
                           unit=(u.deg, u.deg),
                           frame='icrs')
    return vquery.query_region(field,
                               width=("%fd" % rad_deg),
                               catalog="II/349/ps1")[0]
def color_correction(pans_df, clrcoeff, clrcoeffunc, mag, sigma_mag):
    Ps_g = pans_df['gmag'][0]
    Ps_r = pans_df['rmag'][0]

    Ps_err_g = pans_df['e_gmag'][0]
    Ps_err_r = pans_df['e_rmag'][0]

    Ps1 = Ps_g - Ps_r

    Ps1_err = Ps_err_g + Ps_err_r

    mag = mag + clrcoeff*Ps1

    sigma_mag = np.sqrt(sigma_mag**2+(Ps1*clrcoeffunc)**2+
                        (Ps_err_g*clrcoeff)**2+(Ps_err_r*clrcoeff)**2)
    return mag, sigma_mag


def sdss_query(ra_deg, dec_deg, rad_deg, maxmag=40,
                    maxsources=10000):
    pass

def airmass(df, ra, dec):
    Samuel_Oschin = EarthLocation(lat=(33 + (21 + 29 / 60.) / 60.) * u.deg,
                        lon=-(116 + (51 + 43 / 60.) / 60.) * u.deg,
                        height = 1712 * u.m) # 1700 u.m

    df['time,'] = Time(df['jd,'], format='jd')
    df['secz,'] = SkyCoord(ra, dec, unit='deg').transform_to(AltAz(obstime=df['time,'],
                                                                   location=Samuel_Oschin)).secz
    return df

def ccdquad_ID(df):
    df['CCDquadID,'] = np.nan
    for ccdid in df['ccdid,'].unique():
        for qid in df['qid,'].unique():
            
            df['CCDquadID,'][(df['ccdid,'] == ccdid)&(df['qid,'] == qid)]=4*(ccdid-1)+qid
    print("Уникальные значения CCDquadID ",df['CCDquadID,'].unique())
    return df

def zpthres (df, path):
    zp = pd.read_csv(path, sep='  ', comment='#')
    ZPthres_g = []
    ZPthres_r = []
    for aux2 in range(0, len(df), 1):

        ZPthres_g.append(zp[zp.CCDquadID == df['CCDquadID,'].iloc[aux2]].ZPthres_g.iloc[0])
        ZPthres_r.append(zp[zp.CCDquadID == df['CCDquadID,'].iloc[aux2]].ZPthres_r.iloc[0])
        
    df['ZPthres_g'] = ZPthres_g
    df['ZPthres_r'] = ZPthres_r
    return df

def proc_pars(df):
    val = []
    for i in df['procstatus'].values:

        if (type(i) is str):
            if (len(i) >=3):
                if (i[2] == ','):
                    val.append(int(i[:2]))
            else:
                val.append(int(i))
        else:
            val.append(int(i))

    assert len(val) == len(df['procstatus'].values)
    df['procstatus'] = val
    
    '''Пункт 8 (h) - procstatus = 0 (succes), 62, 65(input data problems)'''
    
    df = df[(df['procstatus'] == 0)|(df['procstatus'] == 62)|(df['procstatus'] == 65)]
    
    return df

def pre(df, ra, dec):
    
    '''Формирую колонку с номером куска матрицы'''
    
    df = ccdquad_ID(df)
    
    '''Формируем колонку с airmass'''
    
    df = airmass(df, ra, dec)
    
    '''Формируем колонку с zero points'''
    
    df = zpthres(df, path_zeros)
    
    return df

def filtering(df):
    
    '''Фильтрую по параметру infobitssci'''
    mask_inf = df['infobitssci,'] == 0
    
    '''Пункт 8 (g) - расстояние до ближайшего референса устанавливается < 1'' '''
    mask_dref = df['dnearestrefsrc,'] < 1
    
    '''Размер референса устанавливается от -0.1 до 0.884'' '''
    mask_shref = (df['nearestrefsharp,'] < 0.884)&(df['nearestrefsharp,'] > -0.1)
    
    '''Chi parameter for nearestrefmag(ratio: RMS in PSF-fit residuals / expected RMS from priors)'''
    mask_chiref = df['nearestrefchi,'] < 10.223
    
    mask = mask_inf&mask_dref&mask_shref&mask_chiref
    df_outl = df[~mask]
    df = df[mask]
    return df, df_outl

def noise_filtering(df_i):
    
    mask1_1 = df_i['zpmaginpsci,'] < np.nanmean(df_i['zpmaginpsci,']) + 3*np.nanstd(df_i['zpmaginpsci,'])
    mask1_2 = df_i['zpmaginpsci,'] > np.nanmean(df_i['zpmaginpsci,']) - 3*np.nanstd(df_i['zpmaginpsci,'])
    mask1 = mask1_1&mask1_2
    mask2_1 = df_i['zpmaginpscirms,'] < np.nanmean(df_i['zpmaginpscirms,']) + 3*np.nanstd(df_i['zpmaginpscirms,'])
    mask2_2 = df_i['zpmaginpscirms,'] > np.nanmean(df_i['zpmaginpscirms,']) - 3*np.nanstd(df_i['zpmaginpscirms,'])
    mask2 = mask2_1&mask2_2
    mask3_1 = df_i['scisigpix,'] < np.nanmean(df_i['scisigpix,']) + 3*np.nanstd(df_i['scisigpix,'])
    mask3_2 = df_i['scisigpix,'] > np.nanmean(df_i['scisigpix,']) - 3*np.nanstd(df_i['scisigpix,'])
    mask3 = mask3_1&mask3_2
    mask = mask1&mask2&mask3
    df_i_outl = df_i[~mask]
    df_i = df_i[mask]
    
    return df_i, df_i_outl

def cleaning(df_i):
    
    '''Очищение по критерию из дата-релиза ZTF'''
    if len(df_i) == 0:
        return df_i, None
    else:
        if df_i['filter,'].values[0] == 'ZTF_g':

            clean_g = df_i['ZPthres_g'] - 0.2*df_i['secz,']
            clean2_g = 26.7 - 0.20*df_i['secz,']

            mask_ztf_1_g = df_i['zpmaginpsci,'] < clean2_g
            mask_ztf_2_g = df_i['zpmaginpscirms,'] < 0.06
            mask_ztf_3_g = df_i['ncalmatches,'] > 80
            mask_ztf_4_g = df_i['zpmaginpsci,'] > clean_g

            mask_ztf_g = mask_ztf_1_g&mask_ztf_2_g&mask_ztf_3_g&mask_ztf_4_g

            df_i_outl = df_i[~mask_ztf_g]
            df_i = df_i[mask_ztf_g]

        elif df_i['filter,'].values[0] == 'ZTF_r':
            clean_r = df_i['ZPthres_r'] - 0.15*df_i['secz,']
            clean2_r = 26.65 - 0.15*df_i['secz,']

            mask_ztf_1_r = df_i['zpmaginpsci,'] < clean2_r
            mask_ztf_2_r = df_i['zpmaginpscirms,'] < 0.05
            mask_ztf_3_r = df_i['ncalmatches,'] > 120
            mask_ztf_4_r = df_i['zpmaginpsci,'] > clean_r

            mask_ztf_r = mask_ztf_1_r&mask_ztf_2_r&mask_ztf_3_r&mask_ztf_4_r

            df_i_outl = df_i[~mask_ztf_r]
            df_i = df_i[mask_ztf_r]

        return df_i, df_i_outl

def check_statistic(y_i, y_opt_chi, sig): 
    # y_i-выборка уже почищенная от выбросов и точек с большими ошибками

    """Сравниваю значение статистики являющейся 
    суммой квадратов стандратных случайных величин 
    и альфа-квантиль распределения хи-квадрат 
    со степенями свободы количество точек - 3"""

    statistic = np.nansum((y_i - y_opt_chi)**2/sig)

    for a in range(1, 100):
        kvantile = scipy.stats.distributions.chi2.ppf(1-(a/100), len(y_i)-2)
        if statistic > kvantile:
            print("T = {}".format(statistic), "kvantile = {}".format(kvantile))#, file = f)
            print("There is a variable at significance level {}%".format(a))#, file = f)
            break

def weight_mean(value, error):
    return np.nansum(value/(error**2))/np.nansum(1/(error**2))

def chi_2(mean_weight_magnitude, error, magnitude):
    chi2 = np.nansum(((magnitude - mean_weight_magnitude)**2) /(error**2))
    return chi2

def bootstrap_estimate_mean_stddev(arr, n_samples=1000):
    np.random.seed(0)
    arr = np.array(arr)
    print("Длина изначального вектора хи-квадратов ", len(arr))
    print(np.sum(np.isnan(arr)))
    bs_samples = np.random.randint(0, len(arr), size=(n_samples, len(arr)))
    bs_samples = np.nanmean(arr[bs_samples], axis=1)
    sigma = np.sqrt(np.nansum((bs_samples - np.nanmean(bs_samples))**2) / (n_samples - 1))
    print("Длина бутстрап-вектора ", len(bs_samples))
    print(np.sum(np.isnan(bs_samples)))
    return np.nanmean(bs_samples), sigma

def chi_2_norm(mean_weight_magnitude, error, magnitude):

    array_chi = ((magnitude - mean_weight_magnitude)**2) /(error**2)
    chi2_norm = np.nanmean(((magnitude - mean_weight_magnitude)**2) /(error**2))
    chi_mean, std_chi = bootstrap_estimate_mean_stddev(array_chi)

    return np.round(chi2_norm, 2), np.round(chi_mean, 2), np.round(std_chi, 2)

def p_value_compare(chi, n):

    if n < 3:
        print("n < 3")
        return np.nan, np.nan

    p_value = 1 - chi2.cdf(chi, n - 1)
    p_value_rounded = np.round(p_value, 6)
    print("p_value = {}".format(p_value_rounded))

    for alpha in np.arange(0.000001, 1., 0.000001):
        if p_value_rounded < alpha:
            print("Chi2 = {}".format(chi))
            print("p_value = {}".format(p_value), " < alpha = {}".format(alpha))
            print("There is a variable at significance level {}%".format(np.round(alpha*100, 6)))

            return p_value, alpha
    return p_value, np.nan

def compile_obj(t, flux, flux_err, passband):
    obj = pd.DataFrame()
    obj['mjd']      = t
    obj['flux']     = flux
    obj['flux_err'] = flux_err
    obj['passband'] = passband
    return obj

def plot_lc(filter_data, color, link, ax1, outl=False):
    ra, dec = object_coord(link)
    iau_name = iau_object_name(link)
    """Функция строит кривую блеска и
    гистограмму выборки зв.вел. нормированной 
    на корень оптимального значения дисперсии, 
    а также вычитая среднее значение, 
    таким образом получаются стандратные нормальные величины"""

    """Тут идут формулы из 13 раздела инструкции"""

    nearestrefflux = 10**(0.4*(filter_data['zpdiff,'].values - filter_data['nearestrefmag,'].values))
    nearestreffluxunc = filter_data['nearestrefmagunc,'].values * nearestrefflux / 1.0857
    print("В forcediffimfluxunc есть nan: ", \
          np.any(np.isnan(filter_data['forcediffimfluxunc,'].values)), \
          "В nearestreffluxunc есть nan: ",np.any(np.isnan(nearestreffluxunc)))
    Fluxtot = filter_data['forcediffimflux,'].values + nearestrefflux
    Fluxunctot = np.sqrt(filter_data['forcediffimfluxunc,'].values**2 - nearestreffluxunc**2)

    print("Длина Fluxunctot ", len(Fluxunctot))

    """Если в Fluxunctot есть nan, беру корень из разности квадратов, 
    так величина ошибки будет определена"""

    Fluxunctot[np.isnan(Fluxunctot)] = \
    np.sqrt(np.nanvar(filter_data['forcediffimfluxunc,'].values[np.isnan(Fluxunctot)] \
                                                      - nearestreffluxunc[np.isnan(Fluxunctot)]))
    if (len(Fluxunctot) != 0)&(np.any(Fluxunctot == 0.0)):
        Fluxunctot[Fluxunctot == 0.0] = np.min(Fluxunctot[Fluxunctot != 0.0])
        assert np.any(Fluxunctot != 0.0)
        print("Min Fluxunctot ", np.nanmin(Fluxunctot))

    assert np.any(Fluxunctot is not np.nan)
    assert np.any(Fluxunctot != 0.0)
    print("В Fluxunctot есть nan: ", np.any(np.isnan(Fluxunctot)))

    print("Есть в Fluxtot inf ", np.sum(np.isinf(Fluxtot)))
    SNRtot = Fluxtot / Fluxunctot


    """Условие из инструкции, там в 13 пункте это if\else, 
    но тут проще было сделать mask==True и mask==False""" 

    mask = SNRtot > SNT
    # we have a “confident” detection, compute and plot mag with error bar:
    mag = filter_data['zpdiff,'].values[mask] - 2.5*np.log10(Fluxtot[mask])
    clrcoeff = filter_data['clrcoeff,'].values[mask]
    clrcoeffunc = filter_data['clrcoeffunc,'].values[mask]
    sigma_mag = 1.0857 / SNRtot[mask]
    ccdQuadId_list = filter_data['CCDquadID,'].values[mask]

    """Коррректировка цвета"""

    try:
        pans_df = panstars_query(ra, dec, 0.00041667).to_pandas()

        mag, sigma_mag = color_correction(pans_df, clrcoeff, clrcoeffunc, mag, sigma_mag)

    except IndexError as e:
        print('ЦВЕТОВАЯ КОРРЕКЦИЯ НЕ ПРОВЕДЕНА')

    print("Есть inf в SNRtot", np.sum(np.isinf(SNRtot[mask])))

    sigma_mag = sigma_mag
    x = filter_data['jd,'].values[mask] - 2400000.5
    y = mag
    median_y = np.nanmedian(y)
    print("Медиана выборки = {}".format(median_y))

    print("Количество выбросов, которые отметаем ", len(x) - len(x))

    y_weight = np.array([weight_mean(y, sigma_mag) \
                        for i in range(len(x))]) 

    list_clr = list(mcolors.TABLEAU_COLORS)

    if outl:

        ax1.errorbar(x, y, yerr=sigma_mag,\
                 marker="*", elinewidth=2.7, linewidth = 1.5, markersize=30, \
       markeredgecolor=color, markeredgewidth=3.50, color= color,\
                 label = '“outliers” detection ' + color[0],
                     markerfacecolor = 'black', ls='none')
    else:    
        for ccdQuadId, marker, cm in zip(filter_data['CCDquadID,'].unique(),\
                                         list(markers)[2:], range(len(list_clr))[::2]):
            con = ccdQuadId_list == ccdQuadId

            if filter_data['filter,'].values[0] == 'ZTF_r':
                chi_norm_without_bootstr,chi_norm, std_chi_norm = \
            chi_2_norm(y_weight[con], sigma_mag[con], y[con])

                ax1.errorbar(x[con], y[con], yerr=sigma_mag[con],\
                     marker=marker, elinewidth=3.7, linewidth = 1.5, markersize=20, \
           markeredgecolor='red', markeredgewidth=3.50, color= 'red',\
                     label = 'ID{} “confident” '.format(int(ccdQuadId)) +\
                             color[0]+r' $\chi^2$ {}$\pm${}'.format(chi_norm,std_chi_norm),
                         markerfacecolor = list_clr[cm], ls='none')

            elif filter_data['filter,'].values[0] == 'ZTF_g':

                chi_norm_without_bootstr,chi_norm, std_chi_norm = \
            chi_2_norm(y_weight[con], sigma_mag[con], y[con])

                ax1.errorbar(x[con], y[con], yerr=sigma_mag[con],\
                     marker=marker, elinewidth=3.7, linewidth = 1.5, markersize=20, \
           markeredgecolor='green', markeredgewidth=3.50, color= 'green',\
                     label = 'ID{} “confident” '.format(int(ccdQuadId)) + color[0]+\
                             r' $\chi^2$ {}$\pm${}'.format(chi_norm,std_chi_norm),
                         markerfacecolor = list_clr[cm+1], ls='none')

    print("Количество nan в sigma ", np.sum(np.isnan(sigma_mag)))
    chii = chi_2(y_weight, sigma_mag, y)

    N = len(y) - \
          np.isnan(sigma_mag[~np.isnan(y)]).sum() - \
          np.isnan(y).sum()
    print("Подсчёт p_value ведётся по chi2 = ", chii)
    p_val, alph = p_value_compare(chii, N)
    print("Взвешенное среднее кол-во nan", np.sum(np.isnan(y_weight)))
    chi_norm_without_bootstr,chi_norm, std_chi_norm = \
        chi_2_norm(y_weight, sigma_mag, y)
    print("Chi_without_bootstr = ", chi_norm_without_bootstr)
    print(iau_name, " Chi_norm_", color[0], " = ", chi_norm, " +- ", std_chi_norm)
    print("Количество точек N = ", N)

    return iau_name, chi_norm, std_chi_norm, p_val, alph, N, x, y,sigma_mag 


def check_and_plot(link, *, make_fulu=False, model_name = 'GP', make_8_i=True, make_11 = True, save=False):
    
    iau_name = iau_object_name(link)
    ra, dec = object_coord(link)
    
    '''Считываю txt файл без закомменченных строк'''
    
    df = pd.read_csv(link, sep=' ', comment='#')
    df = df.drop('Unnamed: 0', 1)
    df = proc_pars(df) # очищает от проблемных output с самого сервиса
    print(len(df))
    if len(df) < 3:
        return None
    df = pre(df, ra, dec) #формируются колонки для проверки критерия из dr ztf
    
    '''Делаю 2 дата-фрейма для разных фильтров'''

    red = df[df['filter,'] == 'ZTF_r']
    green = df[df['filter,'] == 'ZTF_g']

    '''Очищение по критерию из дата-релиза ZTF'''
    
    red, red_outl1 = filtering(red) #очищает по критериям из инструкции самого сервиса
    green, green_outl1 = filtering(green) # очищает по критериям из инструкции самого сервиса
    print(len(red), len(green))
    if (len(red) < 3)&(len(green) < 3):
        return None
    
    red, red_outl2 = cleaning(red)
    green, green_outl2 = cleaning(green)
    
    '''Пункт 8 (i) - zpmaginpsci, zpmaginpscirms, and/or scisigpix'''
    if make_8_i:
        red, red_outl3 = noise_filtering(red)
        green, green_outl3 = noise_filtering(green)
    
    '''В инструкции пункт про домножение ошибок на корень из хи-квадрат'''
    if make_11:
        for field in df['field,'].unique():

            chi_mean_r = np.nanmean(red['forcediffimchisq,'][red['field,'] == field].values)
            red['forcediffimfluxunc,'][red['field,'] == field] =\
            red['forcediffimfluxunc,'][red['field,'] == field]*np.sqrt(chi_mean_r)

            #for dframe in green_fields:
            chi_mean_g = np.nanmean(green['forcediffimchisq,'][green['field,'] == field].values)
            green['forcediffimfluxunc,'][green['field,'] == field] =\
            green['forcediffimfluxunc,'][green['field,'] == field]*np.sqrt(chi_mean_g)

    red_outl_end = pd.concat([red_outl1, red_outl2, red_outl3])
    green_outl_end = pd.concat([green_outl1, green_outl2, green_outl3])
    '''Параметры осей и названий графиков'''

    fig = plt.figure(figsize = (35,25), dpi=400)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.91, bottom=0.1)
    plt.rcParams.update({'font.size': 80})
    ax1 = plt.subplot(1, 1, 1)
    ax1.set_title(iau_name, size = 85)
    ax1.set_xlabel('MJD', size = 80)
    ax1.set_ylabel('Magnitude', size = 80)
    ax1.invert_yaxis()
    ax1.spines['bottom'].set_linewidth(7)
    ax1.spines['top'].set_linewidth(7)
    ax1.spines['left'].set_linewidth(7)
    ax1.spines['right'].set_linewidth(7)

    iau_name_red, chi_norm_red, std_chi_norm_red, p_val_red, alph_red, N_red, x_r, y_r, s_r =\
    plot_lc(red, 'red', ax1=ax1, link=link)
    iau_name_red_, chi_norm_red_, std_chi_norm_red_, p_val_red_, alph_red_, N_red_, x_r_, y_r_, s_r_ =\
    plot_lc(red_outl_end, 'red', ax1=ax1, link=link, outl = True)
    iau_name_green, chi_norm_green, std_chi_norm_green, p_val_green, alph_green, N_green, x_g, y_g, s_g =\
    plot_lc(green, 'green', ax1=ax1, link=link)
    iau_name_green_, chi_norm_green_, std_chi_norm_green_, p_val_green_, alph_green_, N_green_, x_g_, y_g_, s_g_ =\
    plot_lc(green_outl_end, 'green', link=link, ax1=ax1, outl = True)
    if make_fulu:

        '''FULU'''
        try:
            t1 = compile_obj(x_r, y_r, s_r, ['ZTF_r' for i in range(len(y_r))])
            t2 = compile_obj(x_g, y_g, s_g, ['ZTF_g' for i in range(len(y_g))])
            t_end = pd.concat([t1, t2])    
            # fit augmentation model
            model = models_dict[model_name]
            model.fit(t_end['mjd'].values, t_end['flux'].values, t_end['flux_err'].values, t_end['passband'].values)
            model.plot(ax = ax1, n_approx=10000)

            t1_ = compile_obj(x_r_, y_r_, s_r_, ['ZTF_r' for i in range(len(y_r_))])
            t2_ = compile_obj(x_g_, y_g_, s_g_, ['ZTF_g' for i in range(len(y_g_))])
            t_end_ = pd.concat([t1, t2, t1_, t2_])    
            # fit augmentation model
            model = models_dict[model_name]
            model.fit(t_end_['mjd'].values, t_end_['flux'].values, t_end_['flux_err'].values,\
                      t_end_['passband'].values)
            model.plot(ax = ax1, n_approx=10000)
        except: ValueError
        ''' FULU ENDED'''

    ax1.grid(color='violet', linewidth = 3.5)
    
    ax1.set_title(iau_name, size = 85)
    ax1.legend(*[*zip(*{l:h for h,l in zip(*ax1.get_legend_handles_labels())}.items())][::-1],
               prop={'size':35}, fontsize='x-large', loc=0, framealpha=0.55)
    
    pathh = '/home/masha/ZTF/Mbh1Msun/adass_plots/' + iau_name + 'fulu_ADASS.svg'
    if save:
        fig.savefig(pathh)
    else:
        plt.show()
    return  iau_name_red, chi_norm_red, std_chi_norm_red, p_val_red, alph_red, N_red, chi_norm_green, std_chi_norm_green, p_val_green, alph_green, N_green

def make_fits(input_files, fits_name, make_latex=False):
    
    iau_names = []
    chi_norm_reds = []
    std_chi_norm_reds = []
    p_val_reds = []
    alph_reds = []
    N_reds = []
    chi_norm_greens = []
    std_chi_norm_greens = []
    p_val_greens = []
    alph_greens = []
    N_greens = []

    for ff in input_files:
        try:
            print(ff)
            iau_name, chi_norm_red, std_chi_norm_red, p_val_red, \
            alph_red, N_red, chi_norm_green, \
            std_chi_norm_green, p_val_green, alph_green, N_green = check_and_plot(ff)

            iau_names.append(iau_name)

            chi_norm_reds.append(chi_norm_red)
            std_chi_norm_reds.append(std_chi_norm_red)
            p_val_reds.append(p_val_red)
            alph_reds.append(alph_red)
            N_reds.append(N_red)

            chi_norm_greens.append(chi_norm_green)
            std_chi_norm_greens.append(std_chi_norm_green)
            p_val_greens.append(p_val_green)
            alph_greens.append(alph_green)
            N_greens.append(N_green)
        except: TypeError

    col1 = fits.Column(name='iau_names', format='20A', array=iau_names)
    col2 = fits.Column(name='chi_norm_reds', format='D', array=chi_norm_reds)
    col3 = fits.Column(name='std_chi_norm_reds', format='D', array=std_chi_norm_reds)
    col4 = fits.Column(name='p_val_reds', format='D', array=p_val_reds)
    col5 = fits.Column(name='alph_reds', format='D', array=alph_reds)
    col6 = fits.Column(name='N_reds', format='J', array=N_reds)
    col7 = fits.Column(name='chi_norm_greens', format='D', array=chi_norm_greens)
    col8 = fits.Column(name='std_chi_norm_greens', format='D', array=std_chi_norm_greens)
    col9 = fits.Column(name='p_val_greens', format='D', array=p_val_greens)
    col10 = fits.Column(name='alph_greens', format='D', array=alph_greens)
    col11 = fits.Column(name='N_greens', format='J', array=N_greens)

    hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11])
    hdu.writeto('{}.fits'.format(fits_name))
    
    if make_latex:
        data = {'Iau name': iau_names, r'$\chi^{2}_{red}$': chi_norm_reds, \
                r'Std of $\chi^{2}_{red}$': std_chi_norm_reds, \
                r'$pvalue_{red}$': ["{:.0e}".format(number) for number in p_val_reds],\
                r'$N_{red}$': N_reds, \
                r'$\chi^{2}_{green}$': chi_norm_greens, r'Std of $\chi^{2}_{green}$': std_chi_norm_greens,\
               r'$pvalue_{green}$': ["{:.0e}".format(number) for number in p_val_greens], r'$N_{green}$': N_greens}
        latexdict = ascii.write(data, Writer=ascii.Latex, latexdict=ascii.latex.latexdicts['AA'])
        return latexdict
    
def plot_chi_hist(table_before, table_after, *, save=False, path_and_name = None):
    '''
    Parameters:
    -----------
    table_before, table_after: astropy.Table or pandas.DataFrame object
    Table with chi means before(after) filtration of points in light curve
    
    save: bool
    True when you want save light curve
    
    path_and_name: str 
    name of destination your saving graph
    '''
    
    fig = plt.figure(figsize = (35,21), dpi=400)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.89, bottom=0.1)
    plt.rcParams.update({'font.size': 80})
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.set_title(r"$\chi^2_g$", size = 85)
    ax2.set_title(r"$\chi^2_r$", size = 85)
    ax1.spines['bottom'].set_linewidth(7)
    ax1.spines['top'].set_linewidth(7)
    ax1.spines['left'].set_linewidth(7)
    ax1.spines['right'].set_linewidth(7)

    ax2.spines['bottom'].set_linewidth(7)
    ax2.spines['top'].set_linewidth(7)
    ax2.spines['left'].set_linewidth(7)
    ax2.spines['right'].set_linewidth(7)

    ax1.hist(table_before['chi_norm_greens'], bins= 50, color = 'blue', alpha = 0.5, label = 'g before filtration')
    ax1.hist(table_after['chi_norm_greens'], bins= 50, color = 'green', alpha = 0.6, label = 'g after filtration')
    ax2.hist(table_before['chi_norm_reds'], bins=50, color = 'blue', alpha = 0.5, label='r before filtration')
    ax2.hist(table_after['chi_norm_reds'], bins=50, color = 'red', alpha = 0.6, label='r after filtration')

    ax1.grid(color='violet', linewidth = 3.5)
    ax2.grid(color='violet', linewidth = 3.5)
    ax1.legend(prop={'size':35}, fontsize='x-large', loc=0, framealpha=0.75)
    ax2.legend(prop={'size':35}, fontsize='x-large', loc=0, framealpha=0.75)
    if save:
        if path_and_name is not None:
            fig.savefig(path_and_name)
        else:
            fig.savefig("{}chi_hist.svg".format(os.getcwd()))
    else:
        plt.show()
