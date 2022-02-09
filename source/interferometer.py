"""interferometer.py
James Gardner 2022
interferometer (IFO) class, not executable, uses analytic solutions for nondegenerate and degenerate internal squeezing, loaded by plotting scripts"""

import numpy as np
from nIS_Mathematica_to_Python_via_Ripoll import *
from ASDSh_dIS_Ripoll import ASDSh_dIS
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.optimize import dual_annealing
from scipy.optimize import minimize
from p_tqdm import p_map
# p_map is equivalent to with Pool() as pool:; pool.map?

# physical/mathematical constants
c = 3e8
hbar = 1e-34
pi = np.pi

readout_rate = lambda tRT, T : -0.5/tRT*np.log(1-T) # tRT: round-trip time, T: transmission

# squeezer parameter standard (high)
xRatio0 = 0.95

# Zhang2021 loss standard: T_loss_a, T_loss_b, T_loss_c, Rpd
losses0 = (100e-6, 1000e-6, 1000e-6, 0.1)

# readout standards: phiPump, psi0, psi1, psi2
pumpPhi0 = pi/2
signalRO = pumpPhi0, pi/2, pumpPhi0, 0 # phiPump and psi1 are arbitrary
idlerRO = pumpPhi0, pi/2, pumpPhi0, pi/2 # psi0 is arbitrary

# for plotting
cm_to_inch = 2.54

class IFO(object):
    def __init__(self, lambda0, L_arm, L_SRC, P_circ, T_ITM, T_SRM, M, ws=None, gbR=None, gcR=None):
        """
        
        if gbR and gcR specified, then ws must be as well
        """
        # interferometer parameters, see below for L_SRC and T_ITM
        self.lambda0 = lambda0
        self.L_arm = L_arm
        self.P_circ = P_circ
        self.T_SRM = T_SRM
        self.M = M
        
        # derived parameters
        self.w0 = 2*pi*c/self.lambda0
        self.B = sqrt(self.P_circ*self.L_arm*self.w0/(hbar*c))
        self.mu = self.M/4
        self.rho = (sqrt(2)*((self.B**2)*((self.L_arm**-2.)*hbar)))/self.mu
        self.tRT_arm = 2*self.L_arm/c

        # optional direct definition of ws, gbR, gcR
        # fixing T_SRM=0.046 and changing T_ITM and L_SRC
        if gbR is None and gcR is None:
            self.L_SRC = L_SRC 
            self.tRT_SRC = 2*self.L_SRC/c
            self.gbR = readout_rate(self.tRT_SRC, self.T_SRM)
            # default to signal-idler symmetric at SRM
            self.gcR = self.gbR
        elif gbR is None:
            # e.g. for setting gcR = 0 (closing idler port)
            self.L_SRC = L_SRC 
            self.tRT_SRC = 2*self.L_SRC/c
            self.gbR = readout_rate(self.tRT_SRC,self.T_SRM)
            self.gcR = gcR
        elif gcR is None:
            # sim. for closing signal port
            self.L_SRC = L_SRC 
            self.tRT_SRC = 2*self.L_SRC/c
            self.gcR = readout_rate(self.tRT_SRC,self.T_SRM)
            self.gbR = gbR
        else: 
            self.gbR = gbR
            self.gcR = gcR
            if self.gbR == 0:
                self.tRT_SRC = -0.5/self.gcR*np.log(1-self.T_SRM)
            else:
                self.tRT_SRC = -0.5/self.gbR*np.log(1-self.T_SRM)
            # if both specified, then L_SRC is inferred from T_SRM
            # Tsrm is for signal, calculate Tsrm_c manually from: 1-np.exp(2*pi*self.gcR/(-0.5/self.tRT_SRC))
            self.L_SRC = c*self.tRT_SRC/2

        if ws is not None: 
            self.ws = ws
            self.T_ITM = (2*self.ws/c)**2*self.L_arm*self.L_SRC # Titm inferred from ws
        else:
            self.T_ITM = T_ITM
            self.ws = 0.5*(c*(sqrt(((self.T_ITM/self.L_SRC)/self.L_arm))))            

    def print_params(self):
        """Titm and Lsrc will be inferred from ws, gbR, gcR if the latter are specified"""      
        print("""----------------------
lambda0   ={:.3e}m,
L_arm     ={:.1f}km,
L_SRC     ={:.1f}m,
P_circ    ={:.1e}W,
T_ITM     ={:.3f},
T_SRM     ={:.3f},
M         ={}kg,
ws/(2pi)  ={:.3f}kHz,
gbR/(2pi) ={:.3f}kHz,
gcR/(2pi) ={:.3f}kHz,
fFSR,arm  ={:.3f}kHz
----------------------""".format(
            self.lambda0, self.L_arm*1e-3, self.L_SRC, self.P_circ, self.T_ITM, self.T_SRM, self.M,
              self.ws/(2*pi)*1e-3, self.gbR/(2*pi)*1e-3, self.gcR/(2*pi)*1e-3, 1/self.tRT_arm*1e-3))

    def ga_fn(self, T_loss_a): return readout_rate(self.tRT_arm, T_loss_a)
    def gbtot_fn(self, T_loss_b): return self.gbR + readout_rate(self.tRT_SRC, T_loss_b)
    def gctot_fn(self, T_loss_c): return self.gcR + readout_rate(self.tRT_SRC, T_loss_c)
        
    def singularity_thr(self, T_loss_a, T_loss_b, T_loss_c):
        """Using result from Mathematica: poleSol = np.array([
        [0,
        sqrt((gctot*(gbtot+((self.ws**2)/ga))))],
        [sqrt(((((gctot*(self.ws**2))-(ga*(self.ws**2)))-((ga**2)*(gbtot+gctot)))/(gbtot+\gctot))),
        sqrt(((ga+gbtot)*(ga+(gctot+((self.ws**2)/(gbtot+gctot))))))]])
        """
        
        # ((W0, x0), (W1, x1))
        ga = self.ga_fn(T_loss_a)
        gbtot = self.gbtot_fn(T_loss_b)
        gctot = self.gctot_fn(T_loss_c)

        if T_loss_a != 0:
            thr = min(sqrt((gctot*(gbtot+((self.ws**2)/ga)))),
                      sqrt(((ga+gbtot)*(ga+(gctot+((self.ws**2)/(gbtot+gctot)))))))
        else: 
            thr = sqrt(((ga+gbtot)*(ga+(gctot+((self.ws**2)/(gbtot+gctot))))))
        return thr
    
    def singularity_thr_2(self, ga, gbtot, gctot):
        """like singularity_thr but if ga, gbtot, gctot already known"""
        if ga != 0:
            thr = min(sqrt((gctot*(gbtot+((self.ws**2)/ga)))),
                      sqrt(((ga+gbtot)*(ga+(gctot+((self.ws**2)/(gbtot+gctot)))))))
        else: 
            thr = sqrt(((ga+gbtot)*(ga+(gctot+((self.ws**2)/(gbtot+gctot))))))
        return thr
    
    def ASDSh(self, f, xRatio, T_loss_a, T_loss_b, T_loss_c, Rpd, phiPump, psi0, psi1, psi2,
              radiation_pressure_on=True, extSqzFactor=1, wm=0, psi3=0):
        """combined readout with full freedom, ASD of NSR, using fn from mathematica
        extSqzFactor = 1 is no external squeezing, 1/10 is 10dB injected external squeezing etc."""
        ga = self.ga_fn(T_loss_a)
        gbtot = self.gbtot_fn(T_loss_b)
        gctot = self.gctot_fn(T_loss_c)
        x = xRatio*self.singularity_thr_2(ga, gbtot, gctot)
        
        if radiation_pressure_on:
            rho = self.rho
        else:
            rho = 0
        
        NSR = ASDShCom(2*pi*f, self.B, self.ws, x, ga, gbtot, self.gbR, gctot, self.gcR, phiPump, psi0, psi1, psi2,
                       rho, Rpd, extSqzFactor=extSqzFactor, wm=wm, psi3=psi3)
        #if NSR.imag > 1e-30:
        #    raise ValueError("significant imaginary component")
        #else:
        #    return NSR.real
        return NSR.real
        
    def sigT(self, f, xRatio, T_loss_a, T_loss_b, T_loss_c, Rpd, phiPump, psi0, psi1, psi2,
              radiation_pressure_on=True, extSqzFactor=1, wm=0, psi3=0):
        """signal transfer function of nIS coherently combined readout from mathematica"""
        ga = self.ga_fn(T_loss_a)
        gbtot = self.gbtot_fn(T_loss_b)
        gctot = self.gctot_fn(T_loss_c)
        x = xRatio*self.singularity_thr_2(ga, gbtot, gctot)
        
        if radiation_pressure_on:
            rho = self.rho
        else:
            rho = 0
        
        sigT = sigTCom(2*pi*f, self.B, self.ws, x, ga, gbtot, self.gbR, gctot, self.gcR, phiPump, psi0, psi1, psi2,
                       rho, Rpd, extSqzFactor=extSqzFactor, wm=wm, psi3=psi3)
        return sigT.real
              
    def ASDSx(self, f, xRatio, T_loss_a, T_loss_b, T_loss_c, Rpd, phiPump, psi0, psi1, psi2,
              radiation_pressure_on=True, extSqzFactor=1, wm=0, psi3=0):
        """ASD of total quantum noise of nIS coherently combined readout from mathematica"""
        ga = self.ga_fn(T_loss_a)
        gbtot = self.gbtot_fn(T_loss_b)
        gctot = self.gctot_fn(T_loss_c)
        x = xRatio*self.singularity_thr_2(ga, gbtot, gctot)
        
        if radiation_pressure_on:
            rho = self.rho
        else:
            rho = 0
        
        sigT = ASDSxCom(2*pi*f, self.B, self.ws, x, ga, gbtot, self.gbR, gctot, self.gcR, phiPump, psi0, psi1, psi2,
                       rho, Rpd, extSqzFactor=extSqzFactor, wm=wm, psi3=psi3)
        return sigT.real
    
    def sensList_vs_freq(self, params, freq_tuple, radiation_pressure_on=True, extSqzFactor=1, wm=0, psi3=0):
        f_List = np.logspace(np.log10(freq_tuple[0]),np.log10(freq_tuple[1]),num=freq_tuple[2])       
        global sens_given_params
        def sens_given_params(f):
            return self.ASDSh(f, *params, radiation_pressure_on=radiation_pressure_on,
                              extSqzFactor=extSqzFactor, wm=wm, psi3=psi3)

        with Pool() as pool:
            return pool.map(sens_given_params, f_List)    
    
    def signalList_vs_freq(self, params, freq_tuple, radiation_pressure_on=True, extSqzFactor=1, wm=0, psi3=0):
        f_List = np.logspace(np.log10(freq_tuple[0]),np.log10(freq_tuple[1]),num=freq_tuple[2])       
        global signal_given_params
        def signal_given_params(f):
            return self.sigT(f, *params, radiation_pressure_on=radiation_pressure_on,
                             extSqzFactor=extSqzFactor, wm=wm, psi3=psi3)

        with Pool() as pool:
            return pool.map(signal_given_params, f_List)    
    
    def noiseList_vs_freq(self, params, freq_tuple, radiation_pressure_on=True, extSqzFactor=1, wm=0, psi3=0):
        f_List = np.logspace(np.log10(freq_tuple[0]),np.log10(freq_tuple[1]),num=freq_tuple[2])       
        global noise_given_params
        def noise_given_params(f):
            return self.ASDSx(f, *params, radiation_pressure_on=radiation_pressure_on,
                              extSqzFactor=extSqzFactor, wm=wm, psi3=psi3)

        with Pool() as pool:
            return pool.map(noise_given_params, f_List)           

    def sql_list_vs_freq(self, freq_tuple):
        f_List = np.logspace(np.log10(freq_tuple[0]), np.log10(freq_tuple[1]), num=freq_tuple[2])
        global SQL
        def SQL(f):
            return sqrt(8*hbar/(self.M*(2*pi*f)**2*self.L_arm**2))
        
        with Pool() as pool:
            return pool.map(SQL, f_List) 

    def plot_NSR_vs_freq(self, paramsList, freq_tuple, labels=None, save_path=None, show_fig=True, fmt_List=None,
                         radiation_pressure_List=None, extSqzFactor_List=None, wm_List=None, psi3_List=None,
                         figsize=(6,4), show_legend=True):
        """plot sensitivity vs frequency in parallel for different parameters,
        paramsList is a list of lists of arguments to ASDSh,
        freq_tuple=(fmin,fmax,fnum),
        labels is a list of legend labels finished by the legend title"""
        f_List = np.logspace(np.log10(freq_tuple[0]),np.log10(freq_tuple[1]),num=freq_tuple[2])
        #results = np.zeros((len(paramsList),len(f_List)))
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(figsize=figsize)
        num_curves = len(paramsList)
        if labels is None:
            labels = [str(i) for i in range(num_curves)]
            legend_title = 'index'
        else:
            legend_title = labels[-1]
        if fmt_List is None:
            fmt_List = np.full(num_curves, '')
        if radiation_pressure_List is None:
            radiation_pressure_List = np.full(num_curves, True)
        if extSqzFactor_List is None:
            extSqzFactor_List = np.full(num_curves, 1)
        if wm_List is None:
            wm_List = np.full(num_curves, 0)
        if psi3_List is None:
            psi3_List = np.full(num_curves, 0)

        for i, params in enumerate(paramsList):  
            global sens_given_params # allows pickling
            def sens_given_params(f):
                return self.ASDSh(f, *params, radiation_pressure_on=radiation_pressure_List[i],
                                  extSqzFactor=extSqzFactor_List[i], wm=wm_List[i], psi3=psi3_List[i])

            # automatically calls pool.close() and pool.join() upon exiting the with block
            with Pool() as pool: # check whether restarting the pool results in any speedu
                #results[i] = pool.map(sens_given_params, f_List)
                ax.loglog(f_List, pool.map(sens_given_params, f_List), fmt_List[i], label=labels[i])

        ax.set_xlabel('frequency (Hz)')
        ax.set_ylabel('sensitivity ($\mathrm{Hz}^{-1/2}$)')
        if show_legend:
            ax.legend(title=legend_title)
        ax.set_xlim(freq_tuple[0], freq_tuple[1])
        if save_path is not None:
            fig.savefig(save_path, bbox_inches = "tight")        
        if show_fig:
            plt.show()
        plt.close(fig)
        
    def plot_N_S_NSR(self, paramsList, freq_tuple, labels=None, save_path=None, show_fig=True, fmt_List=None,
                     radiation_pressure_List=None, extSqzFactor_List=None, wm_List=None, psi3_List=None,
                     color_List=None, figsize=(6,8), show_legend=True):
        """plot quantum noise, signal, and sensitivity vs frequency,
        paramsList is a list of lists of arguments to ASDSh, sigT, ASDSx,
        freq_tuple=(fmin,fmax,fnum),
        labels is a list of legend labels finished by the legend title"""
        f_List = np.logspace(np.log10(freq_tuple[0]),np.log10(freq_tuple[1]),num=freq_tuple[2])
        plt.rcParams.update({'font.size': 18})
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=figsize)
        plt.subplots_adjust(hspace=0.05)
        num_curves = len(paramsList)
        if labels is None:
            labels = [str(i) for i in range(num_curves)]
            legend_title = 'index'
        else:
            legend_title = labels[-1]
        if fmt_List is None:
            fmt_List = np.full(num_curves, '')
        if color_List is None:
            color_List = np.full(num_curves, None)
        if radiation_pressure_List is None:
            radiation_pressure_List = np.full(num_curves, True)
        if extSqzFactor_List is None:
            extSqzFactor_List = np.full(num_curves, 1)
        if wm_List is None:
            wm_List = np.full(num_curves, 0)
        if psi3_List is None:
            psi3_List = np.full(num_curves, 0)
#         axs[0].set_yscale('log')
            
        for i, params in enumerate(paramsList):  
            axs[0].plot(f_List, 20*np.log10(np.array(self.noiseList_vs_freq(params, freq_tuple, radiation_pressure_on=radiation_pressure_List[i], extSqzFactor=extSqzFactor_List[i], wm=wm_List[i], psi3=psi3_List[i]))),
                          fmt_List[i], color=color_List[i])
            axs[1].loglog(f_List, self.signalList_vs_freq(params, freq_tuple, radiation_pressure_on=radiation_pressure_List[i], extSqzFactor=extSqzFactor_List[i],
                                                          wm=wm_List[i], psi3=psi3_List[i]),
                          fmt_List[i], color=color_List[i])
            axs[2].loglog(f_List, self.sensList_vs_freq(params, freq_tuple, radiation_pressure_on=radiation_pressure_List[i], extSqzFactor=extSqzFactor_List[i],
                                                        wm=wm_List[i], psi3=psi3_List[i]),
                          fmt_List[i], color=color_List[i], label=labels[i])

        axs[2].set_xlabel('frequency (Hz)')
        axs[0].set_ylabel('quantum noise\n(dB)')
        axs[1].set_ylabel('signal response')
        axs[2].set_ylabel('sensitivity ($\mathrm{Hz}^{-1/2}$)')
        
        if show_legend=='no sqz/nIS':
            from matplotlib.lines import Line2D
            line1 = Line2D([0], [0], label='no squeezing', color='b')
            line2 = Line2D([0], [0], label='nondegenerate\ninternal squeezing', color='r')
            axs[2].legend(handles=[line1, line2])
        elif show_legend:
            axs[2].legend(title=legend_title)
        axs[2].set_xlim(freq_tuple[0], freq_tuple[1])
        if save_path is not None:
            fig.savefig(save_path, bbox_inches = "tight")        
        if show_fig:
            plt.show()
        plt.close(fig)        

    def plot_N_S_NSR_compact(self, paramsList, freq_tuple, labels=None, save_path=None,
                             show_fig=True, fmt_List=None, radiation_pressure_List=None, extSqzFactor_List=None,
                             wm_List=None, psi3_List=None, figsize=(12, 6), show_legend=True,
                             color_List=None, width_ratios=[1, 2], auto_xticks=True, suptitle=None, show_sql=False):
        """plot_N_S_NSR except that sensitivity plot is beside the other two"""
        f_List = np.logspace(np.log10(freq_tuple[0]),np.log10(freq_tuple[1]),num=freq_tuple[2])    
        num_curves = len(paramsList)
        if labels is None:
            labels = [str(i) for i in range(num_curves)]
            legend_title = 'index'
        else:
            legend_title = labels[-1]
        if fmt_List is None:
            fmt_List = np.full(num_curves, '')
        if color_List is None:
            color_List = np.full(num_curves, None)
        if radiation_pressure_List is None:
            radiation_pressure_List = np.full(num_curves, True)
        if extSqzFactor_List is None:
            extSqzFactor_List = np.full(num_curves, 1)
        if wm_List is None:
            wm_List = np.full(num_curves, 0)
        if psi3_List is None:
            psi3_List = np.full(num_curves, 0)    

        plt.rcParams.update({'font.size': 18})
        fig = plt.figure(figsize=figsize)

        gs = fig.add_gridspec(2, 2, hspace=0.05, wspace=0.32,
                              height_ratios=[1, 1], width_ratios=width_ratios)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[:, 1])
        ax1.axhline(0, color='k', linewidth=1)

        if show_sql:
            ax3.loglog(f_List, self.sql_list_vs_freq(freq_tuple), 'k', label='SQL')
        
        for i, params in enumerate(paramsList):  
            ax1.plot(f_List, 20*np.log10(np.array(self.noiseList_vs_freq(
                params, freq_tuple, radiation_pressure_on=radiation_pressure_List[i], extSqzFactor=extSqzFactor_List[i], wm=wm_List[i], psi3=psi3_List[i]))), fmt_List[i], color=color_List[i])
            ax2.loglog(f_List, self.signalList_vs_freq(
                params, freq_tuple, radiation_pressure_on=radiation_pressure_List[i], extSqzFactor=extSqzFactor_List[i], wm=wm_List[i], psi3=psi3_List[i]), fmt_List[i], color=color_List[i])
            ax3.loglog(f_List, self.sensList_vs_freq(
                params, freq_tuple, radiation_pressure_on=radiation_pressure_List[i], extSqzFactor=extSqzFactor_List[i], wm=wm_List[i], psi3=psi3_List[i]), fmt_List[i], color=color_List[i], label=labels[i])

        ax2.set_xlabel('frequency (Hz)') # units convention: commas over division
        ax3.set_xlabel('frequency (Hz)')
        ax1.set_ylabel('quantum noise (dB)')
        ax2.set_ylabel('signal response')
        ax3.set_ylabel('sensitivity ($\mathrm{Hz}^{-1/2}$)')
        ax1.set_xscale('log')
        ax1.set_xlim(freq_tuple[0], freq_tuple[1])
        ax2.set_xlim(freq_tuple[0], freq_tuple[1])
        ax3.set_xlim(freq_tuple[0], freq_tuple[1])  
        ax1.set_xticklabels([])  
        if auto_xticks:
            ax1.set_xticks([1e1, 1e2, 1e3, 1e4])
            ax2.set_xticks([1e1, 1e2, 1e3, 1e4])
            ax2.set_xticklabels(['10', '100', '1000', '$10^4$'])
            ax3.set_xticks([1e1, 1e2, 1e3, 1e4])
            ax3.set_xticklabels(['10', '100', '1000', '$10^4$'])
        if suptitle is not None:
            fig.suptitle(suptitle)

        if show_legend=='no sqz/nIS':
            from matplotlib.lines import Line2D
            line1 = Line2D([0], [0], label='no squeezing', color='b')
            line2 = Line2D([0], [0], label='nondegenerate\ninternal squeezing', color='r')
#             line3 = Line2D([0], [0], label='nondegenerate internal squeezing\n+ 10dB injected external squeezing', color=(255,0,255))
            ax3.legend(handles=[line1, line2])
        elif show_legend:
            ax3.legend(title=legend_title)

        fig.align_xlabels()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches = "tight")        
        if show_fig:
            plt.show()
        plt.close(fig)        
        
    def point_log_sens_given_angles(self, psi_angles, *params):
        """returns log10 of ASD of NSR to avoid underflow when optimising,
        psi_angles is a list of [psi0, psi1, psi2] to optimise over,
        params is fixed values of [f (the point), xRatio, T_loss_a, T_loss_b, T_loss_c, Rpd, pumpPhi]"""
        return np.log10(self.ASDSh(*params, *psi_angles))    
    
    def point_log_sens_given_angles_complex(self, psi_angles_complex, *params):
        """returns log10 of ASD of NSR to avoid underflow when optimising,
        psi_angles is a list of [psi0, psi1, psi2, psi3] to optimise over,
        params is fixed values of [f (the point), xRatio, T_loss_a, T_loss_b, T_loss_c, Rpd, pumpPhi]"""
        return np.log10(self.ASDSh(*params, *psi_angles_complex[:-1], psi3=psi_angles_complex[-1]))      
    
    def save_idler_varRO(self, freq_tuple, losses1, file_tag, max_iter=5):
        """saves a dataset of the variational readout of the idler mode
        freq_tuple=(fmin, fmax, fnum), losses1 is like losses0, file_tag for path"""
        freq_List = np.logspace(np.log10(freq_tuple[0]), np.log10(freq_tuple[1]), num=freq_tuple[2])

        def point_log_sens_given_psi1(psi1_vec, *params):
            """psi1_vec = [psi1], params as in ifo.point_log_sens_given_angles"""
            # pumpPhi: fixed (pi/2), psi0: arb., psi1: varying, psi2: pi/2, psi3: arb.
            return self.point_log_sens_given_angles([0, psi1_vec[0], pi/2], *params)

        bounds1 = ((0,2*pi),)  
        global idler_varRO_given_freq
        def idler_varRO_given_freq(freq):
            args0 = (freq, xRatio0, *losses1, pumpPhi0)
            result = dual_annealing(point_log_sens_given_psi1,
                                    bounds=bounds1, args=args0, maxiter=max_iter)
            return result.x

        psi1_List = np.array(p_map(idler_varRO_given_freq, freq_List))

        fixed_idler_sens = self.sensList_vs_freq([xRatio0, *losses1, *idlerRO], freq_tuple)

        # sensitivity from psi1_List
        params0 = (xRatio0, *losses1, pumpPhi0)

        global sens_idler_varRO_given_freq
        def sens_idler_varRO_given_freq(f):
            freq_index = np.searchsorted(freq_List, f)
            return self.ASDSh(f, *params0, 0, psi1_List[freq_index], pi/2)

        sens_idler_varRO = np.array(p_map(sens_idler_varRO_given_freq, freq_List))

        data_set = np.empty((freq_tuple[2], 4))
        # frequency, variational psi1, variational sens, fixed sens
        data_set[:, 0], data_set[:, 1], data_set[:, 2], data_set[:, 3] = (freq_List,
                                                                          psi1_List.transpose()[0],
                                                                          sens_idler_varRO.transpose()[0],
                                                                          fixed_idler_sens)
        np.save('./optimal_angles/data_idler_(freq,psi1,varSens,fixedSens)--{}.npy'.format(file_tag), data_set)        
    
    def save_signal_varRO(self, freq_tuple, losses1, file_tag, max_iter=5):
        """saves a dataset of the variational readout of the signal mode
        freq_tuple=(fmin, fmax, fnum), losses1 is like losses0, file_tag for path"""
        freq_List = np.logspace(np.log10(freq_tuple[0]), np.log10(freq_tuple[1]), num=freq_tuple[2])

        def point_log_sens_given_psi0(psi0_vec, *params):
            """psi0_vec = [psi0], params as in ifo.point_log_sens_given_angles"""
            # pumpPhi: arb., psi0: varying, psi1: arb., psi2: 0, psi3: arb.
            return self.point_log_sens_given_angles([psi0_vec[0], pi/2, 0], *params)

        bounds1 = ((0,2*pi),)  
        global signal_varRO_given_freq
        def signal_varRO_given_freq(freq):
            args0 = (freq, xRatio0, *losses1, pumpPhi0)
            result = dual_annealing(point_log_sens_given_psi0,
                                    bounds=bounds1, args=args0, maxiter=max_iter)
            return result.x

        psi0_List = np.array(p_map(signal_varRO_given_freq, freq_List))

        fixed_signal_sens = self.sensList_vs_freq([xRatio0, *losses1, *signalRO], freq_tuple)

        # sensitivity from psi0_List
        params0 = (xRatio0, *losses1, pumpPhi0)

        global sens_signal_varRO_given_freq
        def sens_signal_varRO_given_freq(f):
            freq_index = np.searchsorted(freq_List, f)
            return self.ASDSh(f, *params0, psi0_List[freq_index], pi/2, 0)

        sens_signal_varRO = np.array(p_map(sens_signal_varRO_given_freq, freq_List))

        data_set = np.empty((freq_tuple[2], 4))
        # frequency, variational psi0, variational sens, fixed sens
        data_set[:, 0], data_set[:, 1], data_set[:, 2], data_set[:, 3] = (freq_List,
                                                                          psi0_List.transpose()[0],
                                                                          sens_signal_varRO.transpose()[0],
                                                                          fixed_signal_sens)
        np.save('./optimal_angles/data_signal_(freq,psi0,varSens,fixedSens)--{}.npy'.format(file_tag), data_set)    

    
    def optimal_angles_given_freq(self, freq, xRatio, losses1, pumpPhi=pumpPhi0, max_iter=5):
        bounds1 = ((0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi))
        args0 = (freq, xRatio, *losses1, pumpPhi)
        result = dual_annealing(self.point_log_sens_given_angles_complex,
                                    bounds=bounds1, args=args0, maxiter=max_iter)
        return result.x
    
    def optimal_sens_given_freq(self, freq, xRatio, losses1, pumpPhi=pumpPhi0, max_iter=5):
        bounds1 = ((0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi))
        args0 = (freq, xRatio, *losses1, pumpPhi)
        angles = self.optimal_angles_given_freq(freq, xRatio, losses1)  
        return self.ASDSh(freq, xRatio, *losses1, pumpPhi, *angles[:-1], psi3=angles[-1])    
    
    def save_optimal_filter(self, freq_tuple, losses1, file_tag, max_iter=5):
        """saves the optimal filter using complex coefficients between fmin and fmax
        freq_tuple = (fmin, fmax, fnum), losses1 like losses0, file_tag for path"""
        freq_List = np.logspace(np.log10(freq_tuple[0]), np.log10(freq_tuple[1]), num=freq_tuple[2])       
        
        #bounds1 = ((0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi))
        global global_min_w_complex_given_freq
        def global_min_w_complex_given_freq(freq):
            return self.optimal_angles_given_freq(freq, xRatio0, losses1, maxiter=max_iter)

        angles_complex_List = np.array(p_map(global_min_w_complex_given_freq, freq_List))

        # sensitivity from angles complex
        params0 = (xRatio0, *losses1, pumpPhi0)

        global sens_w_variationalRO_complex
        def sens_w_variationalRO_complex(f):
            def _psi(i, freq_index):
                return angles_complex_List[freq_index, i]
            
            freq_index = np.searchsorted(freq_List, f)
            return self.ASDSh(f, *params0,
                             _psi(0, freq_index), _psi(1, freq_index),
                             _psi(2, freq_index), psi3=_psi(3, freq_index))

        sens_varRO_complex_List = np.array(p_map(sens_w_variationalRO_complex, freq_List))

        data_set = np.empty((freq_tuple[2], 6))
        data_set[:, 0], data_set[:, 1:5], data_set[:, 5] = freq_List, angles_complex_List, sens_varRO_complex_List
        np.save('./optimal_angles/data_optimal_(freq,angles_complex,sens)--{}.npy'.format(file_tag), data_set)            

    def save_optimal_filter_no_variational(self, freq_tuple, losses1, file_tag, max_iter=5):
        """saves the optimal filter only changing psi2 and psi3 between fmin and fmax
        freq_tuple = (fmin, fmax, fnum), losses1 like losses0, file_tag for path"""
        freq_List = np.logspace(np.log10(freq_tuple[0]), np.log10(freq_tuple[1]), num=freq_tuple[2])       
        
        def _point_log_sens_given_psi23(psi23, *params):
            return np.log10(self.ASDSh(*params, psi23[0], psi3=psi23[1]))      
            
        def _optimal_psi23_given_freq(freq, xRatio, losses1, max_iter=5):
            bounds1 = ((0,2*pi), (0,2*pi))
            args0 = (freq, xRatio, *losses1, pumpPhi0, pi/2, pumpPhi0) # signal and idler at sig. response max
            result = dual_annealing(_point_log_sens_given_psi23,
                                    bounds=bounds1, args=args0, maxiter=max_iter)
            return result.x        
        
        global global_min_w_psi23_given_freq
        def global_min_w_psi23_given_freq(freq):
            return _optimal_psi23_given_freq(freq, xRatio0, losses1, max_iter=max_iter)

        psi23_List = np.array(p_map(global_min_w_psi23_given_freq, freq_List))

        # sensitivity from angles complex
        params0 = (xRatio0, *losses1, pumpPhi0, pi/2, pumpPhi0)

        global sens_w_psi23
        def sens_w_psi23(f):
            freq_index = np.searchsorted(freq_List, f)
            return self.ASDSh(f, *params0, psi23_List[freq_index, 0], psi3=psi23_List[freq_index, 1])

        sens_psi23_List = np.array(p_map(sens_w_psi23, freq_List))

        data_set = np.empty((freq_tuple[2], 4))
        data_set[:, 0], data_set[:, 1:3], data_set[:, 3] = freq_List, psi23_List, sens_psi23_List
        np.save('./optimal_angles/data_optimal_no_var_(freq,psi2,psi3,sens)--{}.npy'.format(file_tag), data_set)            
                       
    def denom(self, W_arr, *args):
        """R^2->R^2 version of the shared part of the denominator in complex frequency W,
        args = ga, gbtot, gctot, x(=xRatio*self.singularity_thr_2(ga, gbtot, gctot))"""  
        W1, W2 = W_arr
        ga, gbtot, gctot, x = args
        #gbtot-1j*W+self.ws**2/(ga-1j*W)-x**2/(gctot-1j*W)
        return (np.array([gbtot, 0])
                + np.array([W2, -W1])
                + np.array([ga+W2, W1])*self.ws**2/((ga+W2)**2+W1**2)
                - np.array([gctot+W2, W1])*x**2/((gctot+W2)**2+W1**2))
    
    # degenerate internal squeezing
    def dIS_singularity_thr(self, ga, gbtot):
        """degenerate internal squeezing thr. given ga, gbtot"""
        if ga != 0:
            thr = min(gbtot+self.ws**2/ga, gbtot+ga)
        else: 
            thr = gbtot+ga
        return thr
    
    def dIS_ASDSh(self, f, xRatio, T_loss_a, T_loss_b, Rpd, phiPump, radiation_pressure_on=True, extSqzFactor=1):
        """degenerate internal squeezing, ASD of NSR, using fn from mathematica
        extSqzFactor = 1 is no external squeezing, 1/10 is 10dB injected external squeezing etc."""
        ga = self.ga_fn(T_loss_a)
        gbtot = self.gbtot_fn(T_loss_b)
        x = xRatio*self.dIS_singularity_thr(ga, gbtot)
        if radiation_pressure_on:
            rho = self.rho
        else:
            rho = 0
        
        NSR = ASDSh_dIS(2*pi*f, self.B, self.ws, x, ga, gbtot, self.gbR, phiPump, rho, Rpd,
                        extSqzFactor=extSqzFactor)
        return NSR.real

    def dIS_plot_NSR_vs_freq(self, paramsList, freq_tuple, labels=None, save_path=None, show_fig=True, fmt_List=None,
                     extSqzFactor_List=None, figsize=(6,4), show_legend=True, legend_size=18):
        """degenerate internal squeezing, plot sensitivity vs frequency in parallel for different parameters,
        paramsList is a list of lists of arguments to dIS_ASDSh,
        freq_tuple=(fmin,fmax,fnum),
        labels is a list of legend labels finished by the legend title"""
        f_List = np.logspace(np.log10(freq_tuple[0]),np.log10(freq_tuple[1]),num=freq_tuple[2])
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(figsize=figsize)
        if labels is None:
            labels = [str(i) for i in range(len(paramsList))]
            legend_title = 'index'
        elif len(labels) == len(paramsList):
            legend_title = ''
        else:
            legend_title = labels[-1]
        if fmt_List is None:
            fmt_List = ['' for _ in range(len(paramsList))]
        if extSqzFactor_List is None:
            extSqzFactor_List = [1 for _ in range(len(paramsList))]

        for i, params in enumerate(paramsList):  
            global _sens_given_params
            def _sens_given_params(f):
                return self.dIS_ASDSh(f, *params, extSqzFactor=extSqzFactor_List[i])

            with Pool() as pool:
                ax.loglog(f_List, pool.map(_sens_given_params, f_List), fmt_List[i], label=labels[i])
            
        ax.set_xlabel('frequency (Hz)')
        ax.set_ylabel('sensitivity ($\mathrm{Hz}^{-1/2}$)')
        if show_legend:
            ax.legend(title=legend_title, fontsize=legend_size)
        ax.set_xlim(freq_tuple[0], freq_tuple[1])
        if save_path is not None:
            fig.savefig(save_path, bbox_inches = "tight")        
        if show_fig:
            plt.show()
        plt.close(fig)

def plot_coloured_line_segments(x, y, colour_param, cmap, ax=None):
    """https://stackoverflow.com/a/36521456"""
    if ax is None:
        ax = plt.gca()
    c = cmap(colour_param)
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=c[i])
    return
