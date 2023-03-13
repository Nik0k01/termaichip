# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:51:11 2023

@author: Nik0
"""

from EOS import Gas
import numpy as np

class RealGas(Gas):
    
    def __init__(self, p_crit, t_crit, w):
         super().__init__(self, p_crit, t_crit, w)
         
    def  dept_funcs(self, p, t, state, eos):
        """
        Calculation of departure functions using the virial and cubic EOS
        
        Parameters
        ----------
        p, t : pressure (bar) and temperature (K)
        state : fluid state (Liquid : L or Vapor : V)
        eos : equation used to calculate the departure (VR, VDW, RK, SRK, PR)
        
        Returns
        dH : enthalpy departure J/mol 
        dS : entropy departure J/mol/K
        """
     
        R = RealGas.R
        t_r = t / self.t_crit
        p_r = p / self.p_crit
        # Set parameters
        eos = eos.upper()
        state = state.upper()
        if eos == "VDW":
            al = 1
            sm = 0
            ep = 0
            om = 0.125
            ps = 0.42188
            kappa = 0
        elif eos == "RK":
            al = 1 / np.sqrt(t_r)
            sm = 1
            ep = 0
            om = 0.08664
            ps = 0.42748
        elif eos == "SRK":
            kappa = 0.48 + 1.574 * self.w - 0.176 * self.w ** 2
            al = (1 + kappa * (1 - sqrt(t_r))) ** 2
            sm = 1 + np.sqrt(2)
            ep = 1 - np.sqrt(2)
            om = 0.0778
            ps = 0.45724
        else:
            kappa = 0.37464 + 1.54226 * w - 0.26992 * w ** 2
            al = (1 + kappa * (1 - np.sqrt(t_r))) ** 2
            sm = 1 + np.sqrt(2)
            ep = 1 - np.sqrt(2)
            om = 0.0778
            ps = 0.45724
            
        # Compressibility factor
        if eos == "VR":
            z = self.virial(p, t)[0]
        else: 
            z, v = self.cubic(p, t, state, eos)
            
        # Depature fucntion
        a = ps * al * R ** 2 * self.t_c ** 2 / self.p_c ** 2
        b = om * R * self.t_c / self.p_c
        Ad = a * p / (R ** 2 * t ** 2)
        Bd = b * p / (R * t)
        
        if eos == 'VR':
            # VAPOR ONLY
            if state == 'L' or state == 'LIQUID':
                print("Virial equation does not consider liquid state!")
                return None
            dH = - p_r * (1.0972 / t_r ** 2.6 - 0.083 / t_r + self.w * 
                          (0.8944 / t_r ** 5.2 - 0.139 / t_r)) * R * t 
            dS = - p_r * (0.675 / t_r ** 2.6 + self.w * 0.722 / t_r ** 5.2) * R
        
        elif eos == 'RK':
            dH = (z - 1 - 1.5 * ps / (om * t_r ** 1.5)
                  * np.log(1 + b/v)) * R * t
            dS = (np.log(z - Bd) -0.5 * ps / 
                  (om * t_r ** 1.5) * np.log(1 + b / v)) * R
        
        elif eos == 'VDW':
            dH = (z - 1 - 3.375 * Bd / t_r / z) * R * t
            dS = R * np.log(z - Bd)
            
        elif eos == 'SRK':
            dH = (z - 1 + (-kappa * np.sqrt(t_r) / (1 + kappa * (1 - 
                  np.sqrt(t_r))) - 1) * Ad / Bd * np.log(1 + Bd / z)) * R * t
            dS = (np.log(z - Bd) - kappa * np.sqrt(t_r) / (1 + kappa * 
                 (1 - np.sqrt(t_r))) * Ad / Bd * np.log(1 + Bd / z)) * R
            
        else:
            dH = (z - 1 - (Ad / (Bd * np.sqrt(8))) * (1 + kappa *np.sqrt(t_r)
                  / np.sqrt(al)) * np.log((z + sm*Bd) /
                                           (z + ep * Bd))) * R * t_r
            dS = (np.log(z - Bd) - (Ad / (Bd * np.sqrt(8))) * 
                 (kappa * np.sqrt(t_r) / np.sqrt(al)) * 
                 np.log((z + sm * Bd)/(z + ep * Bd))) * R
            
        return dH, dS
            
if __name__ == '__main__':
    pass
        
            