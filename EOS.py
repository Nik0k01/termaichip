# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:14:19 2023

@author: Nik0
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
lk_constants = pd.read_csv('lkconstants.csv', header=None, 
                           names=['Simple', 'Reference'],)
class Gas:
    R = 8.3144598
    def __init__(self, p_crit, t_crit, w):
        self.p_crit = p_crit
        self.t_crit = t_crit
        self.w = w

    def virial(self, p=1, t=273.15):
        """
        Estimation of compressibility factor and molar volume given the conditions
        i.e. pressure and temperature using the viral equation of state. Accentric
        factor 
        
        Parameters
        ----------
        p : pressure and critical pressure (bar)
        t : temperature  (K)
        
        Returns
        -------
        z : compressibility factor - deviation from the ideal gas behaviour
        v : molar volume [m3/mol]
        """
        R = Gas.R
        p = p * 1e5
        p_crit = self.p_crit * 1e5
        t_r = t / self.t_crit
        B0 = 0.083 - 0.422 / (t_r ** 1.6)
        B1 = 0.139 - 0.172 / (t_r ** 4.2)
        B = R * self.t_crit / p_crit * (B0 + self.w * B1)
        z = 1 + B * p / R / t
        v = R * z * t / p
        return z, v

    def leekessler(self, p=1, t=273.15):
        """
        Estimation of compressibility factor given the conditionsi.e. pressure 
        and temperature using the Lee-Kessler equation (usually very accurate).
        
        Parameters
        ----------
        p : pressure and critical pressure (bar)
        t : temperature  (K)
        
        Returns
        -------
        z : compressibility factor - deviation from the ideal gas behaviour
        """
        p_r = p / self.p_crit
        t_r = t / self.t_crit
        w_r = 0.3978
        def calcBCD(fluid):
            if fluid == 'simple':
                b = lk_constants.iloc[0:4, 0]
                c = lk_constants.iloc[4:8, 0]
                d = lk_constants.iloc[8:, 0]
            elif fluid == 'reference':
                b = lk_constants.iloc[0:4, 1]
                c = lk_constants.iloc[4:8, 1]
                d = lk_constants.iloc[8:, 1]
                
            B = b[0] - b[1] / t_r - b[2] / t_r ** 2 - b[3] / t_r ** 3
            C = c[0] - c[1] / t_r + c[2] / t_r ** 3
            D = d[0] + d[1] / t_r
            return B, C, D
        
        def eqBWR(v_r, fluid):
            if fluid == 'simple':
                beta = lk_constants.loc['beta', 'Simple']
                gamma = lk_constants.loc['gamma', 'Simple']
                c = lk_constants.iloc[4:8, 0]
                B, C, D = calcBCD(fluid)
            elif fluid == 'reference':
                beta = lk_constants.loc['beta', 'Reference']
                gamma = lk_constants.loc['gamma', 'Reference']
                c = lk_constants.iloc[4:8, 1]
                B, C, D = calcBCD(fluid)
                
            f = 1 + B / v_r + C / v_r ** 2 + D / v_r **5 + (
                c[3] / (t_r **3 * v_r **2)) * (beta +
                gamma / v_r **2 ) * np.exp(-gamma / v_r **2) - p_r * v_r / t_r
            return f
        
        v_r0 = t_r / p_r
        # Calculation of z0 of simple fluid
        v_r = fsolve(eqBWR, v_r0, args='simple')
        z_0 = p_r * v_r / t_r
        #  Calculation of zr of reference fluid
        v_r = fsolve(eqBWR, v_r0, args='reference')
        z_r = p_r * v_r / t_r
        z_1 = 1 / w_r * (z_r - z_0)
        return float(z_0 + self.w * z_1)
             
    def cubic(self, t, p, state='gas', eos='VDW'):
        """
        Estimation of the compressibility and molar volume using cubic 
        equations of state.
        Type of cubic equation used:
            VDW - van der Waals
            RK - Redlich-Kwong
            SRK - Soave-Redlich-Kwong
            PR - Peng-Robinson
        Both capital and lowercase letters are permitted.
        
        Parameters
        ----------
        p : pressure and critical pressure (bar)
        t : temperature  (K)
        state : fluid state (either liquid or vapor)
        eos : type of cubic eq used for calculation (VDW, RK, SRK, PR)
        Returns
        -------
        z : compressibility factor - deviation from the ideal gas behaviour
        v : molar volume [m3/mol]        
        """
        t_r = t / self.t_crit
        p_r = p / self.p_crit
        R = Gas.R
        w = self.w
        eos = eos.upper()
        
        if eos == 'VDW':
            sm = 0; ep = 0; om = 0.125; ps = 0.42188; mx = [0]
        elif eos == 'RK':
            ep = 0; sm = 1; om = 0.08664; ps = 0.42748; al = 1 / np.sqrt(t_r)
            mx = (t_r ** 0.25 - 1)  / (1 - np.sqrt(t_r))
        elif eos == 'SRK':
            ep = 0; sm = 1; om = 0.08664; ps = 0.42748
            al = (1 + (0.48 + 1.574 * w - 0.176 * w ** 2) 
                  * (1 - np.sqrt(t_r))) ** 2 
            mc = np.array([0.48, 1.574, 0.176])
            mx = np.array([1, w, -w ** 2]) @ mc
        

        pass                                 
        
    
if __name__ == "__main__":
    gas = Gas(40, 419.6, 0.191)
    print(f'{gas.leekessler(20, 400):.5f}')
    print(f'{gas.virial(20, 400)[0]:.5f}')