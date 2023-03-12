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
        return z_0 + self.w * z_1
             
    def cubic(self, p, t, state='vapor', eos='VDW'):
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
        R = Gas.R * 10
        w = self.w
        eos = eos.upper()
        
        # Set parameters of the cubic equation
        if eos == 'VDW':
            sm = 0
            ep = 0
            om = 0.125
            ps = 0.42188
            al = 1
        elif eos == 'RK':
            ep = 0
            sm = 1
            om = 0.08664
            ps = 0.42748
            mx = (t_r ** 0.25 - 1)  / (1 - np.sqrt(t_r))
            al = t_r ** -0.5
        elif eos == 'SRK':
            ep = 0
            sm = 1
            om = 0.08664
            ps = 0.42748
            mc = np.array([0.48, 1.574, 0.176])
            mx = np.array([1, w, -w ** 2]) @ mc
            al = (1 + mx * (1 - t_r ** 0.5)) ** 2
        elif eos == 'PR':
            ep = 1 - np.sqrt(2)
            sm = 1 + np.sqrt(2)
            om = 0.07780
            ps = 0.45724
            mc = [0.37464, 1.54226, 0.26992]
            mx = np.array([1, w, -w ** 2]) @ mc
            al = (1 + mx * (1 - t_r ** 0.5)) ** 2
                                     

        beta = om * p_r / t_r
        q = ps * al / om / t_r
        
        state = state.upper()
        z_holder = []
        v_holder = []
        # Coefficients of the cubic equation
        for item_beta, item_q in zip(beta, q):
            coefs = [1, 
                     (sm + ep) * item_beta - (1 + item_beta),
                     item_beta * (item_q + ep * sm * item_beta - 
                                  (1 + item_beta) * (sm + ep)),
                     -item_beta** 2 * (item_q + (1 + item_beta) * ep * sm)]
            cubic_eq = np.polynomial.Polynomial(coefs[::-1])
            if state == 'V' or state == 'VAPOR':
                z = cubic_eq.roots()
                # Cut the left over imaginary part
                z = z.real[abs(z.imag)<1e-5]
                z = max(z)
                z_holder.append(z)
                v = z * R * t / p
                v_holder.append(v)
            if state == 'L' or state == 'LIQUID':
                z = cubic_eq.roots()
                # Cut the left over imaginary part
                z = z.real[abs(z.imag)<1e-5]
                z = min(z)
                z_holder.append(z)
                v = z * R * t / p
                v_holder.append(v)             
        return z_holder, v_holder
    
if __name__ == "__main__":
    ammonia = Gas(112.8, 405.65, 0.252608)
    butane = Gas(37.9, 425.12, 0.200164)
    nitrogen = Gas(34, 126.2, 0.0377215)
    co2 = Gas(73.83, 304.21, 0.223621)
    methane = Gas(45.99, 190.564, 0.0115478)
    gases = [ammonia, butane, nitrogen, co2, methane]
    names = ['ammonia', 'butane', 'nitrogen', 'co2', 'methane']
    
    p_range = np.linspace(20, 150, num=100)
    t_range = np.linspace(293, 800, num=100)
    
    z_virial = np.array([gas.virial(p_range, t_range)[0] for gas in gases])
    z_virial = pd.DataFrame(z_virial.T, columns=names)
        
    z_vdw = np.array([gas.cubic(p_range, t_range)[0] for gas in gases])
    z_vdw = pd.DataFrame(z_vdw.T, columns=names)
    
    z_srk = np.array([gas.cubic(p_range, t_range, eos='SRK')[0] for gas in gases])
    z_srk = pd.DataFrame(z_srk.T, columns=names)
    
    
    
    # print(f'{gas.leekessler(9.4573, 350):.5f}')
    # print(f'{gas.virial(9.4573, 350)[0]:.5f}')
    # print(f'{gas.cubic(9.4573, 350, "V", "SRK")[0]:.5f}')