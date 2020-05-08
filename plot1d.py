# Import installed packages
import sys
import numpy as np 
import numpy.linalg as LA
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gs
import matplotlib.legend as lg
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import h5py

# Import athena data reader
sys.path.append('/Users/tsunhinnavintsung/Workspace/Codes/Athena++')
import athena_read3 as ar 

# Matplotlib default param
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['lines.linewidth'] = 1.
plt.rcParams['lines.markersize'] = 0.7
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# Global parameters
gc = 4./3.
gg = 5./3.
gc1 = gc/(gc - 1.)
gg1 = gg/(gg - 1.)
tiny_number = 1.e-10

class Plot1d:
  def __init__(self, inputfile, file_array):
    self.inputfile = inputfile
    self.file_array = file_array
    self.open_file()
    self.read_data()
    self.runshock = False
    self.runtime = False
    self.runinstab = False

  def is_number(self, n):
    try:
      float(n)   
    except ValueError:
      return False
    return True

  def open_file(self):
    with open(inputfile, 'r') as fp:
      line = fp.readline()
      self.nx1 = 0
      self.user_output = False
      while line:
        phrase = line.strip().split(' ')
        for word in phrase:
          if word == 'problem_id':
            self.filename = line.split('=')[1].strip().split(' ')[0]
          elif word == 'nx1':
            grid = int(line.split('=')[1].strip().split(' ')[0])
            self.nx1 = grid if grid > self.nx1 else self.nx1 
          elif word == 'dt':
            for element in phrase:
              if self.is_number(element):
                self.dt = float(element)
          elif word == 'gamma':
            for element in phrase:
              if self.is_number(element):
                gamma = float(element)
                if gamma != gg:
                  globals()[gg] = gamma
                  globals()[gg1] = gamma/(gamma - 1.)
                self.isothermal = False 
          elif word == 'iso_sound_speed':
            for element in phrase:
              if self.is_number(element):
                self.cs = float(element)
                self.isothermal = True
          elif word == 'uov':
            self.user_output = True
          elif word == 'vs_flag':
            for element in phrase:
              if self.is_number(element):
                self.cr_stream = True if int(element) == 1 else False
          elif word == 'gamma_c':
            for element in phrase:
              if self.is_number(element):
                gamma_c = float(element) 
                if gamma_c != gc:
                  globals()[gc] = gamma_c 
                  globals()[gc1] = gamma_c/(gamma_c - 1.)
          elif word == 'vmax':
            for element in phrase:
              if self.is_number(element):
                self.vmax = float(element) 
          else:
            continue  
        line = fp.readline()
    return 
      
  def read_data(self):
    filename = self.filename 

    # For no adaptive mesh refinement
    self.x1v = np.array(ar.athdf('./' + filename + '.out1.00000.athdf')['x1v'])
    x1v = self.x1v
    self.dx = x1v[1] - x1v[0]

    # Choosing files for display
    file_array = self.file_array 
    self.time_array = np.zeros(np.size(file_array))

    # Number of parameters of interest
    self.rho_array = np.zeros((np.size(file_array), np.size(x1v)))
    self.pg_array = np.zeros((np.size(file_array), np.size(x1v)))
    self.vx_array = np.zeros((np.size(file_array), np.size(x1v)))
    self.ecr_array = np.zeros((np.size(file_array), np.size(x1v)))
    self.fcx_array = np.zeros((np.size(file_array), np.size(x1v)))
    self.vs_array = np.zeros((np.size(file_array), np.size(x1v)))
    self.sigma_adv_array = np.zeros((np.size(file_array), np.size(x1v)))
    self.sigma_diff_array = np.zeros((np.size(file_array), np.size(x1v)))
    if self.cr_stream:
      self.bx_array = np.zeros((np.size(file_array), np.size(x1v)))

    # Preparing for uov
    if self.user_output:
      self.uovnum = ar.athdf('./' + filename + '.out2.00000.athdf')['NumVariables'][0]
      self.uovname = ar.athdf('./' + filename + '.out2.00000.athdf')['VariableNames']
      self.uovname = [self.uovname[i].decode('utf-8') for i in range(self.uovnum)] 
      self.uov_array = np.zeros((np.size(file_array), self.uovnum, np.size(x1v)))

    # Extracting data
    for i, file in enumerate(file_array):
      data = ar.athdf('./' + filename + '.out1.' + format(file, '05d') \
        + '.athdf')
      self.time_array[i] = float('{0:f}'.format(data['Time']))
      self.rho_array[i, :] = data['rho'][0, 0, :]
      self.pg_array[i, :] = data['press'][0, 0, :]
      self.vx_array[i, :] = data['vel1'][0, 0, :]
      self.ecr_array[i, :] = data['Ec'][0, 0, :] 
      self.fcx_array[i, :] = data['Fc1'][0, 0, :]
      self.vs_array[i, :] = data['Vc1'][0, 0, :]
      self.sigma_adv_array[i, :] = data['Sigma_adv1'][0, 0, :]
      self.sigma_diff_array[i, :] = data['Sigma_diff1'][0, 0, :]
      if self.cr_stream:
        self.bx_array[i, :] = data['Bcc1'][0, 0, :]

      if self.user_output:
        uov_data = ar.athdf('./' + filename + '.out2.' + format(file, '05d') \
          + '.athdf')
        for j, uov_name in enumerate(self.uovname):
          self.uov_array[i, j, :] = uov_data[uov_name][0, 0, :]
      
    # For constant kappa and magnetic field
    self.kappa = (gc - 1.)*self.vmax/self.sigma_diff_array[0, 0] 
    if self.cr_stream:
      self.b0 = self.bx_array[0, 0]
    return 

  def plot(self):
    file_array = self.file_array 
    time_array = self.time_array
    x1v = self.x1v
    vmax = self.vmax 

    fig = plt.figure(figsize=(12, 8))

    grids = gs.GridSpec(3, 3, figure=fig)
    ax1 = fig.add_subplot(grids[0, 0])
    ax2 = fig.add_subplot(grids[0, 1])
    ax3 = fig.add_subplot(grids[0, 2])
    ax4 = fig.add_subplot(grids[1, 0])
    ax5 = fig.add_subplot(grids[1, 1])
    ax6 = fig.add_subplot(grids[1, 2])
    ax7 = fig.add_subplot(grids[2, 0])
    ax8 = fig.add_subplot(grids[2, 1])
    ax9 = fig.add_subplot(grids[2, 2])

    lab = ['$\\rho$', '$P_g$', '$v$', '$P_c$', '$F_c$', '$P_g/\\rho^{\\gamma_g}$', '$v_s$', '$P_c/\\rho^{\\gamma_c}$', '$\\sigma_c$']

    for i, file in enumerate(file_array):
      if file == 0:
        ax1.plot(x1v, self.rho_array[i, :], 'k--',  label='t={:.3f}'.format(time_array[i]))
        ax2.plot(x1v, self.pg_array[i, :], 'k--', label='t={:.3f}'.format(time_array[i]))
        ax3.plot(x1v, self.vx_array[i, :], 'k--', label='t={:.3f}'.format(time_array[i]))
        ax4.plot(x1v, self.ecr_array[i, :]/3., 'k--',  label='t={:.3f}'.format(time_array[i]))
        ax5.plot(x1v, self.fcx_array[i, :]*vmax, 'k--', label='t={:.3f}'.format(time_array[i]))
        ax6.plot(x1v, self.pg_array[i, :]/self.rho_array[i, :]**(gg), 'k--', label='t={:.3f}'.format(time_array[i]))
        ax7.plot(x1v, self.vs_array[i, :], 'k--',  label='t={:.3f}'.format(time_array[i]))
        ax8.plot(x1v, self.ecr_array[i, :]/3./self.rho_array[i, :]**(gc), 'k--', label='t={:.3f}'.format(time_array[i]))
        ax9.semilogy(x1v, self.sigma_adv_array[i, :]/(self.sigma_adv_array[i, :]/self.sigma_diff_array[i, :] + 1.)/vmax, 'k--', label='t={:.3f}'.format(time_array[i]))
      else:
        ax1.plot(x1v, self.rho_array[i, :], 'o-',  label='t={:.3f}'.format(time_array[i]))
        ax2.plot(x1v, self.pg_array[i, :], 'o-', label='t={:.3f}'.format(time_array[i]))
        ax3.plot(x1v, self.vx_array[i, :], 'o-', label='t={:.3f}'.format(time_array[i]))
        ax4.plot(x1v, self.ecr_array[i, :]/3., 'o-',  label='t={:.3f}'.format(time_array[i]))
        ax5.plot(x1v, self.fcx_array[i, :]*vmax, 'o-', label='t={:.3f}'.format(time_array[i]))
        ax6.plot(x1v, self.pg_array[i, :]/self.rho_array[i, :]**(gg), 'o-', label='t={:.3f}'.format(time_array[i]))
        ax7.plot(x1v, self.vs_array[i, :], 'o-',  label='t={:.3f}'.format(time_array[i]))
        ax8.plot(x1v, self.ecr_array[i, :]/3./self.rho_array[i, :]**(gc), 'o-', label='t={:.3f}'.format(time_array[i]))
        ax9.plot(x1v, self.sigma_adv_array[i, :]/(self.sigma_adv_array[i, :]/self.sigma_diff_array[i, :] + 1.)/vmax, 'o-', label='t={:.3f}'.format(time_array[i]))

    for i, axes in enumerate(fig.axes):
      axes.legend(frameon=False)
      axes.set_xlabel('$x$')
      axes.set_ylabel(lab[i])
      if lab[i] != '$\\sigma_c$':
        axes.xaxis.set_minor_locator(AutoMinorLocator())
        axes.yaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout()
    return fig

  def shock(self):
    # Prompt the user to enter values
    time = float(input('Enter time: '))
    begin = float(input('Enter begining of shock: '))
    end = float(input('Enter ending of shock: '))
    start = float(input('Enter start of check: '))
    check = float(input('Enter end of check: '))

    file = np.argmin(np.abs(self.time_array - time))
    ibeg = np.argmin(np.abs(self.x1v - begin))
    iend = np.argmin(np.abs(self.x1v - end))
    x_sh = self.x1v[ibeg:iend]
    istr = np.argmin(np.abs(x_sh - start))
    ichk = np.argmin(np.abs(x_sh - check))

    # Extract variables
    rho_sh = self.rho_array[file, ibeg:iend]
    vx_sh = self.vx_array[file, ibeg:iend]
    pg_sh = self.pg_array[file, ibeg:iend]
    pc_sh = self.ecr_array[file, ibeg:iend]/3.  
    fc_sh = self.fcx_array[file, ibeg:iend]*self.vmax 
    vs_sh = self.vs_array[file, ibeg:iend]
    sa_sh = self.sigma_adv_array[file, ibeg:iend]/self.vmax 
    sd_sh = self.sigma_diff_array[file, ibeg:iend]/self.vmax 
    sc_sh = sa_sh*sd_sh/(sa_sh + sd_sh)

    # Derived quantities
    cs_sh = np.sqrt(gg*pg_sh/rho_sh)
    cc_sh = np.sqrt(gc*pc_sh/rho_sh)
    vp_sh = np.sqrt(cs_sh**2 + cc_sh**2) 
    if self.cr_stream:
      va_sh = self.b0/np.sqrt(rho_sh)
      beta = 2.*pg_sh/self.b0 

    # Quantities involving first derivatives
    drhodx_sh = np.gradient(rho_sh, x_sh)
    dvdx_sh = np.gradient(vx_sh, x_sh)
    dpgdx_sh = np.gradient(pg_sh, x_sh)
    dpcdx_sh = np.gradient(pc_sh, x_sh)
    dfcdx_sh = np.gradient(fc_sh, x_sh)
    fcsteady = gc1*pc_sh*(vx_sh + vs_sh) - dpcdx_sh/sd_sh

    # Involving second derivatives
    d2pcdx2_sh = np.gradient(dpcdx_sh, x_sh)

    # Extract upstream and downstream variables in the shock frame
    rho1, rho2 = rho_sh[istr], rho_sh[ichk]
    pg1, pg2 = pg_sh[istr], pg_sh[ichk]
    pc1, pc2 = pc_sh[istr], pc_sh[ichk]
    cs1, cs2 = cs_sh[istr], cs_sh[ichk]
    cc1, cc2 = cc_sh[istr], cc_sh[ichk]
    vs1, vs2 = vs_sh[istr], vs_sh[ichk]

    vsh = (rho1*vx_sh[istr] - rho2*vx_sh[ichk])/(rho1 - rho2)
    vx_sh = vx_sh - vsh 
    fc_sh = fc_sh - gc1*pc_sh*vsh
    fcsteady = fcsteady - gc1*pc_sh*vsh
    dfcdx_sh = dfcdx_sh - gc1*dpcdx_sh*vsh
    v1, v2 = vx_sh[istr], vx_sh[ichk] 
    fc1, fc2 = fc_sh[istr], fc_sh[ichk]

    if self.cr_stream:
      va1, va2 = va_sh[istr], va_sh[ichk]
      beta1, beta2 = beta[istr], beta[ichk]
      vp_sh = np.sqrt(cs_sh**2 + cc_sh**2*(np.abs(vx_sh) - va_sh/2.)*(np.abs(vx_sh) \
        + (gg - 1.)*va_sh)/(np.abs(vx_sh)*(np.abs(vx_sh) - va_sh)))

    vp1, vp2 = vp_sh[istr], vp_sh[ichk]
    m1, m2 = np.abs(v1)/vp1, np.abs(v2)/vp2 
    n1, n2 = pc1/(pc1 + pg1), pc2/(pc2 + pg2)

    # Save variables
    self.time = time 
    self.begin, self.end = begin, end 
    self.start, self.check = start, check

    self.file = file 
    self.ibeg, self.iend = ibeg, iend 
    self.istr, self.ichk = istr, ichk

    self.x_sh = x_sh 
    self.rho_sh = rho_sh 
    self.vx_sh = vx_sh
    self.pg_sh = pg_sh 
    self.pc_sh = pc_sh 
    self.fc_sh = fc_sh 
    self.vs_sh = vs_sh 
    self.sa_sh = sa_sh 
    self.sd_sh = sd_sh 
    self.sc_sh = sc_sh

    self.cs_sh = cs_sh 
    self.cc_sh = cc_sh 
    self.vp_sh = vp_sh
    if self.cr_stream:
      self.va_sh = va_sh 
      self.beta = beta

    self.drhodx_sh = drhodx_sh 
    self.dvdx_sh = dvdx_sh 
    self.dpgdx_sh = dpgdx_sh 
    self.dpcdx_sh = dpcdx_sh 
    self.dfcdx_sh = dfcdx_sh 
    self.fcsteady = fcsteady

    self.d2pcdx2_sh = d2pcdx2_sh 

    self.rho1, self.rho2 = rho1, rho2 
    self.pg1, self.pg2 = pg1, pg2 
    self.pc1, self.pc2 = pc1, pc2 
    self.cs1, self.cs2 = cs1, cs2 
    self.cc1, self.cc2 = cc1, cc2
    self.vs1, self.vs2 = vs1, vs2

    self.vsh = vsh 
    self.v1, self.v2 = v1, v2 
    self.fc1, self.fc2 = fc1, fc2 

    if self.cr_stream:
      self.va1, self.va2 = va1, va2 
      self.beta1, self.beta2 = beta1, beta2 

    self.vp1, self.vp2 = vp1, vp2 
    self.m1, self.m2 = m1, m2 
    self.n1, self.n2 = n1, n2 

    # Mark this function as run
    self.runshock = True
    return 

  def time_deriv(self):
    mas_flux = self.rho_sh*self.vx_sh 
    mom_flux = self.rho_sh*self.vx_sh**2 + self.pg_sh 
    eng_flux = (0.5*self.rho_sh*self.vx_sh**2 + gg1*self.pg_sh)*self.vx_sh 
    cre_flux = self.fc_sh 
    crf_flux = self.pc_sh 

    mas_div = np.gradient(mas_flux, self.x_sh)
    mom_div = np.gradient(mom_flux, self.x_sh)
    eng_div = np.gradient(eng_flux, self.x_sh)
    cre_div = np.gradient(cre_flux, self.x_sh)
    crf_div = np.gradient(crf_flux, self.x_sh)

    mom_sou = self.sc_sh*(self.fc_sh - gc1*self.pc_sh*self.vx_sh)
    eng_sou = (self.vx_sh + self.vs_sh)*self.sc_sh*(self.fc_sh - gc1*self.pc_sh*self.vx_sh)
    cre_sou = -(self.vx_sh + self.vs_sh)*self.sc_sh*(self.fc_sh - gc1*self.pc_sh*self.vx_sh)
    crf_sou = -self.sc_sh*(self.fc_sh - gc1*self.pc_sh*self.vx_sh)

    self.drhodt = -mas_div 
    self.dmomdt = mom_sou - mom_div 
    self.dengdt = eng_sou - eng_div 
    self.dcredt = cre_sou - cre_div 
    self.dcrfdt = (crf_sou - crf_div)*self.vmax**2

    # Mark as run
    self.runtime = True
    return 

  def plotshock(self):
    if self.runshock == False:
      self.shock()

    fig_width = 7
    golden_mean = (np.sqrt(5.) - 1.)/2. 
    fig_height = 5.5
    fig = plt.figure(figsize=(fig_width, fig_height))

    grids = gs.GridSpec(5, 1, figure=fig, hspace=0)
    ax1 = fig.add_subplot(grids[0, 0])
    ax2 = fig.add_subplot(grids[1, 0])
    ax3 = fig.add_subplot(grids[2, 0])
    ax4 = fig.add_subplot(grids[3, 0])
    ax5 = fig.add_subplot(grids[4, 0])

    ax1.plot(self.x_sh, self.dpcdx_sh, label='$\\nabla P_c$')

    ax2.semilogy(self.x_sh, self.sc_sh, label='$\\sigma_c$')

    ax3.plot(self.x_sh, self.fc_sh, 'b', label='$F_c$')
    ax3.plot(self.x_sh, self.fcsteady, 'r--', label='Steady $F_c$')

    ax4.plot(self.x_sh, self.rho_sh, label='$\\rho$')

    ax5.plot(self.x_sh, self.pc_sh, label='$P_c$')

    for axes in fig.axes:
      axes.axvline(self.x_sh[self.istr], linestyle='--', color='grey')
      axes.axvline(self.x_sh[self.ichk], linestyle='--', color='purple')
      axes.margins(x=0)
      if axes != ax2:
        axes.xaxis.set_minor_locator(AutoMinorLocator())
        axes.yaxis.set_minor_locator(AutoMinorLocator())

      if axes != ax4:
        axes.legend(frameon=False, fontsize=10)
      else:
        handles, labels = axes.get_legend_handles_labels()
        handles1, labels1 = handles[0:3], labels[0:3]
        handles2, labels2 = handles[3:-1], labels[3:-1]
        handles3, labels3 = [handles[-1]], [labels[-1]]
        axes.legend(handles1, labels1, frameon=False, loc='upper right', ncol=3, fontsize=12)
        leg2 = lg.Legend(axes, handles2, labels2, frameon=False, loc='upper left', ncol=3, fontsize=12)
        leg3 = lg.Legend(axes, handles3, labels3, frameon=False, loc='lower left', fontsize=12)
        axes.add_artist(leg2)
        axes.add_artist(leg3)

      if axes != ax5:
        axes.set_xticks([])
      else:
        axes.set_xlabel('$x$', fontsize=10)

      for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(10)

      fig.tight_layout()
      
    return fig

  def plotuov(self):
    uov_select = self.uov_array[self.file, :, :]

    fig_width = 7
    golden_mean = (np.sqrt(5.) - 1.)/2. 
    fig_height = 5.5
    fig = plt.figure(figsize=(fig_width, fig_height))

    grids = gs.GridSpec(5, 1, figure=fig, hspace=0)
    ax1 = fig.add_subplot(grids[0, 0])
    ax2 = fig.add_subplot(grids[1, 0])
    ax3 = fig.add_subplot(grids[2, 0])
    ax4 = fig.add_subplot(grids[3, 0])
    ax5 = fig.add_subplot(grids[4, 0])

    for i, axes in enumerate(fig.axes):
      axes.plot(self.x_sh, uov_select[i, self.ibeg:self.iend], label=self.uovname[i])
      axes.axvline(self.x_sh[self.istr], linestyle='--', color='grey')
      axes.axvline(self.x_sh[self.ichk], linestyle='--', color='purple')
      axes.margins(x=0)
      axes.xaxis.set_minor_locator(AutoMinorLocator())
      axes.yaxis.set_minor_locator(AutoMinorLocator())
      axes.legend(frameon=False, fontsize=10)

      if axes != ax5:
        axes.set_xticks([])
      else:
        axes.set_xlabel('$x$', fontsize=10)

      for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(10)

    fig.tight_layout()
    return fig

  def plottime(self):
    if self.runtime == False:
      self.time_deriv()

    fig_width = 7
    golden_mean = (np.sqrt(5.) - 1.)/2. 
    fig_height = 5.5
    fig = plt.figure(figsize=(fig_width, fig_height))

    grids = gs.GridSpec(5, 1, figure=fig, hspace=0)
    ax1 = fig.add_subplot(grids[0, 0])
    ax2 = fig.add_subplot(grids[1, 0])
    ax3 = fig.add_subplot(grids[2, 0])
    ax4 = fig.add_subplot(grids[3, 0])
    ax5 = fig.add_subplot(grids[4, 0])

    ax1.plot(self.x_sh, self.drhodt, label='$\\frac{\\mathrm{d}\\rho}{\\mathrm{d}t}$')
    ax2.plot(self.x_sh, self.dmomdt, label='$\\frac{\\mathrm{d}\\rho v}{\\mathrm{d}t}$')
    ax3.plot(self.x_sh, self.dengdt, label='$\\frac{\\mathrm{d}E}{\\mathrm{d}t}$')
    ax4.plot(self.x_sh, self.dcredt, label='$\\frac{\\mathrm{d}E_c}{\\mathrm{d}t}$')
    ax5.plot(self.x_sh, self.dcrfdt/self.vmax**2, label='$\\frac{1}{c^2}\\frac{\\mathrm{d}F_c}{\\mathrm{d}t}$')

    for axes in fig.axes:
      axes.axvline(self.x_sh[self.istr], linestyle='--', color='grey')
      axes.axvline(self.x_sh[self.ichk], linestyle='--', color='purple')
      axes.margins(x=0)
      axes.xaxis.set_minor_locator(AutoMinorLocator())
      axes.yaxis.set_minor_locator(AutoMinorLocator())
      axes.legend(frameon=False, fontsize=10)

      if axes != ax5:
        axes.set_xticks([])
      else:
        axes.set_xlabel('$x$', fontsize=10)

      for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(10)

    fig.tight_layout()
    return fig


  # Full equations
  def disper_full(self, k, c, rho, v, pg, pc, fc, sa, sd, va, drhodx, dvdx, dpgdx, dpcdx):
    sc = sa*sd/(sa + sd)

    disper = np.zeros((5, 5), dtype=complex)

    disper[0, 0] = 1j*dvdx 
    disper[0, 1] = k*rho + 1j*drhodx

    disper[1, 0] = 1j*(1./rho**2)*(sc*(fc - gc1*pc*v)*(1. - sc/(2.*sa)) - dpgdx)
    disper[1, 1] = 1j*dvdx + 1j*gc1*sc*pc/rho 
    disper[1, 2] = k/rho 
    disper[1, 3] = (sc/rho)*(1j*gc1*v - (sc/sa)*(fc - gc1*pc*v)*(k/dpcdx - 1j/pc))
    disper[1, 4] = -1j*sc/rho 

    disper[2, 0] = 1j*(gg - 1.)*va*sc*(fc - gc1*pc*v)*(sc/sd)*(1./(2.*rho))
    disper[2, 1] = gg*k*pg + 1j*dpgdx + 1j*(gg - 1.)*gc1*va*sc*pc 
    disper[2, 2] = 1j*gg*dvdx 
    disper[2, 3] = (gg - 1.)*va*sc*((sc/sa)*(fc - gc1*pc*v)*(1j/pc - k/dpcdx) + 1j*gc1*v)
    disper[2, 4] = -1j*(gg - 1.)*va*sc 

    disper[3, 0] = 1j*(gc - 1.)*sc*(fc - gc1*pc*v)*(1./(2.*rho))*sc*(v/sa - va/sd)
    disper[3, 1] = 1j*(gc - 1.)*sc*((fc - gc1*pc*v) - gc1*pc*(v + va))
    disper[3, 3] = -k*v + (gc - 1.)*(v + va)*sc*((sc/sa)*(fc - gc1*pc*v)*(k/dpcdx - 1j/pc) - 1j*gc1*v)
    disper[3, 4] = (gc - 1.)*(k + 1j*(v + va)*sc)

    disper[4, 0] = 1j*(1./(2.*rho))*(sc/sa)*sc*(fc - gc1*pc*v)*c**2
    disper[4, 1] = -1j*gc1*sc*pc*c**2 
    disper[4, 3] = (k + sc*((sc/sa)*(fc - gc1*pc*v)*(k/dpcdx - 1j/pc) - 1j*gc1*v))*c**2
    disper[4, 4] = -k*v + 1j*sc*c**2
    return disper 

  # Perturbed fully coupled equations
  def disper_coup(self, k, rho, v, pg, pc, va, sd, drhodx, dvdx, dpgdx, dpcdx):
    kap = (gc - 1.)/sd 

    disper = np.zeros((4, 4), dtype=complex)

    disper[0, 0] = 1j*dvdx 
    disper[0, 1] = k*rho + 1j*drhodx 

    disper[1, 0] = -1j*(1./rho**2)*(dpgdx + dpcdx)
    disper[1, 1] = 1j*dvdx
    disper[1, 2] = k/rho 
    disper[1, 3] = k/rho 

    disper[2, 0] = -1j*(gg - 1.)*va*dpcdx/(2.*rho) 
    disper[2, 1] = gg*k*pg + 1j*dpgdx 
    disper[2, 2] = 1j*gg*dvdx 
    disper[2, 3] = (gg - 1.)*k*va 

    disper[3, 0] = (va/(2.*rho))*(gc*pc*((3./(2.*rho))*1j*drhodx - k) - 1j*dpcdx)
    disper[3, 1] = gc*k*pc + 1j*dpcdx 
    disper[3, 3] = k*va + 1j*gc*(dvdx - (va*drhodx/(2.*rho))) + 1j*kap*k**2
    return disper 

  def grow(self, rho, v, pg, pc, va, sd, dpcdx, forward=True):
    direc = 1. if forward else -1.
    kap = (gc - 1.)/sd
    lc = -pc/dpcdx 
    cs = np.sqrt(gg*pg/rho)
    css = direc*cs
    cc = np.sqrt(gc*pc/rho)

    if 1./np.abs(lc) < tiny_number:
      return 0. 

    grad1 = 2.*((gg - 1.) - v/va)/(1. - v**2/cs**2)
    grad2 = 0.5*(gg - 1.)*(va/css)*(1. + grad1)
    gradient = (1. + grad2)/(gc*lc)
    
    ptus1 = css - 0.5*va 
    ptus2 = 1. + (gg - 1.)*(va/css) 
    ptuskin = ptus1*ptus2/kap 

    grow = -(cc**2/(2.*css))*(gradient + ptuskin)
    return grow 

  def instab(self, multiplier=1., full=True):
    if self.runshock == False:
      self.shock()

    low = float(input('Enter lower limit of instability check: '))
    up = float(input('Enter upper limit of instability check: '))
    il = np.argmin(np.abs(self.x_sh - low))
    iu = np.argmin(np.abs(self.x_sh - up))

    x_ins = self.x_sh[il:iu]
    rho_ins = self.rho_sh[il:iu]
    vx_ins = self.vx_sh[il:iu]
    pg_ins = self.pg_sh[il:iu]
    pc_ins = self.pc_sh[il:iu]
    fc_ins = self.fc_sh[il:iu]
    sa_ins = self.sa_sh[il:iu] 
    sd_ins = self.sd_sh[il:iu] 
    cs_ins = self.cs_sh[il:iu] 
    vs_ins = self.vs_sh[il:iu] 
    drhodx_ins = self.drhodx_sh[il:iu] 
    dvdx_ins = self.dvdx_sh[il:iu] 
    dpgdx_ins = self.dpgdx_sh[il:iu] 
    dpcdx_ins = self.dpcdx_sh[il:iu] 

    mode = 0 
    ldiff = self.kappa/cs_ins
    wvlen = multiplier*ldiff 
    k = 2.*np.pi/wvlen 
    c = self.vmax 

    grids = np.size(x_ins)
    growth = np.zeros(grids)
    analy_for = np.zeros(grids)
    analy_back = np.zeros(grids)
    freq = np.zeros(grids)
    index = np.zeros(grids, dtype=int) - 1

    for i, x in enumerate(x_ins):
      invlc = -dpcdx_ins[i]/pc_ins[i]
      if np.abs(invlc) > tiny_number:
        analy_for[i] = self.grow(rho_ins[i], vx_ins[i], pg_ins[i], pc_ins[i], vs_ins[i], sd_ins[i], \
          dpcdx_ins[i], forward=True)
        analy_back[i] = self.grow(rho_ins[i], vx_ins[i], pg_ins[i], pc_ins[i], vs_ins[i], sd_ins[i], \
          dpcdx_ins[i], forward=False)
        if full:
          w, v = LA.eig(self.disper_full(k[i], c, rho_ins[i], vx_ins[i], pg_ins[i], pc_ins[i], \
            fc_ins[i], sa_ins[i], sd_ins[i], vs_ins[i], drhodx_ins[i], \
            dvdx_ins[i], dpgdx_ins[i], dpcdx_ins[i]))
        else:
          w, v = LA.eig(self.disper_coup(k[i], rho_ins[i], vx_ins[i], pg_ins[i], pc_ins[i], \
            vs_ins[i], sd_ins[i], drhodx_ins[i], dvdx_ins[i], dpgdx_ins[i], dpcdx_ins[i]))
        sound = w[np.abs(np.abs(np.real(w))/(k[i]*cs_ins[i]) - 1.) < 0.8]
        if np.all(np.imag(sound) >= 0):
          growth[i] = 0.
          freq[i] = 0.
        else:
          index[i] = np.argsort(np.imag(sound))[mode]
          growth[i] = -np.imag(sound[index[i]])
          freq[i] = np.real(sound[index[i]])/(k[i]*cs_ins[i])
      else:
        growth[i] = 0. 
        freq[i] = 0.

    # Save variables
    self.low = low 
    self.up = up 
    self.il = il 
    self.iu = iu 
    self.x_ins = x_ins
    self.rho_ins = rho_ins
    self.vx_ins = vx_ins
    self.pg_ins = pg_ins
    self.pc_ins = pc_ins
    self.fc_ins = fc_ins
    self.sa_ins = sa_ins
    self.sd_ins = sd_ins
    self.cs_ins = cs_ins
    self.vs_ins = vs_ins
    self.drhodx_ins = drhodx_ins
    self.dvdx_ins = dvdx_ins
    self.dpgdx_ins = dpgdx_ins 
    self.dpcdx_ins = dpcdx_ins
    self.freq = freq # in units of k*cs
    self.growth = growth 
    self.analy_for = analy_for 
    self.analy_back = analy_back
    self.multiplier = multiplier
    self.wvlen = wvlen
    self.k = k

    # Mark this as run
    self.runinstab = True
    return

  def plotinstab(self):
    if self.runinstab == False:
      self.instab()

    x_ins = self.x_ins
    growth = self.growth 
    analy_for = self.analy_for 
    analy_back = self.analy_back
    freq = self.freq 
    signature = np.sign(freq)

    fig = plt.figure()

    # ax1: growth
    ax1 = fig.add_subplot(121)
    ax1.plot(x_ins, np.ma.masked_where(growth==0, growth), label='$\\lambda={0:.3f}\\kappa/c_s$'.format(self.multiplier))
    ax1.plot(x_ins, np.ma.masked_where(analy_for<=0, analy_for), '--')
    ax1.plot(x_ins, np.ma.masked_where(analy_back<=0, analy_back), '.-')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$\\Gamma$')
    ax1.margins(x=0)
    ax1.legend(frameon=False)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    # ax2: freq
    ax2 = fig.add_subplot(222)
    ax2.plot(x_ins, np.ma.masked_where(freq==0, np.abs(freq)), label='$\\lambda={0:.3f}\\kappa/c_s$'.format(self.multiplier))
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$\\omega/k c_s$')
    ax2.margins(x=0)
    ax2.legend(frameon=False)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    # ax3: direction
    ax3 = fig.add_subplot(224)
    ax3.plot(x_ins, np.ma.masked_where(signature==0, signature), label='$\\lambda={0:.3f}\\kappa/c_s$'.format(self.multiplier))
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('Direction')
    ax3.margins(x=0)
    ax3.legend(frameon=False)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout()
    return fig
# End of class

####################################
inputfile = 'athinput.1dcr_shock'
file_array = np.array([0, 30, 35, 39]) 

one = Plot1d(inputfile, file_array)
fig = one.plot()
fig.savefig('./1dinstab_plot', dpi=300)
plt.show(fig)

one.shock()
shkfig = one.plotshock()
shkfig.savefig('./1dinstab_shock', dpi=300)
plt.show(shkfig)

with h5py.File('../analytic/shock.hdf5', 'w') as fp:
  dset = fp.create_dataset('x', data=one.x_sh)
  dset = fp.create_dataset('rho', data=one.rho_sh)
  dset = fp.create_dataset('v', data=one.vx_sh)
  dset = fp.create_dataset('pg', data=one.pg_sh)
  dset = fp.create_dataset('pc', data=one.pc_sh)
  dset = fp.create_dataset('fc', data=one.fc_sh)

# if one.cr_stream:
#   one.instab(multiplier=0.1, full=False)
#   instabfig = one.plotinstab()
#   instabfig.savefig('./1dinstab_instab', dpi=300)
#   plt.show(instabfig)

# one.time_deriv()
# timefig = one.plottime()
# timefig.savefig('./1dinstab_time', dpi=300)
# plt.show(timefig)

# if one.user_output:
#   uovfig = one.plotuov()
#   uovfig.savefig('./1dinstab_uov', dpi=300)
#   plt.show(uovfig)

plt.close('all')