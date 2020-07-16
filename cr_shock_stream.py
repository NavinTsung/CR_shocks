# Import packages
import numpy as np 
import numpy.linalg as LA
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gs
import matplotlib.legend as lg
from matplotlib.ticker import AutoMinorLocator
import scipy.optimize as opt
import scipy.integrate as integrate 
import h5py

# Matplotlib default param
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 2.
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

gamma_g = 5./3.
gamma_c = 4./3.

def plotdefault():
  plt.rcParams.update(plt.rcParamsDefault)
  plt.rcParams['font.size'] = 12
  plt.rcParams['legend.fontsize'] = 12
  plt.rcParams['legend.loc'] = 'best'
  plt.rcParams['lines.linewidth'] = 1.5
  plt.rcParams['lines.markersize'] = 2.
  plt.rcParams['mathtext.fontset'] = 'stix'
  plt.rcParams['font.family'] = 'STIXGeneral'
  return

def latexify(columns=2):
  """
  Set up matplotlib's RC params for LaTeX plotting.
  Call this before plotting a figure.
  Parameters
  ----------
  columns : {1, 2}
  """
  assert(columns in [1, 2])

  fig_width_pt = 240.0 if (columns == 1) else 504.0
  inches_per_pt = 1./72.27 # Convert pt to inch
  golden_mean = (np.sqrt(5.) - 1.)/2. 
  fig_width = fig_width_pt*inches_per_pt # Width in inches
  fig_height = fig_width*golden_mean # Height in inches
  fig_size = [fig_width, fig_height]

  font_size = 10 if columns == 1 else 8

  plt.rcParams['pdf.fonttype'] = 42
  plt.rcParams['ps.fonttype'] = 42
  plt.rcParams['font.size'] = font_size
  plt.rcParams['axes.labelsize'] = font_size
  plt.rcParams['axes.titlesize'] = font_size
  plt.rcParams['xtick.labelsize'] = font_size
  plt.rcParams['ytick.labelsize'] = font_size
  plt.rcParams['legend.fontsize'] = font_size
  plt.rcParams['figure.figsize'] = fig_size
  plt.rcParams['figure.titlesize'] = 12
  return 

# Returns upstream rho, v, pg, pc
def mnbeta_to_gas(rho, pg, m, n, beta):
  B = np.sqrt(2.*pg/beta)
  pc = pg*n/(1. - n)
  cs = np.sqrt(gamma_g*pg/rho)
  ac = np.sqrt(gamma_c*pc/rho)
  va = B/np.sqrt(rho)
  # Evaluate for v
  p0 = 1. 
  p1 = -va 
  p2 = -m**2*(cs**2 + ac**2)
  p3 = m**2*va*(cs**2 - (gamma_g - 1.5)*ac**2)
  p4 = 0.5*(gamma_g - 1.)*(m*ac*va)**2
  roots = np.roots([p0, p1, p2, p3, p4])
  real_roots = np.real(roots[np.isreal(roots)])
  v = real_roots[real_roots > va][0] # select the superalfvenic root
  convert = {} 
  convert['rho'] = rho 
  convert['v'] = v 
  convert['pg'] = pg 
  convert['pc'] = pc 
  convert['B'] = B 
  return convert 

def gas_to_mnbeta(rho, pg, v, pc, B):
  beta = 2.*pg/B**2
  n = pc/(pc + pg)
  cs = np.sqrt(gamma_g*pg/rho)
  ac = np.sqrt(gamma_c*pc/rho)
  va = B/np.sqrt(rho)
  vp = np.sqrt(cs**2 + ac**2*(v - va/2.)*(v + (gamma_g - 1.)*va)/(v*(v - va)))
  m = v/vp
  convert = {}
  convert['rho'] = rho 
  convert['pg'] = pg 
  convert['m'] = m 
  convert['n'] = n 
  convert['beta'] = beta
  return convert

class Shock:
  def __init__(self, rho, pg, v, pc, B, kappa):
    self.rho = rho 
    self.v = v
    self.pg = pg 
    self.pc = pc 
    self.B = B
    self.kappa = kappa

    # Secondary variables
    self.cs = np.sqrt(gamma_g*pg/rho)
    self.ac = np.sqrt(gamma_c*pc/rho)
    self.va = B/np.sqrt(rho)
    self.ms = v/self.cs
    self.mc = v/self.ac 
    self.ma = v/self.va 
    self.fc = (gamma_c/(gamma_c - 1.))*pc*(v - self.va)

    va = self.va 
    ms = self.ms 
    mc = self.mc
    ma = self.ma
    fc = self.fc 

    # Conserved quantities
    self.J = rho*v 
    self.M = self.J*v + pg + pc 
    self.E = 0.5*self.J*v**2 + (gamma_g/(gamma_g - 1.))*pg*v + fc 
    self.S = (pg + (gamma_g - 1.)*B**2*(2*gamma_g*ma + 1. - gamma_g)/(gamma_g*(2.*gamma_g + 1.)))*np.cbrt((ma/(gamma_g - 1.) + 1.)**(10))

    # Parameters for plotting the shock diagram
    self.asymp = (gamma_c*(gamma_g - 1.)/((gamma_g - gamma_c)*ma))**2
    self.v_pgzero = self.M/self.J 
    self.v_pczero = gamma_g*self.M/((gamma_g + 1.)*self.J)
    self.v_hugzeros = self.hugzeros()
    self.v_hug2zeros = self.hug2zeros()
    self.regime = self.regime_num()
    self.v_hugpczeros = self.hugpczeros() # Does not necessarily exist
    self.v_hug2pczeros = self.hug2pczeros() # Does not necessarily exist
    self.v_adpcmaxzeros = self.adpcmaxzeros() 
    self.v_adhugzeros = self.adhugzeros()
    self.v_adhug2zeros = self.adhug2zeros()
    self.v_hugpcmaxzeros = self.hugpcmaxzeros() # Does not necessarily exist
    self.v_hug2pcmaxzeros = self.hug2pcmaxzeros() # Does not necessarily exist
    refhugon = self.refhugcoeff() # Does not necessarily exist
    self.coeff = refhugon['coeff']
    self.v_refpczero = refhugon['y']*v if np.size(refhugon['y'] > 0) else refhugon['y']
    self.v_ver = refhugon['ver']*v
    self.use_refhug = self.use_refhugon()
    ref2hugon = self.ref2hugcoeff() # Does not necessarily exist
    self.coeff2 = ref2hugon['coeff']
    self.v_ref2pczero = ref2hugon['y']*v if np.size(ref2hugon['y'] > 0) else ref2hugon['y']
    self.v_ver2 = ref2hugon['ver']*v
    self.use_ref2hug = self.use_ref2hugon()
    adrefhug = self.adrefhugzeros()
    self.v_adrefhugzeros = adrefhug['v_adrefhugzeros'] # Does not necessarily exist
    self.v_final = adrefhug['v_final']
    self.runprofile = False
    adref2hug = self.adref2hugzeros()
    self.v_adref2hugzeros = adref2hug['v_adrefhugzeros'] # Does not necessarily exist
    self.v_final2 = adref2hug['v_final']
    self.runprofile2 = False
    # End of initialization

  def find_root(self, lower, upper, func, asymptote=None):
    bin_number = 1000
    if asymptote == None:
      x = np.linspace(lower, upper, bin_number)
      y = func(x)
      if np.isscalar(y):
        y = np.array([y])
      sol_sign = np.sign(y)
      sol_sign_change = np.abs(np.diff(sol_sign))
      sol_loc = np.nonzero(sol_sign_change)[0]
      num_sol = np.size(sol_loc)
      x_sol = np.zeros(num_sol)
      for i in np.arange(num_sol):
        x_sol[i] = opt.brentq(func, x[sol_loc[i]], x[sol_loc[i]+1])
      return x_sol
    else:
      x_sol = np.array([])
      asymptote = asymptote[(asymptote > lower) & (asymptote < upper)]
      if np.size(asymptote) == 0:
        x = np.linspace(lower, upper, bin_number) 
        y = func(x)
        if np.isscalar(y):
          y = np.array([y])
        sol_sign = np.sign(y) 
        sol_sign_change = np.abs(np.diff(sol_sign))
        sol_loc = np.nonzero(sol_sign_change)[0]
        num_sol = np.size(sol_loc)
        x_solute = np.zeros(num_sol)
        for j in np.arange(num_sol):
          x_solute[j] = opt.brentq(func, x[sol_loc[j]], x[sol_loc[j]+1])
        x_sol = np.append(x_sol, x_solute)
      else:
        section = np.sort(np.append(np.array([lower, upper]), asymptote))
        for i in np.arange(np.size(section) - 1):
          if i == 0:
            x = np.linspace(lower, 0.9999*section[i+1], bin_number)
          elif (i == np.size(section) - 2):
            x = np.linspace(1.0001*section[i], upper, bin_number)
          else:
            x = np.linspace(1.0001*section[i], 0.9999*section[i+1], bin_number)
          y = func(x)
          if np.isscalar(y):
            y = np.array([y])
          sol_sign = np.sign(y) 
          sol_sign_change = np.abs(np.diff(sol_sign))
          sol_loc = np.nonzero(sol_sign_change)[0]
          num_sol = np.size(sol_loc)
          x_solute = np.zeros(num_sol)
          for j in np.arange(num_sol):
            x_solute[j] = opt.brentq(func, x[sol_loc[j]], x[sol_loc[j]+1])
          x_sol = np.append(x_sol, x_solute)
      return x_sol

  def hugzeros(self):
    asymp = self.asymp
    lower = 0.01
    upper = self.v_pgzero/self.v 
    hug = lambda y: self.hugoniot(y)
    y_sol = self.find_root(lower, upper, hug, asymptote=asymp)
    return y_sol*self.v # Returns the hugoniot zeros

  def hug2zeros(self):
    lower = 0.01
    upper = self.v_pgzero/self.v 
    hug2 = lambda y: self.hugoniot2(y)
    y_sol = self.find_root(lower, upper, hug2)
    return y_sol*self.v # Returns the hugoniot2 zeros

  def regime_num(self):
    vz = self.v_hugzeros
    v = self.v
    asymp = self.asymp
    if np.size(vz) == 2:
      if (vz[0]/v > asymp) and (vz[-1]/v > asymp): # Asymptote left of the two hugzeros
        return 1
      elif (vz[0]/v < asymp) and (vz[-1]/v > asymp): # Asymptote in the middle of the two hugzeros
        return 2
      elif (vz[0]/v < asymp) and (vz[-1]/v < asymp): # Asymptote right of the two hugzeros
        return 3
      else:
        raise ValueError('Not valid regime')
    elif np.size(vz) in [0, 1]: # Only one hugzero
      return 4
    else:
      raise ValueError('There is {} Hugoniot zeros!'.format(np.size(vz)))
      return

  def hugpczeros(self):
    asymp = self.asymp
    lower = 0.01  
    upper = self.v_pgzero/self.v
    hugpc = lambda y: self.hugoniot(y) - self.pczero(y)
    y_sol = self.find_root(lower, upper, hugpc, asymptote=asymp)
    return y_sol*self.v

  def hug2pczeros(self):
    lower = 0.01  
    upper = self.v_pgzero/self.v
    hug2pc = lambda y: self.hugoniot2(y) - self.pczero(y)
    y_sol = self.find_root(lower, upper, hug2pc)
    return y_sol*self.v

  def adpcmaxzeros(self):
    lower = 0.01 
    upper = self.v_pczero/self.v 
    adpcmax = lambda y: self.adiabat(y) - self.pcmax(y)
    y_sol = self.find_root(lower, upper, adpcmax)
    return y_sol*self.v # Returns the adiabat-pc max intersection

  def adhugzeros(self):
    asymp = self.asymp
    lower = 0.01
    upper = 0.99 
    adhug = lambda y: self.adiabat(y) - self.hugoniot(y)
    y_sol = self.find_root(lower, upper, adhug, asymptote=asymp)
    delete_index = np.array([], dtype=int)
    for i, ys in enumerate(y_sol):
      ps = self.adiabat(y_sol[i])
      pcap = self.pczero(y_sol[i])
      if (ps > pcap):
        delete_index = np.append(delete_index, i)
    y_sol = np.delete(y_sol, delete_index)
    return y_sol*self.v # Returns the adiabat-hugoniot intersection

  def adhug2zeros(self):
    lower = 0.01
    upper = 0.99 
    adhug2 = lambda y: self.adiabat(y) - self.hugoniot2(y)
    y_sol = self.find_root(lower, upper, adhug2)
    delete_index = np.array([], dtype=int)
    for i, ys in enumerate(y_sol):
      ps = self.adiabat(y_sol[i])
      pcap = self.pczero(y_sol[i])
      if (ps > pcap):
        delete_index = np.append(delete_index, i)
    y_sol = np.delete(y_sol, delete_index)
    return y_sol*self.v # Returns the adiabat-hugoniot2 intersection

  def hugpcmaxzeros(self):
    asymp = self.asymp
    lower = self.v_hugpczeros[0]/self.v if (self.regime in [2, 3, 4]) else self.v_hugzeros[0]/self.v
    upper = self.v_pczero/self.v 
    hugpcmax = lambda y: self.hugoniot(y) - self.pcmax(y)
    y_sol = self.find_root(lower, upper, hugpcmax, asymptote=asymp)
    return y_sol*self.v # Returns the hugoniot-pc max intersection

  def hug2pcmaxzeros(self):
    lower = self.v_hug2zeros[0]/self.v
    upper = self.v_pczero/self.v 
    hug2pcmax = lambda y: self.hugoniot2(y) - self.pcmax(y)
    y_sol = self.find_root(lower, upper, hug2pcmax)
    return y_sol*self.v # Returns the hugoniot2-pc max intersection

  def use_refhugon(self):
    if np.size(self.v_hugpcmaxzeros) != 0:
      check = self.pcmax(self.v_hugpcmaxzeros[0]/self.v)
      refchk1 = self.refhug(self.v_hugpcmaxzeros[0]/self.v)
      refchk2 = self.refhug2(self.v_hugpcmaxzeros[0]/self.v)
      use_refhug = True if (np.abs(refchk1 - check) < np.abs(refchk2 - check)) else False
    else:
      use_refhug = False
    return use_refhug # Determins whether to use refhug (False) or refhug2 (True) arc of the reflected hugoniot 

  def use_ref2hugon(self):
    if np.size(self.v_hug2pcmaxzeros) != 0:
      check = self.pcmax(self.v_hug2pcmaxzeros[0]/self.v)
      refchk1 = self.ref2hug(self.v_hug2pcmaxzeros[0]/self.v)
      refchk2 = self.ref2hug2(self.v_hug2pcmaxzeros[0]/self.v)
      use_refhug2 = True if (np.abs(refchk1 - check) < np.abs(refchk2 - check)) else False
    else:
      use_refhug2 = False
    return use_refhug2 # Determins whether to use the ref2hug (False) or ref2hug2 (True) arc of the reflected hugoniot2 

  def adrefhugzeros(self):
    asymp = self.asymp
    if np.size(self.v_hugpcmaxzeros) == 0: # The reflected hugoniot doesn't exist
      out = {}
      out['v_adrefhugzeros'] = np.array([])
      out['v_final'] = np.array([])
      return out 
    else: 
      lower  = self.v_hugpcmaxzeros[0]/self.v 
      upper = self.v_refpczero/self.v 
      if np.size(self.v_ver) != 0:
        lower2 = self.v_refpczero/self.v 
        upper = self.v_ver[0]/self.v 
      if self.use_refhug:
        adrefhug = lambda y: self.adiabat(y) - self.refhug(y)
      else:
        adrefhug = lambda y: self.adiabat(y) - self.refhug2(y)
      y_sol = self.find_root(lower, upper, adrefhug)
      if np.size(self.v_ver) != 0:
        if self.use_refhug:
          adrefhug2 = lambda y: self.adiabat(y) - self.refhug2(y)
        else:
          adrefhug2 = lambda y: self.adiabat(y) - self.refhug(y)
        y_sol2 = self.find_root(lower2, upper, adrefhug2)
        # Filter upper arc intersections
        delete_index = np.array([], dtype=int)
        for i, ys in enumerate(y_sol2):
          p_roots = self.adiabat(ys)
          p_capzero = (self.M - self.J*self.v*ys)/self.pg
          p_capmax = (self.J*self.v/(gamma_g*self.pg))*ys
          if (p_roots < 0) or (p_roots > p_capzero) or (p_roots > p_capmax):
            delete_index = np.append(delete_index, i)
        y_sol2 = np.delete(y_sol2, delete_index) if np.size(delete_index) != 0 else y_sol2
        y_sol = np.append(y_sol, y_sol2)
      out = {}
      out['v_adrefhugzeros'] = y_sol*self.v 
      out['v_final'] = np.array([])
      asymp = (gamma_c*(gamma_g - 1.)/((gamma_g - gamma_c)*self.ma))**2
      if np.size(y_sol) != 0:
        for i, ys in enumerate(y_sol):
          ps = self.adiabat(ys)
          pcs = self.M - self.J*self.v*ys - self.pg*ps 
          pc_curve = lambda y: (self.M - pcs - self.J*self.v*y)/self.pg 
          lower_fin = self.v_hugpczeros[0]/self.v if (self.regime in [2, 3, 4]) else self.v_hugpcmaxzeros[0]/self.v
          upper_fin = self.v_hugpcmaxzeros[0]/self.v if (self.regime in [2, 3, 4]) else self.v_pczero/self.v
          hugpc = lambda y: self.hugoniot(y) - pc_curve(y)
          y_solute = self.find_root(lower_fin, upper_fin, hugpc, asymptote=asymp)
          out['v_final'] = np.append(out['v_final'], y_solute*self.v)
    return out # Returns the adiabat-reflect hugoniot intersection and final state

  def adref2hugzeros(self):
    if np.size(self.v_hug2pcmaxzeros) == 0: # The reflected hugoniot doesn't exist
      out = {}
      out['v_adrefhugzeros'] = np.array([])
      out['v_final'] = np.array([])
      return out 
    else: 
      lower  = self.v_hug2pcmaxzeros[0]/self.v 
      upper = self.v_ref2pczero/self.v 
      if np.size(self.v_ver2) != 0:
        lower2 = self.v_ref2pczero/self.v 
        upper = self.v_ver2[0]/self.v 
      if self.use_ref2hug:
        adrefhug = lambda y: self.adiabat(y) - self.ref2hug(y)
      else:
        adrefhug = lambda y: self.adiabat(y) - self.ref2hug2(y)
      y_sol = self.find_root(lower, upper, adrefhug)
      if np.size(self.v_ver2) != 0:
        if self.use_ref2hug:
          adrefhug2 = lambda y: self.adiabat(y) - self.ref2hug2(y)
        else:
          adrefhug2 = lambda y: self.adiabat(y) - self.ref2hug(y)
        y_sol2 = self.find_root(lower2, upper, adrefhug2)
        # Filter upper arc intersections
        delete_index = np.array([], dtype=int)
        for i, ys in enumerate(y_sol2):
          p_roots = self.adiabat(ys)
          p_capzero = (self.M - self.J*self.v*ys)/self.pg
          p_capmax = (self.J*self.v/(gamma_g*self.pg))*ys
          if (p_roots < 0) or (p_roots > p_capzero) or (p_roots > p_capmax):
            delete_index = np.append(delete_index, i)
        y_sol2 = np.delete(y_sol2, delete_index) if np.size(delete_index) != 0 else y_sol2
        y_sol = np.append(y_sol, y_sol2)
      # Eliminate solution for which y < v_adrefhugzero/v
      delete_index2 = np.array([], dtype=int)
      for i, ys in enumerate(y_sol):
        if ys < self.v_adhugzeros[0]/self.v:
          delete_index2 = np.append(delete_index2, i)
      y_sol = np.delete(y_sol, delete_index2) if np.size(delete_index2) !=0 else y_sol 
      out = {}
      out['v_adrefhugzeros'] = y_sol*self.v 
      out['v_final'] = np.array([])
      if np.size(y_sol) != 0:
        for i, ys in enumerate(y_sol):
          ps = self.adiabat(ys)
          pcs = self.M - self.J*self.v*ys - self.pg*ps 
          pc_curve = lambda y: (self.M - pcs - self.J*self.v*y)/self.pg 
          lower_fin = self.v_hug2pcmaxzeros[0]/self.v
          upper_fin = self.v_pczero/self.v
          hugpc = lambda y: self.hugoniot2(y) - pc_curve(y)
          y_solute = self.find_root(lower_fin, upper_fin, hugpc)
          out['v_final'] = np.append(out['v_final'], y_solute*self.v)
    return out # Returns the adiabat-reflect hugoniot intersection and final state

  def pczero(self, y):
    J = self.J 
    M = self.M 
    v = self.v 
    pg = self.pg 
    return (M - J*v*y)/pg # Returns p-bar for the pc=0 curve

  def pcmax(self, y):
    J = self.J 
    v = self.v 
    pg = self.pg 
    return (J*v/(gamma_g*pg))*y # Returns p-bar for the pc max curve

  # Hugoniot curve for streaming upstream
  def hugoniot(self, y):
    ma = self.ma 
    ms = self.ms
    mc = self.mc 
    d = self.pc/self.pg 
    part1 = 0.5*(gamma_c + 1.)*(y - (gamma_c - 1.)/(gamma_c + 1.))
    part2 = (gamma_c/(gamma_g*ms**2))*(1. + d - (gamma_g - gamma_c)/(gamma_c*(gamma_g - 1.)*(1. - y)))
    part3 = (gamma_c/ma)*(np.sqrt(y) - 1./(gamma_c*mc**2*(1. + np.sqrt(y))) + np.sqrt(y)/(gamma_g*ms**2*(1. - y)))
    part4 = gamma_g*ms**2*(1. - y)/np.sqrt(y) 
    part5 = (gamma_g - gamma_c)*np.sqrt(y)/(gamma_g - 1.) - gamma_c/ma 
    return (part1 - part2 - part3)*part4/part5 # Returns p-bar for the hugoniot

  # Hugoniot curve for streaming downstream
  def hugoniot2(self, y):
    ma = self.ma 
    ms = self.ms
    mc = self.mc 
    d = self.pc/self.pg 
    part1 = 0.5*(gamma_c + 1.)*(y - (gamma_c - 1.)/(gamma_c + 1.))
    part2 = (gamma_c/(gamma_g*ms**2))*(1. + d - (gamma_g - gamma_c)/(gamma_c*(gamma_g - 1.)*(1. - y)))
    part3 = (gamma_c/ma)*(np.sqrt(y) - 1./(gamma_c*mc**2*(1. + np.sqrt(y))) + np.sqrt(y)/(gamma_g*ms**2*(1. - y)))
    part4 = gamma_g*ms**2*(1. - y)/np.sqrt(y) 
    part5 = (gamma_g - gamma_c)*np.sqrt(y)/(gamma_g - 1.) + gamma_c/ma 
    return (part1 - part2 + part3)*part4/part5 # Returns p-bar for the hugoniot

  # def hugoniot2(self, y):
  #   ms = self.ms
  #   d = self.pc/self.pg 
  #   part1 = gamma_c*(gamma_g - 1.)*(1. - y)/(gamma_g - gamma_c)
  #   part2 = 1. + d - 0.5*(gamma_c + 1.)*(gamma_g*ms**2/gamma_c)*(y - (gamma_c - 1.)/(gamma_c + 1.)) 
  #   return (1. - part1*part2)/y # Returns p-bar for the hugoniot

  def refhugcoeff(self):
    J = self.J 
    M = self.M 
    v = self.v 
    pg = self.pg
    asymp = self.asymp
    if np.size(self.v_hugpcmaxzeros) == 0: # No intersection or only one between the hugoniot and pc max
      print('There is no reflected Hugoniot')
      out = {}
      out['coeff'] = np.array([])
      out['y'] = np.array([])
      out['ver'] = np.array([])
      return out
    elif (np.size(self.v_hugpcmaxzeros) != 0) and (np.size(self.v_hugpczeros) == 0): # No intersection between the hugoniot and pc zero
      y_min = self.v_hugpcmaxzeros[-1]/v
      p_min = self.pcmax(y_min)
      pc_min = M - p_min*pg - J*y_min*v 
    else:
      y_min = self.v_hugpczeros[0]/v
      p_min = self.pczero(y_min)
      pc_min = M - p_min*pg - J*y_min*v

    y_max = self.v_hugpcmaxzeros[0]/v
    p_max = self.pcmax(y_max)
    pc_max = M - p_max*pg - J*y_max*v 

    # Identify 5 locations on the reflected Hugoniot
    pc_array = np.linspace(pc_min, pc_max, 5)
    y_array = np.zeros(5)
    p_array = np.zeros(5)
    lower  = 0.99*self.v_hugpczeros[0]/v if (self.regime in [2, 3, 4]) else self.v_hugzeros[0]/v
    upper = self.v_pczero/v
    for i, pc in enumerate(pc_array):
      yb = gamma_g*(M - pc)/((gamma_g + 1.)*J*v)
      pb = (M - pc)/((gamma_g + 1.)*pg)
      pc_curve = lambda u: (M - pc - J*v*u)/pg # Returns p-bar for the finite pc curve
      hugpccurve = lambda u: self.hugoniot(u) - pc_curve(u)
      y_sol = self.find_root(lower, upper, hugpccurve, asymptote=asymp)[0]
      p_sol = pc_curve(y_sol)
      d = np.sqrt(v**2*(y_sol - yb)**2 + pg**2*(p_sol - pb)**2)
      # Solve for the reflected hugoniot
      q0 = (J**2 + 1.)
      q1 = -2*((M - pc - pb*pg)*J + yb*v)
      q2 = (M - pc - pb*pg)**2 + (yb*v)**2 - d**2
      y_array[i] = np.amax(np.real(np.roots([q0, q1, q2])))/v
      p_array[i] = pc_curve(y_array[i])

    # Solve for the equation of the reflected Hugoniot
    # A hyperbola has general equation a1*y^2 + a2*y*P + a3*P^2 + a4*y + a5*P = 1 
    matrix = np.zeros((5, 5))
    matrix[:, 0] = y_array**2
    matrix[:, 1] = y_array*p_array 
    matrix[:, 2] = p_array**2
    matrix[:, 3] = y_array 
    matrix[:, 4] = p_array 
    mat_inv = LA.inv(matrix)
    coeff = np.sum(mat_inv, axis=1)

    # Calculate where the reflected hugoniot verticalize
    ver0 = (4.*coeff[0]*coeff[2] - coeff[1]**2)
    ver1 = (4.*coeff[2]*coeff[3] - 2.*coeff[1]*coeff[4])
    ver2 = -(coeff[4]**2 + 4.*coeff[2])
    ver_roots = np.roots([ver0, ver1, ver2])
    ver_roots = ver_roots[np.isreal(ver_roots)]
    ver_roots = ver_roots[(ver_roots > self.v_hugpcmaxzeros[0]/self.v) & (ver_roots < self.v_pgzero/v)]
    p_ver = -(coeff[1]*ver_roots + coeff[4])/(2.*coeff[2])
    p_cap = self.pczero(ver_roots)
    p_max = self.pcmax(ver_roots)
    delete_index = np.array([], dtype=int)
    for i, pe in enumerate(p_ver):
      if (pe < 0) or (pe > p_cap[i]) or (pe > p_max[i]):
        delete_index = np.append(delete_index, i)
    ver_out = np.delete(ver_roots, delete_index) if np.size(delete_index) != 0 else ver_roots

    # Output
    out = {}
    out['coeff'] = coeff 
    out['y'] = y_array[0] # Gives the reflect hugoniot-pc zero/ pc max intersection
    out['ver'] = ver_out 
    return out

  def ref2hugcoeff(self):
    J = self.J 
    M = self.M 
    v = self.v 
    pg = self.pg
    if np.size(self.v_hug2pcmaxzeros) == 0: # No intersection or only one between the hugoniot and pc max
      print('There is no reflected Hugoniot2')
      out = {}
      out['coeff'] = np.array([])
      out['y'] = np.array([])
      out['ver'] = np.array([])
      return out
    elif (np.size(self.v_hug2pcmaxzeros) != 0) and (np.size(self.v_hug2pczeros) == 0): # No intersection between the hugoniot and pc zero
      y_min = self.v_hug2pcmaxzeros[-1]/v
      p_min = self.pcmax(y_min)
      pc_min = M - p_min*pg - J*y_min*v 
    else:
      y_min = self.v_hug2pczeros[0]/v
      p_min = self.pczero(y_min)
      pc_min = M - p_min*pg - J*y_min*v

    y_max = self.v_hug2pcmaxzeros[0]/v
    p_max = self.pcmax(y_max)
    pc_max = M - p_max*pg - J*y_max*v 

    # Identify 5 locations on the reflected Hugoniot
    pc_array = np.linspace(pc_min, pc_max, 5)
    y_array = np.zeros(5)
    p_array = np.zeros(5)
    lower = self.v_hug2zeros[0]/v
    upper = self.v_pczero/v
    for i, pc in enumerate(pc_array):
      yb = gamma_g*(M - pc)/((gamma_g + 1.)*J*v)
      pb = (M - pc)/((gamma_g + 1.)*pg)
      pc_curve = lambda u: (M - pc - J*v*u)/pg # Returns p-bar for the finite pc curve
      hugpccurve = lambda u: self.hugoniot2(u) - pc_curve(u)
      y_sol = self.find_root(lower, upper, hugpccurve)[0]
      p_sol = pc_curve(y_sol)
      d = np.sqrt(v**2*(y_sol - yb)**2 + pg**2*(p_sol - pb)**2)
      # Solve for the reflected hugoniot
      q0 = (J**2 + 1.)
      q1 = -2*((M - pc - pb*pg)*J + yb*v)
      q2 = (M - pc - pb*pg)**2 + (yb*v)**2 - d**2
      y_array[i] = np.amax(np.real(np.roots([q0, q1, q2])))/v
      p_array[i] = pc_curve(y_array[i])

    # Solve for the equation of the reflected Hugoniot
    # A hyperbola has general equation a1*y^2 + a2*y*P + a3*P^2 + a4*y + a5*P = 1 
    matrix = np.zeros((5, 5))
    matrix[:, 0] = y_array**2
    matrix[:, 1] = y_array*p_array 
    matrix[:, 2] = p_array**2
    matrix[:, 3] = y_array 
    matrix[:, 4] = p_array 
    mat_inv = LA.inv(matrix)
    coeff = np.sum(mat_inv, axis=1)

    # Calculate where the reflected hugoniot verticalize
    ver0 = (4.*coeff[0]*coeff[2] - coeff[1]**2)
    ver1 = (4.*coeff[2]*coeff[3] - 2.*coeff[1]*coeff[4])
    ver2 = -(coeff[4]**2 + 4.*coeff[2])
    ver_roots = np.roots([ver0, ver1, ver2])
    ver_roots = ver_roots[np.isreal(ver_roots)]
    ver_roots = ver_roots[(ver_roots > self.v_hug2pcmaxzeros[0]/self.v) & (ver_roots < self.v_pgzero/v)]
    p_ver = -(coeff[1]*ver_roots + coeff[4])/(2.*coeff[2])
    p_cap = self.pczero(ver_roots)
    p_max = self.pcmax(ver_roots)
    delete_index = np.array([], dtype=int)
    for i, pe in enumerate(p_ver):
      if (pe < 0) or (pe > p_cap[i]) or (pe > p_max[i]):
        delete_index = np.append(delete_index, i)
    ver_out = np.delete(ver_roots, delete_index) if np.size(delete_index) != 0 else ver_roots

    # Output
    out = {}
    out['coeff'] = coeff 
    out['y'] = y_array[0] # Gives the reflect hugoniot-pc zero/ pc max intersection
    out['ver'] = ver_out 
    return out

  def refhug(self, y):
    coeff = self.coeff
    if np.size(coeff) == 0:
      return np.array([])
    else:
      r0 = coeff[2]
      r1 = coeff[1]*y + coeff[4]
      r2 = coeff[0]*y**2 + coeff[3]*y - 1.
      if np.isscalar(y) == False:
        p_out = np.zeros(np.size(y))
        for i in np.arange(np.size(y)):
          p_roots = np.roots([r0, r1[i], r2[i]])
          p_out[i] = np.amin(np.real(p_roots))
      else:
        p_roots = np.roots([r0, r1, r2])
        p_out = np.amin(np.real(p_roots))
      return p_out # Returns the lower arc of the reflected hugoniot

  def refhug2(self, y):
    coeff = self.coeff
    if np.size(coeff) == 0:
      return np.array([])
    else:
      r0 = coeff[2]
      r1 = coeff[1]*y + coeff[4]
      r2 = coeff[0]*y**2 + coeff[3]*y - 1.
      if np.size(y) > 1:
        p_out = np.zeros(np.size(y))
        for i in np.arange(np.size(y)):
          p_roots = np.roots([r0, r1[i], r2[i]])
          p_out[i] = np.amax(np.real(p_roots))
      else:
        p_roots = np.roots([r0, r1, r2])
        p_out = np.amax(np.real(p_roots))
      return p_out # Returns the upper arc of the reflected hugoniot, if there were any

  def ref2hug(self, y):
    coeff = self.coeff2
    if np.size(coeff) == 0:
      return np.array([])
    else:
      r0 = coeff[2]
      r1 = coeff[1]*y + coeff[4]
      r2 = coeff[0]*y**2 + coeff[3]*y - 1.
      if np.isscalar(y) == False:
        p_out = np.zeros(np.size(y))
        for i in np.arange(np.size(y)):
          p_roots = np.roots([r0, r1[i], r2[i]])
          p_out[i] = np.amin(np.real(p_roots))
      else:
        p_roots = np.roots([r0, r1, r2])
        p_out = np.amin(np.real(p_roots))
      return p_out # Returns the lower arc of the reflected hugoniot

  def ref2hug2(self, y):
    coeff = self.coeff2
    if np.size(coeff) == 0:
      return np.array([])
    else:
      r0 = coeff[2]
      r1 = coeff[1]*y + coeff[4]
      r2 = coeff[0]*y**2 + coeff[3]*y - 1.
      if np.size(y) > 1:
        p_out = np.zeros(np.size(y))
        for i in np.arange(np.size(y)):
          p_roots = np.roots([r0, r1[i], r2[i]])
          p_out[i] = np.amax(np.real(p_roots))
      else:
        p_roots = np.roots([r0, r1, r2])
        p_out = np.amax(np.real(p_roots))
      return p_out # Returns the upper arc of the reflected hugoniot, if there were any

  def adiabat(self, y):
    S = self.S 
    pg = self.pg 
    ma = self.ma 
    B = self.B
    part1 = S/np.cbrt((ma*np.sqrt(y)/(gamma_g - 1.) + 1.)**(10.))
    part2 = (gamma_g - 1.)*B**2*(2.*gamma_g*ma*np.sqrt(y) + 1. - gamma_g)/(gamma_g*(2.*gamma_g + 1.))
    return (part1 - part2)/pg # Returns p-bar for the adiabat

  # D function in Volk et al. 1984
  def D(self, y): 
    ms = self.ms
    ma = self.ma
    pbar = self.adiabat(y) 
    return (pbar/(y*ms**2) - 1.)/(1. + (gamma_g - 1.)/(ma*np.sqrt(y))) 

  # N function in Volk et al. 1984
  def N(self, y):
    ms = self.ms
    mc = self.mc 
    ma = self.ma 
    d = self.pc/self.pg 
    pbar = self.adiabat(y)
    part1 = 0.5*(gamma_c + 1.)*(y - (gamma_c - 1.)/(gamma_c + 1.))
    part2 = (gamma_c/(gamma_g*ms**2))*(1. + d - (gamma_g - gamma_c)*(1. - pbar*y)/(gamma_c*(gamma_g - 1.)*(1. - y)))
    part3 = (gamma_c/ma)*(np.sqrt(y) - 1./(gamma_c*mc**2*(1. + np.sqrt(y))) + np.sqrt(y)*(1. - pbar)/(gamma_g*ms**2*(1. - y)))
    return part1 - part2 - part3

  def solution(self):
    if np.size(self.v_final) != 0: 
      v2 = self.v_final 
      rho2 = self.J/v2
      pg2 = self.pg*self.hugoniot(v2/self.v)
      pc2 = self.M - self.J*v2 - pg2 
    elif np.size(self.v_adhugzeros) != 0:
      v2 = self.v_adhugzeros
      rho2 = self.J/v2 
      pg2 = self.pg*self.hugoniot(v2/self.v)
      pc2 = self.M - self.J*v2 - pg2 
    else:
      print('No solution uni-directional streaming')
      v2 = np.array([])
      rho2 = np.array([])
      pg2 = np.array([])
      pc2 = np.array([])
    # Store final solution
    downstream = {}
    downstream['rho'] = rho2 
    downstream['v'] = v2 
    downstream['pg'] = pg2 
    downstream['pc'] = pc2
    return downstream

  def solution2(self):
    if np.size(self.v_final2) != 0: 
      v2 = self.v_final2 
      rho2 = self.J/v2
      pg2 = self.pg*self.hugoniot2(v2/self.v)
      pc2 = self.M - self.J*v2 - pg2 
    elif (np.size(self.v_adhug2zeros) != 0) and (self.v_adhug2zeros[0] > self.v_adhugzeros[0]):
      v2 = self.v_adhug2zeros
      rho2 = self.J/v2 
      pg2 = self.pg*self.hugoniot2(v2/self.v)
      pc2 = self.M - self.J*v2 - pg2 
    else:
      print('No solution to bi-directional streaming')
      v2 = np.array([])
      rho2 = np.array([])
      pg2 = np.array([])
      pc2 = np.array([])
    # Store final solution
    downstream = {}
    downstream['rho'] = rho2 
    downstream['v'] = v2 
    downstream['pg'] = pg2 
    downstream['pc'] = pc2
    return downstream

  def shock_diagram(self, don_want_axis=True):
    # Plot shock diagram (Drury & Volk 1980)
    # The pc = 0 line
    y_pczero = np.linspace(0, self.v_pgzero/self.v, 1000)

    # The max pc line
    y_pcmax = np.linspace(0, self.v_pczero/self.v, 1000) 

    # The Hugoniot curve
    if (np.size(self.v_hugzeros) > 0):
      if self.regime == 1:
        y_hug = np.linspace(self.v_hugzeros[0]/self.v, self.v_hugzeros[-1]/self.v, 1000) 
      elif self.regime == 2:
        y_hug = np.linspace(0.01, 0.9999*self.asymp, 1000)
        y_hug2 = np.linspace(1.0001*self.asymp, self.v_hugzeros[-1]/self.v, 1000)
      elif self.regime == 3:
        y_hug = np.linspace(0.01, self.v_hugzeros[0]/self.v, 1000)
        y_hug2 = np.linspace(self.v_hugzeros[-1]/self.v, 0.9999*self.asymp, 1000)
      else:
        pass
    else:
      y_hug = np.linspace(0.01, 0.9999*self.asymp, 1000)

    if (np.size(self.v_hug2zeros) > 0):
      y_2hug = np.linspace(self.v_hug2zeros[0]/self.v, self.v_hug2zeros[-1]/self.v, 1000)

    # The adiabat
    if (np.size(self.v_adhugzeros) > 0):
      y_adia = np.linspace(self.v_adhugzeros[0]/self.v, 1., 1000)
    if (np.size(self.v_adhug2zeros) > 0) and (self.v_adhug2zeros[0] < self.v_adhugzeros[0]):
      y_adia2 = np.linspace(self.v_adhug2zeros[0]/self.v, 1., 1000)

    # The reflected hugoniot
    if (np.size(self.v_hugpcmaxzeros) > 0):
      if (np.size(self.v_ver) > 0):
          y_ref = np.linspace(self.v_hugpcmaxzeros[0]/self.v, self.v_ver[0]/self.v, 1000)
          y_ref2 = np.linspace(self.v_refpczero/self.v, self.v_ver[0]/self.v, 1000)
      else:
        y_ref = np.linspace(self.v_hugpcmaxzeros[0]/self.v, self.v_refpczero/self.v, 1000)

    if (np.size(self.v_hug2pcmaxzeros) > 0):
      if (np.size(self.v_ver2) > 0):
          y_2ref = np.linspace(self.v_hug2pcmaxzeros[0]/self.v, self.v_ver2[0]/self.v, 1000)
          y_2ref2 = np.linspace(self.v_ref2pczero/self.v, self.v_ver2[0]/self.v, 1000)
      else:
        y_2ref = np.linspace(self.v_hug2pcmaxzeros[0]/self.v, self.v_ref2pczero/self.v, 1000)

    # Plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(self.v*y_pczero, self.pg*self.pczero(y_pczero), label='$P_c=0$')
    ax.plot(self.v*y_pcmax, self.pg*self.pcmax(y_pcmax), label='$P_{c,\\mathrm{max}}$')
  
    if self.regime in [1, 4]:
      ax.plot(self.v*y_hug, self.pg*self.hugoniot(y_hug), 'k', label='Hugoniot')
    elif self.regime in [2, 3]:
      ax.plot(self.v*y_hug, self.pg*self.hugoniot(y_hug), 'k', label='Hugoniot')
      ax.plot(self.v*y_hug2, self.pg*self.hugoniot(y_hug2), 'k')
    if (np.size(self.v_hug2zeros) > 0):
      ax.plot(self.v*y_2hug, self.pg*self.hugoniot2(y_2hug), 'k--', label='Hugoniot2')
    
    if (np.size(self.v_adhugzeros) > 0):
      ax.plot(self.v*y_adia, self.pg*self.adiabat(y_adia), color='tab:green', label='Adiabat')
      ax.scatter(self.v_adhugzeros, self.pg*self.adiabat(self.v_adhugzeros/self.v), marker='o', color='k')
    if (np.size(self.v_adhug2zeros) > 0) and (self.v_adhug2zeros[0] < self.v_adhugzeros[0]):
      ax.plot(self.v*y_adia2, self.pg*self.adiabat(y_adia2), color='tab:green', label='Adiabat')
      ax.scatter(self.v_adhug2zeros, self.pg*self.adiabat(self.v_adhug2zeros/self.v), marker='o', color='k')
    
    if (np.size(self.v_adrefhugzeros) > 0) and (np.size(self.v_final) > 0):
      for i, v_adrefhug in enumerate(self.v_adrefhugzeros):
        ax.plot(np.array([v_adrefhug, self.v_final[i]]), np.array([self.pg*self.adiabat(v_adrefhug/self.v), self.pg*self.hugoniot(self.v_final[i]/self.v)]), color='tab:red') 
      ax.scatter(self.v_adrefhugzeros, self.pg*self.adiabat(self.v_adrefhugzeros/self.v), marker='o', color='k')
      ax.scatter(self.v_final, self.pg*self.hugoniot(self.v_final/self.v), marker='o', color='k')
    if (np.size(self.v_adref2hugzeros) > 0) and (np.size(self.v_final2) > 0):
      for i, v_adref2hug in enumerate(self.v_adref2hugzeros):
        ax.plot(np.array([v_adref2hug, self.v_final2[i]]), np.array([self.pg*self.adiabat(v_adref2hug/self.v), self.pg*self.hugoniot2(self.v_final2[i]/self.v)]), '--', color='tab:red') 
      ax.scatter(self.v_adref2hugzeros, self.pg*self.adiabat(self.v_adref2hugzeros/self.v), marker='o', color='k')
      ax.scatter(self.v_final2, self.pg*self.hugoniot2(self.v_final2/self.v), marker='o', color='k')
    
    if (np.size(self.v_hugpcmaxzeros) > 0):
      if (np.size(self.v_ver) > 0):
        if self.use_refhug:
          ax.plot(self.v*np.append(y_ref, y_ref2[::-1]), self.pg*np.append(self.refhug(y_ref), self.refhug2(y_ref2[::-1])), color='tab:purple', label='Reflect')
        else:
          ax.plot(self.v*np.append(y_ref, y_ref2[::-1]), self.pg*np.append(self.refhug2(y_ref), self.refhug(y_ref2[::-1])), color='tab:purple', label='Reflect')
      else:
        if self.use_refhug:
          ax.plot(self.v*y_ref, self.pg*self.refhug(y_ref), color='tab:purple', label='Reflect')
        else:
          ax.plot(self.v*y_ref, self.pg*self.refhug2(y_ref), color='tab:purple', label='Reflect')

    if (np.size(self.v_hug2pcmaxzeros) > 0):
      if (np.size(self.v_ver2) > 0):
        if self.use_ref2hug:
          ax.plot(self.v*np.append(y_2ref, y_2ref2[::-1]), self.pg*np.append(self.ref2hug(y_2ref), self.ref2hug2(y_2ref2[::-1])), '--', color='tab:purple', label='Reflect2')
        else:
          ax.plot(self.v*np.append(y_2ref, y_2ref2[::-1]), self.pg*np.append(self.ref2hug2(y_2ref), self.ref2hug(y_2ref2[::-1])), '--', color='tab:purple', label='Reflect2')
      else:
        if self.use_ref2hug:
          ax.plot(self.v*y_2ref, self.pg*self.ref2hug(y_2ref), '--', color='tab:purple', label='Reflect2')
        else:
          ax.plot(self.v*y_2ref, self.pg*self.ref2hug2(y_2ref), '--', color='tab:purple', label='Reflect2')

    ax.scatter(self.v, self.pg, marker='o', color='k')
    
    ax.set_xlim(0, self.v_pgzero)
    ax.set_ylim(0, self.M)
    ax.set_xlabel('$v$')
    ax.set_ylabel('$P_g$')
    ax.margins(x=0, y=0)
    # ax.legend(frameon=False)

    ax.set_xticks([])
    ax.set_yticks([])

    if don_want_axis:
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

    return fig

  def profile(self, old_solution=True, mode=0): # In case of multiple solution, mode determines (starting from 0), in order of decreasing Pc, the displayed solution
    ldiff = self.kappa/self.v 
    if old_solution:
      vf = self.v_adrefhugzeros[mode] if np.size(self.v_adrefhugzeros) != 0 else self.v_adhugzeros[0]
    else:
      vf = self.v_adref2hugzeros[mode] if np.size(self.v_adref2hugzeros) != 0 else self.v_adhug2zeros[0] 

    if np.size(vf) == 0:
      print('No solution')
      return 

    int_bin = 5000
    yf = vf/self.v
    y_int = np.linspace(0.9999999999, 1.000000000000001*yf, int_bin)
    dxdy = lambda y, x: ldiff*self.D(y)/((1. - y)*self.N(y))
    sol = integrate.solve_ivp(dxdy, [y_int[0], yf], [0.], t_eval=y_int) # 0.9999 to ensure t_span encompasses y_int
    x_int = sol.y[0]

    rho_int = self.J/(self.v*y_int) 
    v_int = self.v*y_int
    pg_int = self.pg*np.array([self.adiabat(y) for i, y in enumerate(y_int)])
    pc_int = self.M - self.J*self.v*y_int - pg_int 
    va_int = self.B/np.sqrt(rho_int)
    fc_int = self.E - 0.5*self.J*(self.v*y_int)**2 - (gamma_g/(gamma_g - 1.))*pg_int*self.v*y_int
    ma_int = v_int/(self.B/np.sqrt(rho_int))
    wave_int = (1. + ma_int/(gamma_g - 1.))**(2.*gamma_g)*(pg_int + (gamma_g - 1.)*self.B**2*(2*gamma_g*ma_int + 1. - gamma_g)/(gamma_g*(2.*gamma_g + 1.)))
    dpcdx_int = ((gamma_c/(gamma_c - 1.))*pc_int*(v_int - va_int) - fc_int)*(gamma_c - 1.)/self.kappa
    sa_int = ((gamma_c - 1.)/gamma_c)*np.abs(dpcdx_int)/(pc_int*va_int)
    sd_int = (gamma_c - 1.)/self.kappa 
    sc_int = sa_int*sd_int/(sa_int + sd_int)

    # Append final solution if necessary
    if old_solution and (np.size(self.v_adrefhugzeros) != 0):
      v2 = self.v_final[mode]
      rho2 = self.J/v2
      pg2 = self.pg*self.hugoniot(v2/self.v)
      pc2 = self.M - self.J*v2 - pg2 
      va2 = self.B/np.sqrt(rho2)

      x_int = np.append(x_int, x_int[-1])
      v_int = self.v*np.append(y_int, v2/self.v)
      rho_int = np.append(rho_int, rho2)
      pg_int = np.append(pg_int, pg2)
      pc_int = np.append(pc_int, pc2)
      fc_int = np.append(fc_int, fc_int[-1])
      ma2 = v2/(self.B/np.sqrt(rho2))
      wave2 = (1. + ma2/(gamma_g - 1.))**(2.*gamma_g)*(pg2 + (gamma_g - 1.)*self.B**2*(2*gamma_g*ma2 + 1. - gamma_g)/(gamma_g*(2.*gamma_g + 1.)))
      wave_int = np.append(wave_int, wave2)
      dpcdx_int = np.append(dpcdx_int, 0.)
      sa_int = np.append(sa_int, 0.)
      sc_int = np.append(sc_int, 0.)
    elif (old_solution == False) and (np.size(self.v_adref2hugzeros) != 0):
      v2 = self.v_final2[mode]
      rho2 = self.J/v2
      pg2 = self.pg*self.hugoniot2(v2/self.v)
      pc2 = self.M - self.J*v2 - pg2 
      va2 = self.B/np.sqrt(rho2)

      x_int = np.append(x_int, x_int[-1])
      v_int = self.v*np.append(y_int, v2/self.v)
      rho_int = np.append(rho_int, rho2)
      pg_int = np.append(pg_int, pg2)
      pc_int = np.append(pc_int, pc2)
      fc_int = np.append(fc_int, fc_int[-1])
      ma2 = v2/(self.B/np.sqrt(rho2))
      wave2 = np.cbrt((1. - ma2/(gamma_g - 1.))**(10))*(pg2 + (gamma_g - 1.)*self.B**2*(-2*gamma_g*ma2 + 1. - gamma_g)/(gamma_g*(2.*gamma_g + 1.)))
      wave_int = np.append(wave_int, wave2)
      dpcdx_int = np.append(dpcdx_int, 0.)
      sa_int = np.append(sa_int, 0.)
      sc_int = np.append(sc_int, 0.)

    # Save to memory
    self.x_int = x_int
    self.v_int = v_int 
    self.rho_int = rho_int 
    self.pg_int = pg_int 
    self.pc_int = pc_int 
    self.fc_int = fc_int
    self.wave_int = wave_int
    self.dpcdx_int = dpcdx_int 
    self.sa_int = sa_int 
    self.sd_int = sd_int 
    self.sc_int = sc_int

    # Mark as run
    if old_solution:
      self.runprofile = True
    else:
      self.runprofile2 = True
    return

  # Plot shock profile
  def plotprofile(self, compare=None, old_solution=True, mode=0):
    mode_num = mode
    if old_solution and (self.runprofile == False):
      self.profile(mode=mode_num)
    elif (old_solution == False) and (self.runprofile2 == False):
      self.profile(old_solution=False, mode=mode_num)

    signature = 1
    if compare != None:
      with h5py.File(compare, 'r') as fp:
        x = np.array(fp['x'])
        rho = np.array(fp['rho'])
        v = np.array(fp['v'])
        pg = np.array(fp['pg'])
        pc = np.array(fp['pc'])
        fc = np.array(fp['fc'])

        mass = np.abs(rho*v)
        mom = rho*v**2 + pg + pc
        eng = np.abs((0.5*rho*v**2 + (gamma_g/(gamma_g - 1.))*pg)*v + fc)
        ma = np.abs(v/(self.B/np.sqrt(rho)))
        wave = (1. + ma/(gamma_g - 1.))**(2.*gamma_g)*(pg + (gamma_g - 1.)*self.B**2*(2*gamma_g*ma + 1. - gamma_g)/(gamma_g*(2.*gamma_g + 1.)))   
        wave2 = np.cbrt((1. - ma/(gamma_g - 1.))**(10))*(pg + (gamma_g - 1.)*self.B**2*(-2*gamma_g*ma + 1. - gamma_g)/(gamma_g*(2.*gamma_g + 1.)))

      # Save to memory
      self.x_sim = x 
      self.rho_sim = rho 
      self.v_sim = v 
      self.pg_sim = pg 
      self.pc_sim = pc 
      self.fc_sim = fc

      # Shift position of curves by finding the location of max grad pc
      drhodx = np.gradient(rho,x)
      dpcdx = np.gradient(pc, x)
      x0 = x[np.argmax(np.abs(dpcdx))]
      x0_int = self.x_int[np.argmax(np.abs(np.gradient(self.pc_int[:-1], self.x_int[:-1])))]
      if dpcdx[np.argmax(np.abs(dpcdx))] > 0:
        self.x_int = self.x_int - x0_int + x0 
        signature = 1
      else:
        self.x_int = -(self.x_int - x0_int) + x0
        signature = -1

    fig = plt.figure()

    grids = gs.GridSpec(5, 1, figure=fig, hspace=0)
    ax1 = fig.add_subplot(grids[0, 0])
    ax2 = fig.add_subplot(grids[1, 0])
    ax3 = fig.add_subplot(grids[2, 0])
    ax4 = fig.add_subplot(grids[3, 0])
    ax5 = fig.add_subplot(grids[4, 0])

    ax1.plot(self.x_int, self.rho_int, 'o-', label='$\\rho$')
    ax2.plot(self.x_int, signature*self.v_int, 'o-', label='$v$')
    ax3.plot(self.x_int, self.pg_int, 'o-', label='$P_g$')
    ax4.plot(self.x_int, self.pc_int, 'o-', label='$P_c$')
    ax5.plot(self.x_int, signature*self.fc_int, 'o-', label='$F_c$')

    if compare != None:
      ax1.plot(x, rho, '--', label='Sim')
      ax2.plot(x, v, '--', label='Sim')
      ax3.plot(x, pg, '--', label='Sim')
      ax4.plot(x, pc, '--', label='Sim')
      ax5.plot(x, fc, '--', label='Sim')

    for axes in fig.axes:
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

    # Plot conserved quantities
    fig2 = plt.figure()

    grids2 = gs.GridSpec(4, 1, figure=fig2, hspace=0)
    axx1 = fig2.add_subplot(grids2[0, 0])
    axx2 = fig2.add_subplot(grids2[1, 0])
    axx3 = fig2.add_subplot(grids2[2, 0])
    axx4 = fig2.add_subplot(grids2[3, 0])

    axx1.plot(self.x_int, self.J*np.ones(np.size(self.x_int)), label='Mass flux')
    axx2.plot(self.x_int, self.M*np.ones(np.size(self.x_int)), label='Momentum flux')
    axx3.plot(self.x_int, self.E*np.ones(np.size(self.x_int)), label='Energy flux')
    axx4.plot(self.x_int, self.wave_int, label='Wave adiabat')

    if compare != None:
      axx1.plot(x, mass, '--', label='Sim')
      axx2.plot(x, mom, '--', label='Sim')
      axx3.plot(x, eng, '--', label='Sim')
      axx4.plot(x, wave, '--', label='Sim/unidir')
      axx4.plot(x, wave2, '--', label='Sim/bidir')

    for axes in fig2.axes:
      axes.xaxis.set_minor_locator(AutoMinorLocator())
      axes.yaxis.set_minor_locator(AutoMinorLocator())
      axes.legend(frameon=False, fontsize=10)
      if axes != axx4:
        axes.set_xticks([])
      else:
        axes.set_xlabel('$x$', fontsize=10)

      for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(10)

    fig2.tight_layout()

    return (fig, fig2)

  # Evaluate data for Athena++ input
  def athinput(self, grid, block, old_solution=True, mode=0, nghost=2):
    if (grid%block != 0):
      print('Grid not evenly divisible by block!')
      return
    else:
      num_block = grid//block
      size_block = block
      block_id = np.arange(num_block)

    ldiff = self.kappa/self.v 
    if old_solution:
      vf = self.v_adrefhugzeros[mode] if np.size(self.v_adrefhugzeros) != 0 else self.v_adhugzeros[0]
    else:
      vf = self.v_adref2hugzeros[mode] if np.size(self.v_adref2hugzeros) != 0 else self.v_adhug2zeros[0] 

    if np.size(vf) == 0:
      print('No solution')
      return 

    # Prepare by calculating what the shock width would be
    totgrid = grid + 2*nghost
    pre_bin = 4*totgrid
    yf = vf/self.v
    y_pre = np.linspace(0.99999999999999, 1.00000000000001*yf, pre_bin)
    dxdy = lambda y, x: ldiff*self.D(y)/((1. - y)*self.N(y))
    sol = integrate.solve_ivp(dxdy, [y_pre[0], yf], [0.], t_eval=y_pre) # 0.9999 to ensure t_span encompasses y_int
    x_pre = sol.y[0]

    y_ath = np.zeros(totgrid)
    rho_ath = np.zeros(totgrid)
    v_ath = np.zeros(totgrid)
    pg_ath = np.zeros(totgrid)
    pc_ath = np.zeros(totgrid)
    fc_ath = np.zeros(totgrid)
    if old_solution and (np.size(self.v_adrefhugzeros) != 0):
      x0 = x_pre[0]
      xmid = x_pre[-1]
      x1 = 4.*(xmid - x0)/3. + x0
      x_ath = np.linspace(x0, x1, totgrid)
      dx = (x1 - x0)/(totgrid - 1)
      mid_index = np.argmin(np.abs(x_ath - xmid))
      mid_index = mid_index if x_ath[mid_index] < xmid else mid_index - 1
      y_ath[0:(mid_index+1)] = np.interp(x_ath[0:(mid_index+1)], x_pre, y_pre)

      # Values in the compressive region
      rho_ath[0:(mid_index+1)] = self.J/(self.v*y_ath[0:(mid_index+1)]) 
      v_ath[0:(mid_index+1)] = self.v*y_ath[0:(mid_index+1)]
      pg_ath[0:(mid_index+1)] = self.pg*np.array([self.adiabat(y) for i, y in enumerate(y_ath[0:(mid_index+1)])])
      pc_ath[0:(mid_index+1)] = self.M - self.J*self.v*y_ath[0:(mid_index+1)] - pg_ath[0:(mid_index+1)]
      fc_ath[0:(mid_index+1)] = self.E - 0.5*self.J*(self.v*y_ath[0:(mid_index+1)])**2 - (gamma_g/(gamma_g - 1.))*pg_ath[0:(mid_index+1)]*self.v*y_ath[0:(mid_index+1)]

      # Downstream solution after subshock
      v2 = self.v_final[mode]
      rho2 = self.J/v2
      pg2 = self.pg*self.hugoniot(v2/self.v)
      pc2 = self.M - self.J*v2 - pg2 
      fc2 = self.E - 0.5*self.J*v2**2 - (gamma_g/(gamma_g - 1.))*pg2*v2

      rho_ath[(mid_index+1):] = rho2 
      v_ath[(mid_index+1):] = v2 
      pg_ath[(mid_index+1):] = pg2
      pc_ath[(mid_index+1):] = pc2 
      fc_ath[(mid_index+1):] = fc2
    elif (old_solution == False) and (np.size(self.v_adref2hugzeros) != 0):
      x0 = x_pre[0]
      xmid = x_pre[-1]
      x1 = 4.*(xmid - x0)/3. + x0 
      x_ath = np.linspace(x0, x1, totgrid)
      dx = (x1 - x0)/(totgrid - 1)
      mid_index = np.argmin(np.abs(x_ath - xmid))
      mid_index = mid_index if x_ath[mid_index] < xmid else mid_index - 1
      y_ath[0:(mid_index+1)] = np.interp(x_ath[0:(mid_index+1)], x_pre, y_pre)

      # Values in the compressive region
      rho_ath[0:(mid_index+1)] = self.J/(self.v*y_ath[0:(mid_index+1)]) 
      v_ath[0:(mid_index+1)] = self.v*y_ath[0:(mid_index+1)]
      pg_ath[0:(mid_index+1)] = self.pg*np.array([self.adiabat(y) for i, y in enumerate(y_ath[0:(mid_index+1)])])
      pc_ath[0:(mid_index+1)] = self.M - self.J*self.v*y_ath[0:(mid_index+1)] - pg_ath[0:(mid_index+1)]
      fc_ath[0:(mid_index+1)] = self.E - 0.5*self.J*(self.v*y_ath[0:(mid_index+1)])**2 - (gamma_g/(gamma_g - 1.))*pg_ath[0:(mid_index+1)]*self.v*y_ath[0:(mid_index+1)]

      # Downstream solution after subshock
      v2 = self.v_final2[mode]
      rho2 = self.J/v2
      pg2 = self.pg*self.hugoniot2(v2/self.v)
      pc2 = self.M - self.J*v2 - pg2 
      fc2 = self.E - 0.5*self.J*v2**2 - (gamma_g/(gamma_g - 1.))*pg2*v2

      rho_ath[(mid_index+1):] = rho2 
      v_ath[(mid_index+1):] = v2 
      pg_ath[(mid_index+1):] = pg2
      pc_ath[(mid_index+1):] = pc2 
      fc_ath[(mid_index+1):] = fc2
    else:
      x0 = x_pre[0]
      x1 = x_pre[-1]
      x_ath = np.linspace(x0, x1, totgrid)
      dx = (x1 - x0)/(totgrid - 1)
      y_ath = np.interp(x_ath, x_pre, y_pre)

      # Values in the compressive region
      rho_ath = self.J/(self.v*y_ath) 
      v_ath = self.v*y_ath
      pg_ath = self.pg*np.array([self.adiabat(y) for i, y in enumerate(y_ath)])
      pc_ath = self.M - self.J*self.v*y_ath - pg_ath
      fc_ath = self.E - 0.5*self.J*(self.v*y_ath)**2 - (gamma_g/(gamma_g - 1.))*pg_ath*self.v*y_ath

    # Shifting to accommodate for ghost zones
    x_ath = x_ath - (nghost - 0.5)*dx

    # Process the data into format that can be input into Athena++
    x_input = np.zeros((num_block, block+2*nghost))
    rho_input = np.zeros((num_block, block+2*nghost))
    v_input = np.zeros((num_block, block+2*nghost))
    pg_input = np.zeros((num_block, block+2*nghost))
    pc_input = np.zeros((num_block, block+2*nghost))
    fc_input = np.zeros((num_block, block+2*nghost))

    for i in np.arange(num_block):
      # Fill the active zones
      x_input[i, nghost:(nghost+block)] = x_ath[i*block:(i+1)*block]
      rho_input[i, nghost:(nghost+block)] = rho_ath[i*block:(i+1)*block] 
      v_input[i, nghost:(nghost+block)] = v_ath[i*block:(i+1)*block]
      pg_input[i, nghost:(nghost+block)] = pg_ath[i*block:(i+1)*block]
      pc_input[i, nghost:(nghost+block)] = pc_ath[i*block:(i+1)*block]
      fc_input[i, nghost:(nghost+block)] = fc_ath[i*block:(i+1)*block]

      # Fill the ghost zones
      x_input[i, 0:nghost] = x_ath[0:nghost]
      x_input[i, -nghost:] = x_ath[-nghost:]
      rho_input[i, 0:nghost] = rho_ath[0:nghost]
      rho_input[i, -nghost:] = rho_ath[-nghost:]
      v_input[i, 0:nghost] = v_ath[0:nghost]
      v_input[i, -nghost:] = v_ath[-nghost:]
      pg_input[i, 0:nghost] = pg_ath[0:nghost]
      pg_input[i, -nghost:] = pg_ath[-nghost:]
      pc_input[i, 0:nghost] = pc_ath[0:nghost]
      pc_input[i, -nghost:] = pc_ath[-nghost:]
      fc_input[i, 0:nghost] = fc_ath[0:nghost]
      fc_input[i, -nghost:] = fc_ath[-nghost:]

    # Save data to memory
    self.xinner = x_ath[0] + (nghost - 0.5)*dx
    self.xouter = x_ath[-1] - (nghost - 0.5)*dx 
    self.dx = dx
    self.x_ath = x_ath
    self.rho_ath = rho_ath
    self.v_ath = v_ath
    self.pg_ath = pg_ath 
    self.pc_ath = pc_ath 
    self.fc_ath = fc_ath
    self.x_input = x_input
    self.rho_input = rho_input
    self.v_input = v_input
    self.pg_input = pg_input
    self.pc_input = pc_input 
    self.fc_input = fc_input
    return 
# End of class

###########################################
plotdefault()

# rho1 = 72.5
# pg1 = 0.01
# m1 = 40.
# n1 = 0.5
# beta1 = 2.
# upstream = mnbeta_to_gas(rho1, pg1, m1, n1, beta1)

upstream = {}
upstream['rho'] = 72.5
upstream['v'] = 0.5699032262849384
upstream['pg'] = 0.009999999776482582
upstream['pc'] = 0.009999999776482582
upstream['B'] = 0.05000000074505806

kappa = 0.01

alter = gas_to_mnbeta(upstream['rho'], upstream['pg'], upstream['v'], upstream['pc'], upstream['B'])

shock = Shock(upstream['rho'], upstream['pg'], upstream['v'], upstream['pc'], upstream['B'], kappa)
downstream = shock.solution()
downstream2 = shock.solution2()
down_mnbeta = gas_to_mnbeta(downstream['rho'], downstream['pg'], downstream['v'], downstream['pc'], upstream['B'])
down_mnbeta2 = gas_to_mnbeta(downstream2['rho'], downstream2['pg'], downstream2['v'], downstream2['pc'], upstream['B'])

# Plot diagram
# latexify(columns=1)
fig = shock.shock_diagram(don_want_axis=False)
fig.savefig('./sh_struct_stream.png', dpi=300)
plt.show()
# plotdefault()

# Plot shock profile
shkfig, convfig = shock.plotprofile(compare='./shock.hdf5', old_solution=False, mode=2)
# shkfig, convfig = shock.plotprofile(old_solution=False, mode=0)
shkfig.savefig('./sh_profile_stream.png', dpi=300)
convfig.savefig('./sh_conv_stream.png', dpi=300)
plt.show() 

# # First argument: total number of cells; Second argument: number of cells per meshblock
# shock.athinput(4096, 64, old_solution=False, mode=0, nghost=2)
# with h5py.File('./shock_still.hdf5', 'w') as fp:
#   dset = fp.create_dataset('x', data=shock.x_input)
#   dset = fp.create_dataset('rho', data=shock.rho_input)
#   dset = fp.create_dataset('v', data=shock.v_input)
#   dset = fp.create_dataset('pg', data=shock.pg_input)
#   dset = fp.create_dataset('ec', data=shock.pc_input/(gamma_c - 1.))
#   dset = fp.create_dataset('fc', data=shock.fc_input)
#   fp.attrs.create('B', shock.B )

# print('rho0 = {}'.format(shock.rho))
# print('v0 = {}'.format(shock.v))
# print('pg0 = {}'.format(shock.pg))
# print('ec0 = {}'.format(shock.pc/(gamma_c - 1.)))
# print('drhodx = {}'.format(np.gradient(shock.rho_ath, shock.x_ath)[0]))
# print('dvdx = {}'.format(np.gradient(shock.v_ath, shock.x_ath)[0]))
# print('dpgdx = {}'.format(np.gradient(shock.pg_ath, shock.x_ath)[0]))
# print('decdx = {}'.format(np.gradient(shock.pc_ath, shock.x_ath)[0]/(gamma_c - 1.)))

plt.close('all')

# Section for acceleration efficiency
# rho1 = 1.
# pg1 = 1.
# m1 = 1.5
# n1 = 0.5
# beta1 = np.logspace(-1, 3, 100)
# kappa = 1.

# # Plot acceleration efficiency
# efficiency = np.array([])
# beta_eff = np.array([])
# for i, beta in enumerate(beta1):
#   print(i)
#   upstream = mnbeta_to_gas(rho1, pg1, m1, n1, beta)
#   shock = Shock(upstream['rho'], upstream['pg'], upstream['v'], upstream['pc'], upstream['B'], kappa) 
#   downstream = shock.solution2()
#   rho1 = shock.rho 
#   rho2 = downstream['rho']
#   va1 = shock.B/np.sqrt(rho1)
#   va2 = shock.B/np.sqrt(rho2)
#   v1 = shock.v 
#   v2 = downstream['v']
#   pc1 = shock.pc 
#   pc2 = downstream['pc']
#   eta = (gamma_c/(gamma_c - 1.))*((v2 + va2)*pc2 - (v1 - va1)*pc1)/(0.5*rho1*v1**3 - 0.5*rho2*v2**3)
#   for j, eff in enumerate(eta):
#     beta_eff = np.append(beta_eff, beta)
#     efficiency = np.append(efficiency, eff)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('M = {:.1f}, N = {:.1f}'.format(m1, n1))

# ax.scatter(beta_eff, efficiency)

# ax.margins(x=0)
# ax.set_ylim(0)
# ax.set_xlabel('$\\beta$')
# ax.set_ylabel('$\\eta_s$')
# ax.yaxis.set_minor_locator(AutoMinorLocator())
# ax.set_xscale('log')

# fig.tight_layout()
# fig.savefig('/Users/tsunhinnavintsung/Box/Share/Shock2/stream_eff_m{:.1f}_n{:.1f}.png'.format(m1, n1), dpi=300)
# plt.show(fig)
# plt.close('all')

# Section for Pc/Total momentum flux against N 
rho1 = 72.5
pg1 = 0.01
m1 = 40.
n1 = np.linspace(0.01, 0.99, 300)
beta1 = 15.
kappa = 0.01

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('M = {0}, $\\beta$ = {1}'.format(m1, beta1))

for i, n in enumerate(n1):
  print(i)
  upstream = mnbeta_to_gas(rho1, pg1, m1, n, beta1)
  shock = Shock(upstream['rho'], upstream['pg'], upstream['v'], upstream['pc'], upstream['B'], kappa) 
  downstream = shock.solution()
  downstream2 = shock.solution2()
  pc_frac = downstream['pc']/(shock.rho*shock.v**2)
  pc_frac2 = downstream2['pc']/(shock.rho*shock.v**2)
  for j, frac in enumerate(pc_frac):
    if (i == 0) and (j == 0):
      ax.scatter(n, frac, color='k', label='Old sol.')
    else:
      ax.scatter(n, frac, color='k')
  for k, frac2 in enumerate(pc_frac2): 
    if (i == 0) and (k == 0):
      ax.scatter(n, frac2, marker='*', color='b', label='New sol.')
    else:
      ax.scatter(n, frac2, marker='*', color='b')

ax.legend(frameon=False)
ax.margins(x=0)
ax.set_ylim(0, 1)
ax.set_xlabel('$N$')
ax.set_ylabel('$\\frac{P_{c2}}{\\rho_1 v_1^2}$')

fig.tight_layout()
# fig.savefig('/Users/tsunhinnavintsung/Box/Share/Shock2/stream_m{:.1f}_b{:.1f}.png'.format(m1, beta1), dpi=300)
plt.show()
plt.close('all')

############################################
# Plots for publication

# # Plot non-monotonicity in path x
# latexify(columns=1)
# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.plot(shock.x_int, shock.rho_int) 
# ax.set_xlabel('$x$')
# ax.set_ylabel('$\\rho$')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.margins(x=0)

# fig.tight_layout()
# fig.savefig('/Users/tsunhinnavintsung/Box/Share/Publish/non_mono.png', dpi=300)
# plt.show(fig)
# plt.close('all')
# plotdefault()



# # Compare shock analytics and simulation
# # Shift position of curves by finding the location of max grad pc
# drhodx = np.gradient(shock.rho_sim, shock.x_sim)
# dpcdx = np.gradient(shock.pc_sim, shock.x_sim)
# x0 = shock.x_sim[np.argmax(np.abs(dpcdx))]
# x0_int = shock.x_int[np.argmax(np.abs(np.gradient(shock.pc_int[:-1], shock.x_int[:-1])))]
# if dpcdx[np.argmax(np.abs(dpcdx))] > 0:
#   x_int = shock.x_int - x0_int + x0 
#   signature = 1
# else:
#   x_int = -(shock.x_int - x0_int) + x0
#   signature = -1

# latexify(columns=1)
# fig1 = plt.figure()
# fig2 = plt.figure()
# fig = [fig1, fig2]
# ax1 = fig1.add_subplot(111)
# ax2 = fig2.add_subplot(111)
# ax = [ax1, ax2]

# ax1.plot(x_int, shock.rho_int, '-o', label='$\\rho$')
# ax2.plot(x_int, shock.pc_int, '-o', label='$P_c$')
# ax1.plot(shock.x_sim, shock.rho_sim, '--', label='Sim')
# ax2.plot(shock.x_sim, shock.pc_sim, '--', label='Sim')

# ax1.set_ylabel('$\\rho$')
# ax2.set_ylabel('$P_c$')

# for axes in ax:
#   axes.set_xlabel('$x$')
#   axes.xaxis.set_minor_locator(AutoMinorLocator())
#   axes.yaxis.set_minor_locator(AutoMinorLocator())
#   axes.legend(frameon=False)

# for i, figu in enumerate(fig):
#   figu.tight_layout()
#   # figu.savefig('/Users/tsunhinnavintsung/Box/Share/Publish/free_{}.png'.format(i), dpi=300)
# plt.show()
# plt.close('all')
# plotdefault()



# # Pc/P_tot against N for one set of parameter
# rho1 = 1000. 
# pg1 = 1. 
# m1 = 20. 
# n1 = np.linspace(0.01, 0.99, 100)
# beta1 = 1. 
# kappa = 0.1 

# q_t = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# t_diff = np.array([2424., 2725, 2092., 2452., 3003., 1959., 2867., 2787.])

# q_old = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# eta_up_old = np.zeros(np.size(q_old))
# eta1_old = np.array([0.5705606187656989, 0.5835184795542948, 0.597910452659204, \
#   0.6141650821672666, 0.6329550810085214, 0.6554453681130281, 0.6839953631732907, \
#   0.7250716440690459])
# eta2_old = np.array([0.5554536402824949, 0.5663263375363822, 0.5898098747942703, \
#   0.6060709949381512, 0.6247812611797006, 0.6526192223218231, 0.6819884349713877, \
#   0.7245896975611089])

# q_new = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
# eta_up_new = np.zeros(np.size(q_new))
# eta1_new = np.array([0.48089374889596215, 0.559162125464414, 0.612783280686031, \
#   0.6629019830377982, 0.7194997419252992])
# eta2_new = np.array([0.4917601275517108, 0.5696870759381972, 0.6162115546021458, \
#   0.6665856288837653, 0.7222247058863441])

# q_new1 = np.array([0.5, 0.6, 0.7])
# eta_up_new1 = np.zeros(np.size(q_new1))
# eta1_new1 = np.array([0.35728640141469686, 0.2656025434050776, 0.191137897915609])
# eta2_new1 = np.array([0.5225403882553337, 0.6320887199970091, 0.650577444858814])

# q_new2 = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
# eta_up_new2 = np.zeros(np.size(q_new2))
# eta1_new2 = np.array([0.006568164740519238, 0.010740619622377384, 0.015868588032381496, \
#   0.022527241094393087, 0.03202416560589042, 0.04864944585002258])
# eta2_new2 = np.array([0.006590172439589351, 0.010773589328490327, 0.015910333514280672, \
#   0.022592266981859725, 0.03214284733136734, 0.04885696329060824])

# latexify(columns=1)
# fig = plt.figure()
# ax = fig.add_subplot(111)

# for i, n in enumerate(n1):
#   print(i) 
#   upstream  = mnbeta_to_gas(rho1, pg1, m1, n, beta1)
#   shock = Shock(upstream['rho'], upstream['pg'], upstream['v'], upstream['pc'], upstream['B'], kappa) 
#   downstream = shock.solution() 
#   downstream2 = shock.solution2() 
#   pc_frac = (downstream['pc'] - upstream['pc'])/(shock.J*shock.v)
#   pc_frac2 = (downstream2['pc'] - upstream['pc'])/(shock.J*shock.v)
#   for r, frac in enumerate(pc_frac):
#       ax.scatter(n, frac, color='k')
#   for s, frac2 in enumerate(pc_frac2): 
#     ax.scatter(n, frac2, marker='*', color='b')

# for i, n in enumerate(q_old):
#   upstream = mnbeta_to_gas(rho1, pg1, m1, n, beta1)
#   eta_up_old[i] = (pg1/(rho1*upstream['v']**2))*(n/(1. - n))

# for i, n in enumerate(q_new):
#   upstream = mnbeta_to_gas(rho1, pg1, m1, n, beta1)
#   eta_up_new[i] = (pg1/(rho1*upstream['v']**2))*(n/(1. - n))

# for i, n in enumerate(q_new1):
#   upstream = mnbeta_to_gas(rho1, pg1, m1, n, beta1)
#   eta_up_new1[i] = (pg1/(rho1*upstream['v']**2))*(n/(1. - n))

# for i, n in enumerate(q_new2):
#   upstream = mnbeta_to_gas(rho1, pg1, m1, n, beta1)
#   eta_up_new2[i] = (pg1/(rho1*upstream['v']**2))*(n/(1. - n))

# # ax.scatter(q_old, eta1_old, s=20, marker='o')
# # ax.scatter(q_new, eta1_new, s=20, marker='^')
# # ax.scatter(q_new1, eta1_new1, s=20, marker='s')
# # ax.scatter(q_new2, eta1_new2, s=20,  marker='D')

# ax.scatter(q_old, eta2_old - eta_up_old, s=20, marker='o')
# ax.scatter(q_new, eta2_new - eta_up_new, s=20, marker='^')
# ax.scatter(q_new1, eta2_new1 - eta_up_new1, s=20, marker='s')
# ax.scatter(q_new2, eta2_new2 - eta_up_new2, s=20,  marker='D')

# ax.set_xlabel('$Q$')
# ax.set_ylabel('$\\Delta P_c/\\rho_{1}v_{1}^2$')
# ax.xaxis.set_minor_locator(AutoMinorLocator())
# ax.yaxis.set_minor_locator(AutoMinorLocator())

# fig.tight_layout()
# # fig.savefig('/Users/tsunhinnavintsung/Box/Share/Publish/final.png', dpi=300)
# plt.show()
# plt.close('all')
# plotdefault()



# # Section for Pc/Total momentum flux against M for different N
# rho1 = 1.
# pg1 = 1.
# m1 = np.logspace(0.01, 2, 1000)
# n1 = np.array([0.1, 0.5])
# beta1 = 1.
# kappa = 0.1

# latexify(columns=1)
# fig1 = plt.figure()
# fig2 = plt.figure()
# fig = [fig1, fig2]
# ax1 = fig1.add_subplot(111)
# ax2 = fig2.add_subplot(111)
# ax = [ax1, ax2]

# for i, n in enumerate(n1):
#   for j, m in enumerate(m1):
#     print(i, j)
#     upstream = mnbeta_to_gas(rho1, pg1, m, n, beta1)
#     shock = Shock(upstream['rho'], upstream['pg'], upstream['v'], upstream['pc'], upstream['B'], kappa) 
#     downstream = shock.solution()
#     downstream2 = shock.solution2()
#     pc_frac = (downstream['pc'] - upstream['pc'])/(shock.rho*shock.v**2)
#     pc_frac2 = (downstream2['pc'] - upstream['pc'])/(shock.rho*shock.v**2)
#     for l, frac in enumerate(pc_frac):
#       ax[i].scatter(m, frac, color='k')
#     for k, frac2 in enumerate(pc_frac2): 
#       ax[i].scatter(m, frac2, marker='*', color='b')

# for i, axes in enumerate(ax):
#   axes.margins(x=0)
#   axes.set_ylim(0, 1)
#   axes.set_xlabel('$M$')
#   axes.set_ylabel('$\\Delta P_c/\\rho_1 v_1^2}$')
#   axes.yaxis.set_minor_locator(AutoMinorLocator())
#   axes.set_title('$Q = ${:.1f}, $\\beta = ${:.1f}'.format(n1[i], beta1))
#   axes.set_xscale('log')

# for i, figu in enumerate(fig):
#   figu.tight_layout()
#   # figu.savefig('/Users/tsunhinnavintsung/Box/Share/Publish/Q{:.1f}.png'.format(n1[i]), dpi=300)
# plt.show()
# plt.close('all')
# plotdefault()



# # Pc/P_tot against N for different M and beta
# rho1 = 1.
# pg1 = 1.
# m1 = np.array([2., 5., 10., 20.])
# n1 = np.linspace(0.01, 0.99, 100)
# beta1 = np.array([1., 5., 20., 1000.])
# kappa = 1.

# latexify(columns=2)
# fig = plt.figure()
# grids = gs.GridSpec(4, 4, figure=fig, hspace=0, wspace=0)
# ax11 = fig.add_subplot(grids[0, 0])
# ax21 = fig.add_subplot(grids[1, 0])
# ax31 = fig.add_subplot(grids[2, 0])
# ax41 = fig.add_subplot(grids[3, 0])

# ax12 = fig.add_subplot(grids[0, 1])
# ax22 = fig.add_subplot(grids[1, 1])
# ax32 = fig.add_subplot(grids[2, 1])
# ax42 = fig.add_subplot(grids[3, 1])

# ax13 = fig.add_subplot(grids[0, 2])
# ax23 = fig.add_subplot(grids[1, 2])
# ax33 = fig.add_subplot(grids[2, 2])
# ax43 = fig.add_subplot(grids[3, 2])

# ax14 = fig.add_subplot(grids[0, 3])
# ax24 = fig.add_subplot(grids[1, 3])
# ax34 = fig.add_subplot(grids[2, 3])
# ax44 = fig.add_subplot(grids[3, 3])

# for num, axes in enumerate(fig.axes):
#   i = num//4
#   j = num%4
#   for k, n in enumerate(n1):
#     print(i, j, k)
#     upstream = mnbeta_to_gas(rho1, pg1, m1[i], n, beta1[j])
#     shock = Shock(upstream['rho'], upstream['pg'], upstream['v'], upstream['pc'], upstream['B'], kappa) 
#     downstream = shock.solution()
#     downstream2 = shock.solution2()
#     pc_frac = (downstream['pc'] - upstream['pc'])/(shock.J*shock.v)
#     pc_frac2 = (downstream2['pc'] - upstream['pc'])/(shock.J*shock.v)
#     for r, frac in enumerate(pc_frac):
#       axes.scatter(n, frac, color='k')
#     for s, frac2 in enumerate(pc_frac2): 
#       axes.scatter(n, frac2, marker='*', color='b')

#   axes.xaxis.set_minor_locator(AutoMinorLocator())
#   axes.yaxis.set_minor_locator(AutoMinorLocator())
#   if axes in [ax11, ax21, ax31]:
#     axes.set_ylabel('$\\Delta P_c/\\rho_1 v^2_1$')
#     axes.set_xticks([])
#     if axes == ax11:
#       axes.annotate('$M = ${:.1f}'.format(m1[i]), xy=(0.5, 1.05), xycoords='axes fraction')
#   elif axes in [ax42, ax43, ax44]:
#     axes.set_xlabel('$Q$')
#     axes.set_yticks([])
#     if axes == ax44:
#       axes.annotate('$\\beta = ${:.1f}'.format(beta1[j]), xy=(1.05, 0.5), xycoords='axes fraction', rotation=270)
#   elif axes == ax41:
#     axes.set_xlabel('$Q$')
#     axes.set_ylabel('$\\Delta P_c/\\rho_1 v^2_1$')
#   elif axes in [ax12, ax13, ax14]:
#     axes.set_xticks([])
#     axes.set_yticks([])
#     axes.annotate('$M = ${:.1f}'.format(m1[i]), xy=(0.5, 1.05), xycoords='axes fraction')
#     if axes == ax14:
#       axes.annotate('$\\beta = ${:.1f}'.format(beta1[j]), xy=(1.05, 0.5), xycoords='axes fraction', rotation=270)
#   elif axes in [ax24, ax34]:
#     axes.set_xticks([])
#     axes.set_yticks([])
#     axes.annotate('$\\beta = ${:.1f}'.format(beta1[j]), xy=(1.05, 0.5), xycoords='axes fraction', rotation=270)
#   else:
#     axes.set_xticks([])
#     axes.set_yticks([])

# fig.tight_layout()
# # fig.savefig('/Users/tsunhinnavintsung/Box/Share/Publish/Sol_structure.png', dpi=300)
# plt.show()
# plt.close('all')
# plotdefault()



# Plot acceleration efficiency
# rho1 = 1.
# pg1 = 1.
# m1 = np.array([2., 3., 4., 6., 10., 15.])
# n1 = 0.5
# beta1 = np.logspace(-1, 3, 200)
# kappa = 1.

# latexify(columns=1)
# fig = plt.figure()
# fig2 = plt.figure()
# ax = fig.add_subplot(111)
# ax2 = fig2.add_subplot(111)

# for k, m in enumerate(m1):
#   efficiency1 = np.array([])
#   efficiency2 = np.array([])
#   beta_eff1 = np.array([])
#   beta_eff2 = np.array([])
#   for i, beta in enumerate(beta1):
#     print(i)
#     upstream = mnbeta_to_gas(rho1, pg1, m, n1, beta)
#     shock = Shock(upstream['rho'], upstream['pg'], upstream['v'], upstream['pc'], upstream['B'], kappa) 
#     downstream = shock.solution()
#     downstream2 = shock.solution2()
#     rho = shock.rho
#     rho_1 = downstream['rho']
#     rho_2 = downstream2['rho']
#     va = shock.B/np.sqrt(rho)
#     va_1 = shock.B/np.sqrt(rho_1)
#     va_2 = shock.B/np.sqrt(rho_2)
#     v = shock.v
#     v_1 = downstream['v']
#     v_2 = downstream2['v']
#     pc = shock.pc
#     pc_1 = downstream['pc']
#     pc_2 = downstream2['pc']
#     eta_1 = (gamma_c/(gamma_c - 1.))*((v_1 - va_1)*pc_1 - (v - va)*pc)/(0.5*rho*v**3 - 0.5*rho_1*v_1**3)
#     eta_2 = (gamma_c/(gamma_c - 1.))*((v_2 + va_2)*pc_2 - (v - va)*pc)/(0.5*rho*v**3 - 0.5*rho_2*v_2**3)
#     for j, eff in enumerate(eta_1):
#       beta_eff1 = np.append(beta_eff1, beta)
#       efficiency1 = np.append(efficiency1, eff)
#     for j, eff in enumerate(eta_2):
#       beta_eff2 = np.append(beta_eff2, beta)
#       efficiency2 = np.append(efficiency2, eff)

#   ax.scatter(beta_eff1, efficiency1, label='{:.1f}'.format(m))
#   ax2.scatter(beta_eff2, efficiency2, marker='*', label='{:.1f}'.format(m))

# ax.margins(x=0)
# ax.set_ylim(0)
# ax.set_xlabel('$\\beta$')
# ax.set_ylabel('$\\eta_s$')
# ax.yaxis.set_minor_locator(AutoMinorLocator())
# ax.set_xscale('log')
# ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")

# ax2.margins(x=0)
# ax2.set_ylim(0)
# ax2.set_xlabel('$\\beta$')
# ax2.set_ylabel('$\\eta_s$')
# ax2.yaxis.set_minor_locator(AutoMinorLocator())
# ax2.set_xscale('log')
# ax2.legend(bbox_to_anchor=(1.04,1), loc="upper left")

# fig.tight_layout()
# fig2.tight_layout()
# fig.savefig('/Users/tsunhinnavintsung/Box/Share/Shock2/stream_eff_old.png'.format(m1, n1), dpi=300)
# fig2.savefig('/Users/tsunhinnavintsung/Box/Share/Shock2/stream_eff_new.png'.format(m1, n1), dpi=300)
# plt.show(fig)
# plt.close('all')



# # Image parameters with multiple branches of solution
# rho1 = 1.
# pg1 = 1.
# num_m = 100
# num_n = 100
# num_beta = 3
# m1 = np.linspace(1.1, 30., num_m)
# n1 = np.linspace(0.01, 0.99, num_n)
# beta1 = np.array([1., 20., 100.])
# kappa = 1.

# sol_num = np.zeros((num_beta, num_m, num_n))

# for k, beta in enumerate(beta1):
#   for i, m in enumerate(m1):
#     for j, n in enumerate(n1):
#       print(k, i, j)
#       upstream = mnbeta_to_gas(rho1, pg1, m, n, beta)
#       shock = Shock(upstream['rho'], upstream['pg'], upstream['v'], upstream['pc'], upstream['B'], kappa) 
#       downstream2 = shock.solution2()
#       sol_num[k, i, j] = np.size(downstream2['rho'])

# latexify(columns=2)
# fig = plt.figure()
# grids = gs.GridSpec(1, 3, figure=fig)
# ax1 = fig.add_subplot(grids[0, 0])
# ax2 = fig.add_subplot(grids[0, 1])
# ax3 = fig.add_subplot(grids[0, 2])

# for g, axes in enumerate(fig.axes):
#   cp = axes.imshow(np.transpose(sol_num[g, :, :]), aspect='auto', origin='lower', extent=(m1[0], m1[-1], n1[0], n1[-1]))
#   axes.set_xlabel('$M$')
#   axes.set_ylabel('$Q$')
#   axes.set_title('$\\beta = ${}'.format(beta1[g]))
# # fig.colorbar(cp)

# fig.tight_layout()
# fig.savefig('/Users/tsunhinnavintsung/Box/Share/Publish/num_sol.png', dpi=300)
# plt.show(fig)
# plt.close('all')
# plotdefault()



# # Section for Pc/Total momentum flux against N down lower N
# rho1 = 1.
# pg1 = 1.
# m1 = 30.
# n1 = np.linspace(0.01, 0.99, 300)
# beta1 = 0.5
# kappa = 0.1

# latexify(columns=1)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax.set_title('M = {0}, $\\beta$ = {1}'.format(m1, beta1))

# for i, n in enumerate(n1):
#   print(i)
#   upstream = mnbeta_to_gas(rho1, pg1, m1, n, beta1)
#   shock = Shock(upstream['rho'], upstream['pg'], upstream['v'], upstream['pc'], upstream['B'], kappa) 
#   downstream = shock.solution()
#   downstream2 = shock.solution2()
#   pc_frac = (downstream['pc'] - upstream['pc'])/(shock.rho*shock.v**2)
#   pc_frac2 = (downstream2['pc'] - upstream['pc'])/(shock.rho*shock.v**2)
#   for j, frac in enumerate(pc_frac):
#     if (i == 0) and (j == 0):
#       ax.scatter(n, frac, color='k')
#     else:
#       ax.scatter(n, frac, color='k')
#   for k, frac2 in enumerate(pc_frac2): 
#     if (i == 0) and (k == 0):
#       ax.scatter(n, frac2, marker='*', color='b')
#     else:
#       ax.scatter(n, frac2, marker='*', color='b')

# ax.scatter(0.95, 0.76, s=80, c='tab:brown', linewidths=1., edgecolors='k')
# ax.scatter(0.8, 0.030, s=80, c='tab:green', linewidths=1., edgecolors='k')

# # ax.legend(frameon=False)
# ax.margins(x=0)
# ax.set_ylim(0, 1)
# ax.set_xlabel('$Q$')
# ax.set_ylabel('$\\Delta P_c/\\rho_1 v_1^2$')
# ax.annotate('', xy=(0.2, 0.65), xycoords='data', xytext=(0.95, 0.83), textcoords='data', \
#   size=10, va="center", ha="center",  arrowprops=dict(arrowstyle="simple", connectionstyle="arc3,rad=-0.05"))
# ax.annotate('', xy=(0.2, 0.05), xycoords='data', xytext=(0.8, 0.1), textcoords='data', \
#   size=10, va="center", ha="center", arrowprops=dict(arrowstyle="simple", connectionstyle="arc3,rad=-0.05"))
# ax.text(0.95, 1.0, 'Start', bbox=dict(boxstyle='round', fc='w'))
# ax.text(0.8, 0.25, 'Start', bbox=dict(boxstyle='round', fc='w'))

# fig.tight_layout()
# # fig.savefig('/Users/tsunhinnavintsung/Box/Share/Publish/explain.png', dpi=300)
# plt.show()
# plt.close('all')
# plotdefault()



# # Section for Pc/Total momentum flux against N up N
# rho1 = 1.
# pg1 = 1.
# m1 = 30.
# n1 = np.linspace(0.01, 0.99, 300)
# beta1 = 0.5
# kappa = 0.1

# latexify(columns=1)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax.set_title('M = {0}, $\\beta$ = {1}'.format(m1, beta1))

# for i, n in enumerate(n1):
#   print(i)
#   upstream = mnbeta_to_gas(rho1, pg1, m1, n, beta1)
#   shock = Shock(upstream['rho'], upstream['pg'], upstream['v'], upstream['pc'], upstream['B'], kappa) 
#   downstream = shock.solution()
#   downstream2 = shock.solution2()
#   pc_frac = (downstream['pc'] - upstream['pc'])/(shock.rho*shock.v**2)
#   pc_frac2 = (downstream2['pc'] - upstream['pc'])/(shock.rho*shock.v**2)
#   for j, frac in enumerate(pc_frac):
#     if (i == 0) and (j == 0):
#       ax.scatter(n, frac, color='k')
#     else:
#       ax.scatter(n, frac, color='k')
#   for k, frac2 in enumerate(pc_frac2): 
#     if (i == 0) and (k == 0):
#       ax.scatter(n, frac2, marker='*', color='b')
#     else:
#       ax.scatter(n, frac2, marker='*', color='b')

# ax.scatter(0.2, 0.003, s=80, c='tab:green', linewidths=1., edgecolors='k')

# # ax.legend(frameon=False)
# ax.margins(x=0)
# ax.set_ylim(0, 1)
# ax.set_xlabel('$Q$')
# ax.set_ylabel('$\\Delta P_c/\\rho_1 v_1^2$')
# ax.annotate('', xy=(0.95, 0.70), xycoords='data', xytext=(0.2, 0.05), textcoords='data', \
#   size=10, va="center", ha="center",  arrowprops=dict(arrowstyle='simple', connectionstyle='angle3, angleA=5, angleB=90'))
# ax.text(0.2, 0.15, 'Start', bbox=dict(boxstyle='round', fc='w'))

# fig.tight_layout()
# # fig.savefig('/Users/tsunhinnavintsung/Box/Share/Publish/explain_rev.png', dpi=300)
# plt.show()
# plt.close('all')
# plotdefault()