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
plt.rcParams['lines.markersize'] = 3.
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

gamma_g = 5./3.
gamma_c = 4./3.

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
  def __init__(self, rho, pg, v, pc, B):
    self.rho = rho 
    self.v = v
    self.pg = pg 
    self.pc = pc 
    self.B = B

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
    self.S = (pg + (gamma_g - 1.)*B**2*(2*gamma_g*ma + 1. - gamma_g)/(gamma_g*(2.*gamma_g + 1.))) \
      *np.cbrt((ma + gamma_g - 1.)**10)

    # Parameters for plotting the shock diagram
    self.asymp = (gamma_c*(gamma_g - 1.)/((gamma_g - gamma_c)*ma))**2
    self.v_pgzero = self.M/self.J 
    self.v_pczero = gamma_g*self.M/((gamma_g + 1.)*self.J)
    self.v_hugzeros = self.hugzeros()
    self.regime = self.regime_num()
    self.v_hugpczeros = self.hugpczeros() # Does not necessarily exist
    self.v_adpcmaxzeros = self.adpcmaxzeros() 
    self.v_adhugzeros = self.adhugzeros()
    self.v_hugpcmaxzeros = self.hugpcmaxzeros() # Does not necessarily exist
    refhugon = self.refhugcoeff() # Does not necessarily exist
    self.coeff = refhugon['coeff']
    self.v_refpczero = refhugon['y']*v if np.size(refhugon['y'] > 0) else refhugon['y']
    self.v_ver = refhugon['ver']*v
    self.use_refhug = self.use_refhugon()
    adrefhug = self.adrefhugzeros()
    self.v_adrefhugzeros = adrefhug['v_adrefhugzeros'] # Does not necessarily exist
    self.v_final = adrefhug['v_final']
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
    # ms = self.ms 
    # mc = self.mc 
    # ma = self.ma 
    # delta = self.pc/self.pg
    # part1 = lambda y: 0.5*(gamma_c + 1.)*(y - (gamma_c - 1.)/(gamma_c + 1.))
    # part2 = lambda y: (gamma_c/(gamma_g*ms**2))*(1. + delta - (gamma_g - gamma_c)/(gamma_c*(gamma_g - 1.)*(1. - y)))
    # part3 = lambda y: (gamma_c/ma)*(np.sqrt(y) - 1./(gamma_c*mc**2*(1. + np.sqrt(y))) + np.sqrt(y)/(gamma_g*ms**2*(1. - y)))
    # lower = (gamma_c*(gamma_g - 1.)/((gamma_g - gamma_c)*ma))**2
    # upper = self.v_pgzero/self.v 
    # hug = lambda y: part1(y) - part2(y) - part3(y)
    J = self.J 
    M = self.M 
    E = self.E 
    v = self.v
    pg = self.pg
    ma = self.ma 
    asymp = self.asymp
    factor = lambda y: 1. - 1./(ma*np.sqrt(y))
    part1 = lambda y: E/(v*y) - J*v*y*(0.5 - (gamma_c/(gamma_c - 1.))*factor(y)) - (gamma_c/(gamma_c - 1.))*M*factor(y) 
    hug = lambda y: part1(y)/pg
    lower = 0.01
    upper = self.v_pgzero/self.v 
    y_sol = self.find_root(lower, upper, hug, asymptote=asymp)
    return y_sol*self.v # Returns the hugoniot zeros

  def regime_num(self):
    vz = self.v_hugzeros
    v = self.v
    asymp = self.asymp
    if np.size(vz) == 2:
      if (vz[0]/v > asymp) and (vz[-1]/v > asymp):
        return 1
      elif (vz[0]/v < asymp) and (vz[-1]/v > asymp):
        return 2
      elif (vz[0]/v < asymp) and (vz[-1]/v < asymp):
        return 3
      else:
        raise ValueError('Not valid regime')
    elif np.size(vz) in [0, 1]:
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

  def hugpcmaxzeros(self):
    asymp = self.asymp
    if (self.regime in [1, 2, 3]):
      lower = self.v_hugpczeros[0]/self.v if (self.v_hugzeros[0]/self.v < asymp) else self.v_hugzeros[0]/self.v 
    else:
      lower = self.v_hugpczeros[0]/self.v
    upper = self.v_pczero/self.v 
    hugpcmax = lambda y: self.hugoniot(y) - self.pcmax(y)
    y_sol = self.find_root(lower, upper, hugpcmax, asymptote=asymp)
    return y_sol*self.v # Returns the hugoniot-pc max intersection

  def use_refhugon(self):
    check = self.pcmax(self.v_hugpcmaxzeros[0]/self.v)
    refchk1 = self.refhug(self.v_hugpcmaxzeros[0]/self.v)
    refchk2 = self.refhug2(self.v_hugpcmaxzeros[0]/self.v)
    use_refhug = True if (np.abs(refchk1 - check) < np.abs(refchk2 - check)) else False
    return use_refhug

  def adrefhugzeros(self):
    asymp = self.asymp
    if np.size(self.v_refpczero) == 0: # The reflected hugoniot doesn't exist
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
          lower_fin = self.v_hugpczeros[0]/self.v if (self.v_hugzeros[0]/self.v < asymp) else self.v_hugpcmaxzeros[0]/self.v 
          upper_fin = self.v_pczero/self.v 
          hugpc = lambda y: self.hugoniot(y) - pc_curve(y)
          y_solute = self.find_root(lower_fin, upper_fin, hugpc, asymptote=asymp)
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

  def hugoniot(self, y):
    # ms = self.ms 
    # mc = self.mc 
    # ma = self.ma 
    # delta = self.pc/self.pg 
    # part1 = 0.5*(gamma_c + 1.)*(y - (gamma_c - 1.)/(gamma_c + 1.))
    # part2 = (gamma_c/(gamma_g*ms**2))*(1. + delta - (gamma_g - gamma_c)/(gamma_c*(gamma_g - 1.)*(1. - y)))
    # part3 = (gamma_c/ma)*(np.sqrt(y) - 1./(gamma_c*mc**2*(1. + np.sqrt(y))) + np.sqrt(y)/(gamma_g*ms**2*(1. - y)))
    # part4 = gamma_g*ms**2*(1. - y)/np.sqrt(y)
    # part5 = ((gamma_g - gamma_c)*np.sqrt(y)/(gamma_g - 1.) - gamma_c/ma)*np.sqrt(y)/(gamma_g*ms**2*(1. - y))
    
    J = self.J 
    M = self.M 
    E = self.E 
    v = self.v
    pg = self.pg
    ma = self.ma 
    factor = 1. - 1./(ma*np.sqrt(y))
    part1 = E/(v*y) - J*v*y*(0.5 - (gamma_c/(gamma_c - 1.))*factor) - (gamma_c/(gamma_c - 1.))*M*factor 
    part2 = (gamma_g/(gamma_g - 1.)) - (gamma_c/(gamma_c - 1.))*factor 
    return part1/(part2*pg)
    # return (part1 - part2 - part3)*part4/part5 # Returns p-bar for the hugoniot

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
    if self.regime in [1, 2, 3]:
      lower = 0.99*self.v_hugpczeros[0]/v if (self.v_hugzeros[0]/v < asymp) else self.v_hugzeros[0]/v 
    else:
      lower = 0.99*self.v_hugpczeros[0]/v
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
    delete_index = np.array([], dtype=int)
    for i, pe in enumerate(p_ver):
      if (pe < 0) or (pe > p_cap[i]):
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

  def adiabat(self, y):
    S = self.S 
    pg = self.pg 
    ma = self.ma 
    B = self.B
    part1 = S/np.cbrt((ma*np.sqrt(y) + gamma_g - 1.)**10)
    part2 = (gamma_g - 1.)*B**2*(2.*gamma_g*ma*np.sqrt(y) + 1. - gamma_g)/(gamma_g*(2.*gamma_g + 1.))
    return (part1 - part2)/pg # Returns p-bar for the adiabat

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
      print('No solution')
      v2 = np.array([])
      rho2 = np.array([])
      pg2 = np.array([])
      pc2 = np.array([])
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

    # The adiabat
    if (np.size(self.v_adhugzeros) > 0):
      y_adia = np.linspace(self.v_adhugzeros[0]/self.v, 1., 1000)

    # The reflected hugoniot
    if (np.size(self.v_hugpcmaxzeros) > 0) and (np.size(self.v_refpczero) > 0):
      if (np.size(self.v_ver) > 0):
          y_ref = np.linspace(self.v_hugpcmaxzeros[0]/self.v, self.v_ver[0]/self.v, 1000)
          y_ref2 = np.linspace(self.v_refpczero/self.v, self.v_ver[0]/self.v, 1000)
      else:
        y_ref = np.linspace(self.v_hugpcmaxzeros[0]/self.v, self.v_refpczero/self.v, 1000)

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
    
    if (np.size(self.v_adhugzeros) > 0):
      ax.plot(self.v*y_adia, self.pg*self.adiabat(y_adia), label='Adiabat')
      ax.scatter(self.v_adhugzeros, self.pg*self.adiabat(self.v_adhugzeros/self.v), marker='o', color='k')
    
    if (np.size(self.v_adrefhugzeros) > 0) and (np.size(self.v_final) > 0):
      for i, v_adrefhug in enumerate(self.v_adrefhugzeros):
        ax.plot(np.array([v_adrefhug, self.v_final[i]]), np.array([self.pg*self.adiabat(v_adrefhug/self.v), self.pg*self.hugoniot(self.v_final[i]/self.v)])) 
      ax.scatter(self.v_adrefhugzeros, self.pg*self.adiabat(self.v_adrefhugzeros/self.v), marker='o', color='k')
      ax.scatter(self.v_final, self.pg*self.hugoniot(self.v_final/self.v), marker='o', color='k')
    
    if (np.size(self.v_hugpcmaxzeros) > 0) and (np.size(self.v_refpczero) > 0):
      if (np.size(self.v_ver) > 0):
        if self.use_refhug:
          ax.plot(self.v*np.append(y_ref, y_ref2[::-1]), self.pg*np.append(self.refhug(y_ref), self.refhug2(y_ref2[::-1])), label='Reflect')
        else:
          ax.plot(self.v*np.append(y_ref, y_ref2[::-1]), self.pg*np.append(self.refhug2(y_ref), self.refhug(y_ref2[::-1])), label='Reflect')
      else:
        if self.use_refhug:
          ax.plot(self.v*y_ref, self.pg*self.refhug(y_ref), label='Reflect')
        else:
          ax.plot(self.v*y_ref, self.pg*self.refhug2(y_ref), label='Reflect')

    ax.scatter(self.v, self.pg, marker='o', color='k')

    # y1 = np.linspace(0.01, self.v_pgzero/self.v, 100)
    # y2 = np.linspace(0.01, self.v_pgzero/self.v, 100)
    # ax.plot(self.v*y1, self.pg*self.refhug(y1), label='ref_min')
    # ax.plot(self.v*y2, self.pg*self.refhug2(y2), label='ref_max')
    
    ax.set_xlim(0, self.v_pgzero)
    ax.set_ylim(0, self.M/self.pg)
    ax.margins(x=0, y=0)
    ax.legend(frameon=False)

    if don_want_axis:
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

    return fig
      
# End of class

###########################################
rho1 = 1. 
pg1 = 1. 
m1 = 1.01
n1 = 0.5 
beta1 = 0.01

upstream = mnbeta_to_gas(rho1, pg1, m1, n1, beta1)
# upstream = {}
# upstream['rho'] = 1.
# upstream['v'] = 8.58
# upstream['pg'] = 1.
# upstream['pc'] = 1.
# upstream['B'] = 1.

alter = gas_to_mnbeta(upstream['rho'], upstream['pg'], upstream['v'], upstream['pc'], upstream['B'])

shock = Shock(upstream['rho'], upstream['pg'], upstream['v'], upstream['pc'], upstream['B'])
# downstream = shock.solution()

# Plot diagram
fig = shock.shock_diagram(don_want_axis=False)
fig.savefig('./sh_struct.png', dpi=300)
plt.show(fig)

plt.close('all')
