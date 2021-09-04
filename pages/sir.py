import scipy
from scipy.integrate import odeint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
import streamlit as st


def app():
################################################################################################################


  population = 30000000  # Population of the United States

  # Initial conditions for infected and recovered people
  initial_infected, initial_recovered = 1, 0

  # Initial conditions for everyone else.
  initial_everyone_else = population - initial_infected - initial_recovered

  initial_conditions = initial_everyone_else, initial_infected, initial_recovered

  n_days = 500  # Days over which to integrate
  time = np.linspace(0, n_days, n_days)

  # Contact Rate - We don't know this for coronavirus, so use it as a relative term for comparison.
  contact_rate = 0.25
  recovery_rate = 1/14  # Recovery Rate -

  # The SIR model, integreated over 500 days.


  def SIR(initial_conditions, t, population, contact_rate, recovery_rate):
      S, I, R = initial_conditions
      dS = -contact_rate*S*I/population
      dI = contact_rate*S*I/population - recovery_rate*I
      dR = recovery_rate*I
      return dS, dI, dR


  result = odeint(SIR, initial_conditions, time, args=(
      population, contact_rate, recovery_rate))
  S, I, R = result.T


  # What happens if we're in contact with fewer people?
  contact_rate = st.slider('Contact rate', 0.10, 1.0, 0.30, 0.05)


  result = odeint(SIR, initial_conditions, time, args=(
      population, contact_rate, recovery_rate))
  S, I, R = result.T

  fig = plt.figure(facecolor='w')
  ax = fig.add_subplot(111, axisbelow=True)
  ax.plot(time, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
  ax.plot(time, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
  ax.plot(time, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
  ax.set_xlabel('Days')
  ax.set_ylabel('Number (1000s)')
  legend = ax.legend()
  legend.get_frame().set_alpha(0.5)
  st.write(fig)
