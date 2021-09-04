# big data
import COVID19Py
# python libraries
from datetime import datetime, timedelta
import json
import itertools
import time
import os
# data tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as p
import plotly.graph_objs as go
import pylab
import seaborn as sns
import pycountry
from PIL import Image
from IPython.display import HTML as html_print
# machine learning libraries
import scipy
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
# app
# from pages import utils
import streamlit as st

# all countries, and alpha 2 code for api query (global locations need alpha 3)
countries_and_codes = [
   ['Afghanistan', 'AF'], ['Albania', 'AL'], ['Algeria', 'DZ'], ['Andorra', 'AD'], ['Angola', 'AO'], 
   ['Antigua and Barbuda', 'AG'], ['Argentina', 'AR'], ['Armenia', 'AM'], ['Australia', 'AU'], ['Austria', 'AT'], 
   ['Azerbaijan', 'AZ'], ['Bahamas', 'BS'], ['Bahrain', 'BH'], ['Bangladesh', 'BD'], ['Barbados', 'BB'], 
   ['Belarus', 'BY'], ['Belgium', 'BE'], ['Belize', 'BZ'], ['Benin', 'BJ'], ['Bhutan', 'BT'], ['Bolivia', 'BO'], 
   ['Bosnia and Herzegovina', 'BA'], ['Botswana', 'BW'], ['Brazil', 'BR'], ['Brunei', 'BN'], ['Bulgaria', 'BG'], 
   ['Burkina Faso', 'BF'], ['Burma', 'MM'], ['Burundi', 'BI'], ['Cabo Verde', 'CV'], ['Cambodia', 'KH'], 
   ['Cameroon', 'CM'], ['Canada', 'CA'], ['Central African Republic', 'CF'], ['Chad', 'TD'], ['Chile', 'CL'], 
   ['China', 'CN'], ['Colombia', 'CO'], ['Congo (Brazzaville)', 'CG'], ['Congo (Kinshasa)', 'CD'], 
   ['Costa Rica', 'CR'], ["Cote d'Ivoire", 'CI'], ['Croatia', 'HR'], ['Cuba', 'CU'], ['Cyprus', 'CY'], 
   ['Czechia', 'CZ'], ['Denmark', 'DK'], ['Diamond Princess', 'XX'], ['Djibouti', 'DJ'], ['Dominica', 'DM'], 
   ['Dominican Republic', 'DO'], ['Ecuador', 'EC'], ['Egypt', 'EG'], ['El Salvador', 'SV'], 
   ['Equatorial Guinea', 'GQ'], ['Eritrea', 'ER'], ['Estonia', 'EE'], ['Eswatini', 'SZ'], ['Ethiopia', 'ET'], 
   ['Fiji', 'FJ'], ['Finland', 'FI'], ['France', 'FR'], ['Gabon', 'GA'], ['Gambia', 'GM'], ['Georgia', 'GE'], 
   ['Germany', 'DE'], ['Ghana', 'GH'], ['Greece', 'GR'], ['Grenada', 'GD'], ['Guatemala', 'GT'], ['Guinea', 'GN'], 
   ['Guinea-Bissau', 'GW'], ['Guyana', 'GY'], ['Haiti', 'HT'], ['Holy See', 'VA'], ['Honduras', 'HN'], 
   ['Hungary', 'HU'], ['Iceland', 'IS'], ['India', 'IN'], ['Indonesia', 'ID'], ['Iran', 'IR'], ['Iraq', 'IQ'], 
   ['Ireland', 'IE'], ['Israel', 'IL'], ['Italy', 'IT'], ['Jamaica', 'JM'], ['Japan', 'JP'], ['Jordan', 'JO'], 
   ['Kazakhstan', 'KZ'], ['Kenya', 'KE'], ['Korea, South', 'KR'], ['Kosovo', 'XK'], ['Kuwait', 'KW'], 
   ['Kyrgyzstan', 'KG'], ['Laos', 'LA'], ['Latvia', 'LV'], ['Lebanon', 'LB'], ['Liberia', 'LR'], ['Libya', 'LY'], 
   ['Liechtenstein', 'LI'], ['Lithuania', 'LT'], ['Luxembourg', 'LU'], ['MS Zaandam', 'XX'], ['Madagascar', 'MG'], 
   ['Malawi', 'MW'], ['Malaysia', 'MY'], ['Maldives', 'MV'], ['Mali', 'ML'], ['Malta', 'MT'], ['Mauritania', 'MR'], 
   ['Mauritius', 'MU'], ['Mexico', 'MX'], ['Moldova', 'MD'], ['Monaco', 'MC'], ['Mongolia', 'MN'], ['Montenegro', 'ME'], 
   ['Morocco', 'MA'], ['Mozambique', 'MZ'], ['Namibia', 'NA'], ['Nepal', 'NP'], ['Netherlands', 'NL'], ['New Zealand', 'NZ'], 
   ['Nicaragua', 'NI'], ['Niger', 'NE'], ['Nigeria', 'NG'], ['North Macedonia', 'MK'], ['Norway', 'NO'], ['Oman', 'OM'], 
   ['Pakistan', 'PK'], ['Panama', 'PA'], ['Papua New Guinea', 'PG'], ['Paraguay', 'PY'], ['Peru', 'PE'], ['Philippines', 'PH'], 
   ['Poland', 'PL'], ['Portugal', 'PT'], ['Qatar', 'QA'], ['Romania', 'RO'], ['Russia', 'RU'], ['Rwanda', 'RW'], 
   ['Saint Kitts and Nevis', 'KN'], ['Saint Lucia', 'LC'], ['Saint Vincent and the Grenadines', 'VC'], ['San Marino', 'SM'], 
   ['Saudi Arabia', 'SA'], ['Senegal', 'SN'], ['Serbia', 'RS'], ['Seychelles', 'SC'], ['Sierra Leone', 'SL'], 
   ['Singapore', 'SG'], ['Slovakia', 'SK'], ['Slovenia', 'SI'], ['Somalia', 'SO'], ['South Africa', 'ZA'], ['Spain', 'ES'], 
   ['Sri Lanka', 'LK'], ['Sudan', 'SD'], ['Suriname', 'SR'], ['Sweden', 'SE'], ['Switzerland', 'CH'], ['Syria', 'SY'], 
   ['Taiwan*', 'TW'], ['Tanzania', 'TZ'], ['Thailand', 'TH'], ['Timor-Leste', 'TL'], ['Togo', 'TG'], 
   ['Trinidad and Tobago', 'TT'], ['Tunisia', 'TN'], ['Turkey', 'TR'], ['US', 'US'], ['Uganda', 'UG'], ['Ukraine', 'UA'], 
   ['United Arab Emirates', 'AE'], ['United Kingdom', 'GB'], ['Uruguay', 'UY'], ['Uzbekistan', 'UZ'], ['Venezuela', 'VE'], 
   ['Vietnam', 'VN'], ['West Bank and Gaza', 'PS'], ['Zambia', 'ZM'], ['Zimbabwe', 'ZW']]

# APP
def app():
  # Add a title
  st.title('COVID19 predictions')
  st.sidebar.title("Sub-notification")
  st.sidebar.info('You can test predictions assuming which percentage of the official number of cases are not being reported. For example, if only 50% of cases are being reported, move the slider to the middle. Notification depends on the capacity health services have for testing the population, and may vary greatly from country to country, and even from region to region within a country.')
  notification_percentual = st.sidebar.slider(
        "Notification in %", 
        min_value=0,
        max_value=100,
        step=5,
        value=100)
 
  countries, codes, cases, world_cases_now, evolution_of_cases_worldwide = get_codes()

  st.header('Predict the spread of COVID-19')
  # pick your country

  try:
      select = ['Sudan','SD']
      # query selected country
      with st.spinner('Data is being prepared...'):
        time.sleep(3)
        country = get_data(countries_and_codes, select)
      # notification factor where official data == 100% accurate
      #notification_percentual = 100
      # create sidebar for sub-notification scenarios
      
      #show timeline
      first_day, df = timeline_of_cases_and_deaths(country, notification_percentual)
      # show also the data using a logarithic scale
      plot_logarithmic(country, notification_percentual)
      # plot  daily increase of cases
     
      # A brief theoretical explanation
      
     
      # Data & projections for cases, for today and the former 2 days
      pred = prediction_of_maximum_cases(df, notification_percentual)
      # Data & projections for deaths, for today and the former 2 days
      prediction_of_deaths(df, notification_percentual, pred)
      # Final considerations
      

      
  except Exception as e:
    st.write(e)



# - world data - alpha3 code for countries 
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_codes():
  # global, generic data
  df = pd.read_csv('https://covid.ourworldindata.org/data/ecdc/total_cases.csv')
  # lookup function
  def look(x):
      try:
          return pycountry.countries.search_fuzzy(x)[0].alpha_3
      except:
          return x
  # get countries
  countries = list(df)[2:]
  # last entry
  world_cases = df['World'].iloc[-1]
  evolution_of_cases_worldwide = df['World']
  #get cases
  cases=[]
  for item in df:
    if item in countries:
      # most recent is the last
      n = df[item].iloc[-1]
      cases.append(n)
  # get alpha 3 code for map locations
  iso3_codes = [look(c) for c in countries]
  return (countries, iso3_codes, cases, world_cases, evolution_of_cases_worldwide)







# - single country data - use alpha2 code for countries 
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_data(countries_and_codes, select):
  # instantiate wrapper to data api
  covid19 = COVID19Py.COVID19()
  # get data from Hopkins University
  country_data = covid19.getLocationByCountryCode([item[1] for item in countries_and_codes if item[0] == select[0]], timelines=True)
  return country_data




def plot_logarithmic(country, notification_percentual):
  plt.rcParams["font.family"] = "Times New Roman"
  plt.rcParams["font.size"] = "8"
  plt.rcParams['axes.grid'] = True
  # filter target data
  cases = country[0]["timelines"]["confirmed"]["timeline"]
  #print ('CASES', cases, 'ITEMS', cases.items())
  deaths =  country[0]["timelines"]["deaths"]["timeline"]
  # create dataframes for cases
  cases_df = pd.DataFrame(list(cases.items()),
               columns=['day', 'cases'])
  # apply subnotification percentage
  # if none was entered, it is == 1
  cases_df.cases = cases_df.cases*100/notification_percentual
  #a = [pow(10, i) for i in range(10)]
  fig = plt.figure()
  ax = fig.add_subplot(2, 1, 1)

  line, = ax.plot(cases_df.cases, color='blue', lw=1)
  ax.set_yscale('log')
  st.write(fig)
  st.write('**Logarithmic scale**')
  # st.pyplot()



# here we get the main dataframe
def timeline_of_cases_and_deaths(country, notification_percentual):
  # filter target data
  cases = country[0]["timelines"]["confirmed"]["timeline"]
  #print ('CASES', cases, 'ITEMS', cases.items())
  deaths =  country[0]["timelines"]["deaths"]["timeline"]
  # create dataframes for cases
  cases_df = pd.DataFrame(list(cases.items()),
               columns=['day', 'cases'])
  # apply subnotification percentage
  # if none was entered, it is == 1
  cases_df.cases = cases_df.cases*100/notification_percentual
  # create dataframes for deaths
  deaths_df = pd.DataFrame(list(deaths.items()),
               columns=['day', 'deaths'])
  # apply subnotification percentage
  # if none was entered, it is == 1
  #deaths_df.deaths*=sub_factor
  # merge into one single dataframe
  df = cases_df.merge(deaths_df, on='day')
  # add culumn for 'day'
  df = df.loc[:, ['day','deaths','cases']]
  # set first day of pendemic
  first_day = datetime(2020, 1, 2) - timedelta(days=1)
  # time format
  FMT = "%Y-%m-%dT%H:%M:%SZ"
  # strip and correct timelines
  df['day'] = df['day'].map(lambda x: (datetime.strptime(x, FMT) - first_day).days)
  # bring steramlit to the stage
  st.header('Timeline of cases and deaths')
  st.write('Day 01 of pandemic outbreak is January 1st, 2020.')
  st.write('(*For scenarios with sub-notification, click on side bar*)')
  # make numerical dataframe optional
  
  st.write('The data plots the following line chart for cases and deaths.')
  # show data on a line chart
  st.line_chart(df)
  return first_day, df




# formula for the model
def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))


# relevant functions
def predict_logistic_maximum(df, column = 'cases'):
      samples = df.shape[0]
      x_days = df['day'].tolist()
      y_cases = df[column].tolist()
      speed_guess = 2.5
      peak_guess = 120
      amplitude_guess = 250000
      if (column == 'deaths'):
        amplitude_guess = (amplitude_guess * speed_guess/100)   
      initial_guess =speed_guess, peak_guess, amplitude_guess

      fit = curve_fit(logistic_model, x_days, y_cases,p0=initial_guess,  maxfev=9999)

      # parse the result of the fit
      speed, x_peak, y_max = fit[0]
      speed_error, x_peak_error, y_max_error = [np.sqrt(fit[1][i][i]) for i in [0, 1, 2]]

      # find the "end date", as the x (day of year) where the function reaches 99.99%
      end = int(fsolve(lambda x: logistic_model(x, speed, x_peak, y_max) - y_max * 0.9999, x_peak))

      return x_days, y_cases, speed, x_peak, y_max, x_peak_error, y_max_error, end, samples


def print_prediction(df, label, column = 'cases'):
    x, y, speed, x_peak, y_max, x_peak_error, y_max_error, end, samples = predict_logistic_maximum(df, column)
    print(label + "'s prediction: " +
          "maximum " + column + " : " + str(np.int64(round(y_max))) +
          " (± " + str(np.int64(round(y_max_error))) + ")" +
          ", peak at calendar day: " + str(datetime(2020, 1, 2) + timedelta(days=int(round(x_peak)))) +
          " (± " + str(round(x_peak_error, 2)) + ")" +
          ", ending on day: " + str(datetime(2020, 1, 2) + timedelta(days=end)))

    st.markdown(label + "'s prediction: " + "maximum " + column + " : **" + str(np.int64(round(y_max))) + "** (± " + str(np.int64(round(y_max_error))) + ")" + ", peak at calendar day: " + str(datetime(2020, 1, 2) + timedelta(days=int(round(x_peak)))) + " (± " + str(round(x_peak_error, 2)) + ")" + ", ending on day: " + str(datetime(2020, 1, 2) + timedelta(days=end)))

    return y_max


def add_real_data(df, label,column = 'cases', color=None):
    x = df['day'].tolist()
    y = df[column].tolist()
    plt.scatter(x, y, label="Data (" + label + ")", c=color)


def add_logistic_curve(df, label,column = 'cases', **kwargs):
    x, _, speed, x_peak, y_max, _, _, end, _ = predict_logistic_maximum(df, column)
    x_range = list(range(min(x), end))
    plt.plot(x_range,
             [logistic_model(i, speed, x_peak, y_max) for i in x_range],
             label="Logistic model (" + label + "): " + str(int(round(y_max))),
             **kwargs)
    return y_max


def label_and_show_plot(plt, title, y_max=None):
    plt.title(title)
    plt.xlabel("Days since 1 January 2020")
    plt.ylabel("Total number of people")
    if (y_max):
        plt.ylim(0, y_max * 1.1)
    plt.legend()
    plt.show()


def prediction_of_maximum_cases(df, notification_percentual):
  plt.figure(figsize=(12, 8))
  add_real_data(df[:-2], "2 days ago")
  add_real_data(df[-2:-1], "yesterday")
  add_real_data(df[-1:], "today")
  add_logistic_curve(df[:-2], "2 days ago", dashes=[8, 8])
  add_logistic_curve(df[:-1], "yesterday", dashes=[4, 4])
  y_max = add_logistic_curve(df, "today")
  label_and_show_plot(plt, "Best logistic fit with the freshest data", y_max)
  # A bit more theory 
  st.header('Prediction of maximum cases')
  # considering the user entered notification values
  if notification_percentual == 1:
    st.markdown("With sub-notification of 0%.")
  else:
    st.markdown("With sub-notification of " + str(int(round(100 - notification_percentual))) + " %.")

  st.write('At high time values, the number of infected people gets closer and closer to *c* and that’s the point at which we can say that the infection has ended. This function has also an inflection point at *b*, that is the point at which the first derivative starts to decrease (i.e. the peak after which the infection starts to become less aggressive and decreases).')
  # plot
  st.pyplot(clear_figure=False)
  # fit the data to the model (find the model variables that best approximate)
  st.subheader('Predictions as of *today*, *yesterday* and *2 days ago*')

  print_prediction(df[:-2], "2 days ago")
  print_prediction(df[:-1], "yesterday")
  pred = print_prediction(df, "today")
  # PREDICTION 1
  st.header('Infection stabilization')
  st.markdown("Predictions as of today, the total infection should stabilize at **" + str(int(round(pred))) + "** cases.")
  return int(round(pred))


def prediction_of_deaths(df, notification_percentual, pred):


  # With subotification, deaths prediction must be within the range of 0.5% and 3.0% of total cases
  if notification_percentual < 100:
    st.write('With the present notification value of ' + str(notification_percentual) + "%, we apply the global mortality rate of 3.5% of total cases.")
    st.markdown('[COVID-19 Global Mortality Rate](https://www.worldometers.info/coronavirus/coronavirus-death-rate/)')
    prediction_of_deaths_3_100 = pred*3.5/100
    st.markdown("- Considering maximum death rate being around 3.5% of the total number of cases, we should expect ** " + str(int(round(prediction_of_deaths_3_100))) + "** deaths")
  else:
    plt.figure(figsize=(12, 8))
    add_real_data(df[:-2], "2 days ago", column = 'deaths')
    add_real_data(df[-2:-1], "yesterday", column = 'deaths')
    add_real_data(df[-1:], "today", column = 'deaths')
    add_logistic_curve(df[:-2], "2 days ago",column='deaths', dashes=[8, 8])
    add_logistic_curve(df[:-1], "yesterday",column='deaths', dashes=[4, 4])
    y_max = add_logistic_curve(df, "today", column='deaths')
    label_and_show_plot(plt, "Best logistic fit with the freshest data", y_max)

    st.header('Prediction of deaths')
    st.pyplot(clear_figure=False)

    st.subheader('Predictions as of *today*, *yesterday* and *2 days ago*')

    print_prediction(df[:-2], "2 days ago", 'deaths')
    print_prediction(df[:-1], "yesterday", 'deaths')
    pred = print_prediction(df, "today", 'deaths')
    print()
    html_print("As of today, the total deaths should stabilize at <b>" + str(int(round(pred))) + "</b>")
    # PREDICTION 2
    st.header('Deaths stabilization')
    st.markdown("As of today, the total number of deaths should stabilize at **" + str(int(round(pred))) + "** cases.")


#if __name__ == '__main__':
 # main()

































