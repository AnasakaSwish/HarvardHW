'''
Created on 10 nov. 2017

@author: anase
'''
import requests
from functionsHW1 import get_poll_lxml
from functionsHW1 import _strip
from functionsHW1 import plot_colors
from functionsHW1 import rcp_poll_data
from functionsHW1 import poll_plot
from functionsHW1 import find_governor_races
from functionsHW1 import race_result
from functionsHW1 import id_from_url
from functionsHW1 import plot_race
from functionsHW1 import error_data
poll_id=1044
urlRCP='https://www.realclearpolitics.com/epolls/2010/governor/ca/california_governor_whitman_vs_brown-1113.html'
#poll_plot(poll_id)
#find_governor_races(urlRCP)
#plot_race("https://www.realclearpolitics.com/epolls/2010/governor/ca/california_governor_whitman_vs_brown-1113.html")

error_data(urlRCP)
