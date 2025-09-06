from datetime import date
from errno import WSAENOTCONN
import imp
from logging import error
from operator import ipow
from symbol import try_stmt
from textwrap import indent
from time import time
from turtle import pos
import chess
import chess.engine as engine
import pandas as pd
import numpy as np
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt





lost_df = pd.read_csv('lost_on_time.csv')
won_df = pd.read_csv('won_on_time.csv')


# print(lost_df.dtypes)

unlucky_df = lost_df.loc[(lost_df['Score'] > 0)]
lucky_df = won_df.loc[(won_df['Score'] > 0)]
lucky_df['date'] = pd.to_datetime(lucky_df['date'])
unlucky_df['date'] = pd.to_datetime(unlucky_df['date'])

print("Percentage of unlucky losses: " , (len(unlucky_df),  len(lost_df)))
print("Percentage of lucky wins: " , (len(lucky_df) , len(won_df)))

# print(tabulate(lucky_df, headers = 'keys', tablefmt = 'psql'))


data = (73, 56, 211+189)

# colors = sns.color_palette('pastel')[0:5] 
colors = ['#32a852', '#6394bf', '#748077' ]

fig = plt.figure()
fig.patch.set_facecolor('#4c473b')


# Change color of text
plt.rcParams['text.color'] = 'white'
 
# Create a circle at the center of the plot
my_circle=plt.Circle( (0,0), 0.7, color='#4c473b')
 
# Pieplot + circle on it
plt.pie(data, colors=colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()











# ---SCATTER PLOT ---
"""
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize = (9, 6))

sns.scatterplot(data = lucky_df, x = "date", y = 'Score' , hue='time', palette=['#769656'], legend=False)
# plt.scatter(data = unlucky_df, x = "date", y = 'Score', markerfacecolor = 'none' )

ax.set_yscale('log')
fig.patch.set_facecolor('#635e5a')
ax.set_facecolor('#312E2B')
ax.set_title("Games won on time when losing")
ax.set_ylabel("Score")  
ax.set_xlabel("Date")

plt.show()

"""