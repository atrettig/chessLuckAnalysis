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
from stockfish import Stockfish
df = pd.read_csv('games2.csv')
df = df.drop_duplicates()


# print(tabulate(df.head(90), headers = 'keys', tablefmt = 'psql'))




won_on_time_df = df.loc[(df['termination'] == 'atrettig won on time') & (df['time'] == '300')]


lost_on_time_df = df.loc[(df['termination'].str.contains("won on time")) & (df['time'] == '300')]
lost_on_time_df = lost_on_time_df[~lost_on_time_df.termination.str.contains('atrettig')]

# print(tabulate(won_on_time_df, headers = 'keys', tablefmt = 'psql'))


engine = chess.engine.SimpleEngine.popen_uci("stockfish_15_win_x64_popcnt\stockfish_15_x64_popcnt.exe")


"""

# ---ADDING ANALYSIS TO DF---

score_list = []
i = 0

for row in won_on_time_df.index:

    try:
        position = won_on_time_df['position'][row]

        board = chess.Board(position)

        # info = engine.analyse(board, chess.engine.Limit(depth=20))
        info = engine.analyse(board, chess.engine.Limit(time = 1))

        score = str(info['score'])
        print(i)
        if (score.__contains__('Cp(')):
            score = score.split('Cp(')
        else:
            score = score.split('Mate(')
        score = score[1].split(')')
        score = score[0]
        i += 1

        score_list.append(score)    
    except:
        print(error)
        
won_on_time_df.insert(2,"Score", score_list)


won_on_time_df.to_csv('won_on_time.csv')

# print(tabulate(lost_on_time_df.head(20), headers = 'keys', tablefmt = 'psql'))




"""

stockfish = Stockfish(path='stockfish_15_win_x64_popcnt\stockfish_15_x64_popcnt.exe')
stockfish.set_depth(15)




def eval(fen):
    position = stockfish.set_fen_position(fen)
    eval = stockfish.get_evaluation()

    return eval['value']/100

print(eval('r4k1r/3N1ppp/p2bpqb1/2pp4/Q2P4/2P5/PP1N1PPP/R4RK1 b - - 5 15'))


# ---TESTING ANALYSIS---


# board = chess.Board()


# board = chess.Board("r5k1/5pp1/p1q1p2p/Bp4r1/3RP3/2P4P/2Q3P1/3R3K b - - 7 41")
# info = engine.analyse(board, chess.engine.Limit(depth=20))
# print("White winning by 2.7"+"Score:", info['base'])
# info = engine.analyse(board, chess.engine.Limit(time=0.1))




 

  