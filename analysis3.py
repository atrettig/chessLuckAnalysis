from random import sample
import pandas as pd
import numpy as np
from tabulate import tabulate
from ast import literal_eval
import traceback



def assess_luck(df):
 
    # Measures luck by comparing final game results with final position evaluations.
    
 
    luck_data = []
    
    for index, row in df.iterrows():

            
        atrettig_color = 'white' if df.at[index, 'white_name'] == 'atrettig' else 'black'
        atrettig_result = df.at[index, f'{atrettig_color}_result']
        atrettig_rating = df.at[index, f'{atrettig_color}_rating']

        final_eval = df.at[index, 'Final_Evaluation']
        date = df.at[index, 'date']

        evaluations = df.at[index, 'Evaluations']
        

        if atrettig_color == 'black':
            atrettig_eval = [-eval for eval in evaluations]
        else:
            atrettig_eval = evaluations.copy()



       
        luck_score = 0
        luck_category = 'normal'
        
        winning_threshold = 1.5
        losing_threshold = -1.5

        winning_moves = sum(1 for eval in atrettig_eval if eval > 0.5)
        losing_moves = sum(1 for eval in atrettig_eval if eval < -0.5)
        
        total_moves = len(atrettig_eval)
    

        pct_winning = winning_moves / total_moves
        pct_losing = losing_moves / total_moves
        pct_equal = (total_moves - winning_moves - losing_moves) / total_moves
        
        avg_evaluation = np.mean(atrettig_eval)

        if atrettig_result == 'win':
            if pct_winning < .20:
                # Won despite being in losing position 
                luck_category = 'lucky'

            elif pct_winning > pct_losing:
                
                luck_category = 'neutral'
        else:  # Loss or draw
            if pct_winning > .20:
                # Lost despite winning position - unlucky
                luck_category = 'unlucky'
            elif pct_winning < pct_losing:
                
                luck_score = 0
                luck_category = 'neutral'
                
        luck_data.append({
            'index': index,
            'pct_winning': round(pct_winning, 2),
            'pct_losing': round(pct_losing, 2),
            'pct_equal': round(pct_equal, 2),
            'avg_evaluation': round(avg_evaluation, 2),
            'color': atrettig_color,
            'result': atrettig_result,
            'luck_category': luck_category,
            'atrettig_rating':atrettig_rating,
            'date': date

        })
        
        print(f"Game {index}: {atrettig_color} {atrettig_result}, pct_winning={round(pct_winning, 2)},pct_losing={round(pct_losing, 2)},pct_equal={round(pct_equal, 2)},avg_evaluation={round(avg_evaluation, 2)}, date={date}, luck_category={luck_category}")
    
    return luck_data





import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def create_luck_stacked_bar_chart(df, rating_bracket_size=100, min_games_per_bracket=5):
    """
    Creates a stacked bar chart showing the distribution of unlucky, lucky, and neutral games
    across different rating brackets.
    
    """
    
    luck_data = assess_luck(df)
    luck_df = pd.DataFrame(luck_data)
    
    min_rating = luck_df['atrettig_rating'].min()
    max_rating = luck_df['atrettig_rating'].max()
    
    min_bracket = (min_rating // rating_bracket_size) * rating_bracket_size
    max_bracket = ((max_rating // rating_bracket_size) + 1) * rating_bracket_size
    
    brackets = []
    bracket_labels = []
    
    current_bracket = min_bracket
    while current_bracket < max_bracket:
        brackets.append(current_bracket)
        bracket_labels.append(f"{int(current_bracket)}-{int(current_bracket + rating_bracket_size - 1)}")
        current_bracket += rating_bracket_size
    
    def assign_bracket(rating):
        return (rating // rating_bracket_size) * rating_bracket_size
    
    luck_df['rating_bracket'] = luck_df['atrettig_rating'].apply(assign_bracket)
    
    bracket_luck_counts = luck_df.groupby(['rating_bracket', 'luck_category']).size().unstack(fill_value=0)
    
    total_games_per_bracket = bracket_luck_counts.sum(axis=1)
    bracket_luck_counts = bracket_luck_counts[total_games_per_bracket >= min_games_per_bracket]
    
    for category in ['unlucky', 'neutral', 'lucky']:
        if category not in bracket_luck_counts.columns:
            bracket_luck_counts[category] = 0
    
    bracket_luck_counts = bracket_luck_counts[['unlucky', 'neutral', 'lucky']]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {
        'unlucky': "#d0f114",    
        'neutral': '#95a5a6',   
        'lucky': '#2ecc71'      
    }
    
    bottom = np.zeros(len(bracket_luck_counts))
    
    for category in ['unlucky', 'neutral', 'lucky']:
        values = bracket_luck_counts[category].values
        ax.bar(range(len(bracket_luck_counts)), values, bottom=bottom, 
               label=category.capitalize(), color=colors[category], alpha=0.8)
        bottom += values
    
    ax.set_xlabel('Rating Bracket', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Games', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Lucky, Unlucky, and Neutral Games by Rating Bracket', 
                 fontsize=14, fontweight='bold', pad=20)
    
    bracket_labels_filtered = [f"{int(bracket)}-{int(bracket + rating_bracket_size - 1)}" 
                              for bracket in bracket_luck_counts.index]
    ax.set_xticks(range(len(bracket_luck_counts)))
    ax.set_xticklabels(bracket_labels_filtered, rotation=45, ha='right')
    
    ax.legend(title='Game Category', title_fontsize=11, fontsize=10, loc='upper right')
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    for i, bracket in enumerate(bracket_luck_counts.index):
        total_height = 0
        for category in ['unlucky', 'neutral', 'lucky']:
            height = bracket_luck_counts.loc[bracket, category]
            if height > 0:  # Only add label if there are games in this category
                ax.text(i, total_height + height/2, str(int(height)), 
                       ha='center', va='center', fontweight='bold', fontsize=9)
            total_height += height
        
        ax.text(i, total_height + 0.5, f'Total: {int(total_height)}', 
               ha='center', va='bottom', fontsize=8, style='italic')
    
    plt.tight_layout()
    
    print(f"\n=== LUCK DISTRIBUTION BY RATING BRACKET ===")
    print(f"Rating bracket size: {rating_bracket_size}")
    print(f"Minimum games per bracket: {min_games_per_bracket}")
    print(f"Total brackets shown: {len(bracket_luck_counts)}")
    print("\nBreakdown by bracket:")
    
    for bracket in bracket_luck_counts.index:
        bracket_label = f"{int(bracket)}-{int(bracket + rating_bracket_size - 1)}"
        total = bracket_luck_counts.loc[bracket].sum()
        unlucky = bracket_luck_counts.loc[bracket, 'unlucky']
        neutral = bracket_luck_counts.loc[bracket, 'neutral']
        lucky = bracket_luck_counts.loc[bracket, 'lucky']
        
        print(f"  {bracket_label}: {total} games "
              f"(Unlucky: {unlucky}, Neutral: {neutral}, Lucky: {lucky})")
    
    return fig






def analyze_games(df):

    print("\n=== LUCK ANALYSIS ===")
    luck_data = assess_luck(df)
    # create_luck_stacked_bar_chart(luck_data)


    
    # Add results to dataframe

    for i, data in enumerate(luck_data):
        idx = data['index']
        # df.at[idx, 'luck_score'] = data['luck_score']
        # df.at[idx, 'luck_category'] = data['luck_category']
        

    return df









df = pd.read_csv("secondAttempt/algoTrainingData.csv")
# df = pd.read_csv("secondAttempt/evaluated_games.csv")

df['Evaluations'] = df['Evaluations'].apply(literal_eval)
df['Final_Evaluation'] = df['Evaluations'].apply(lambda x: x[-1] if x and len(x) > 0 else 0)
# sample_df = df.sample(n=30)
# sample_df =  analyze_games(sample_df) 
df = analyze_games(df)

fig = create_luck_stacked_bar_chart(df, rating_bracket_size=50, min_games_per_bracket=3)
plt.show()

