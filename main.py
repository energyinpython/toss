import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from toss import TOSS

from pyrepo_mcda import normalizations as norms
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda.mcda_methods import TOPSIS
from pyrepo_mcda import distance_metrics as dists
from pyrepo_mcda import correlations as corrs



# bar (column) chart
def plot_barplot(df_plot, legend_title, comment = ''):
    """
    Visualization method to display column chart of alternatives rankings obtained with 
    different methods.
    Parameters
    ----------
        df_plot : DataFrame
            DataFrame containing rankings of alternatives obtained with different methods.
            The particular rankings are included in subsequent columns of DataFrame.
        title : str
            Title of the legend (Name of group of explored methods, for example MCDA methods or Distance metrics).
    
    Examples
    ----------
    >>> plot_barplot(df_plot, legend_title='MCDA methods')
    """
    step = 1
    list_rank = np.arange(1, len(df_plot) + 1, step)

    # colors = ['#1f77b4', 'orange']
    # color = colors, po stacked
    ax = df_plot.plot(kind='bar', width = 0.8, stacked=False, edgecolor = 'black', figsize = (10,5))
    ax.set_xlabel('Alternatives', fontsize = 12)
    ax.set_ylabel('Rank', fontsize = 12)
    ax.set_yticks(list_rank)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)
    y_ticks = ax.yaxis.get_major_ticks()
    ax.set_ylim(0, len(df_plot) + 1)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=4, mode="expand", borderaxespad=0., edgecolor = 'black', title = legend_title, fontsize = 12)

    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    legend_title = legend_title.replace("$", "")
    legend_title = legend_title.replace(" ", "_")
    plt.savefig('./results/' + 'bar_chart_' + legend_title + comment + '.eps')
    plt.show()


# radar chart
def plot_radar(data, title, comment = ''):
    """
    Visualization method to display rankings of alternatives obtained with different methods
    on the radar chart.

    Parameters
    -----------
        data : DataFrame
            DataFrame containing containing rankings of alternatives obtained with different 
            methods. The particular rankings are contained in subsequent columns of DataFrame.
        title : str
            Chart title

    Examples
    ----------
    >>> plot_radar(data, title)
    """
    fig=plt.figure()
    ax = fig.add_subplot(111, polar = True)

    for col in list(data.columns):
        labels=np.array(list(data.index))
        stats = data.loc[labels, col].values

        angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        # close the plot
        stats=np.concatenate((stats,[stats[0]]))
        angles=np.concatenate((angles,[angles[0]]))
    
        lista = list(data.index)
        lista.append(data.index[0])
        labels=np.array(lista)

        ax.plot(angles, stats, '-o', linewidth=2)
    
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.set_rgrids(np.arange(1, data.shape[0] + 1, 2))
    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.legend(data.columns, bbox_to_anchor=(1.0, 0.95, 0.4, 0.2), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('./results/' + 'radar_chart' + comment + '.png')
    plt.show()


# heat maps with correlations
def draw_heatmap(df_new_heatmap, title, typ='rank'):
    """
    Visualization method to display heatmap with correlations of compared rankings generated using different methods
    
    Parameters
    ----------
        data : DataFrame
            DataFrame with correlation values between compared rankings
        title : str
            title of chart containing name of used correlation coefficient
    
    Examples
    ---------
    >>> draw_heatmap(df_new_heatmap, title)
    """
    plt.figure(figsize = (8, 5))
    sns.set(font_scale = 1.4)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="RdYlGn",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('MCDA methods')
    plt.title('Correlation: ' + title)
    plt.tight_layout()
    title = title.replace("$", "")
    plt.savefig('./results/' + 'correlations_' + typ + '_' + title + '.png')
    plt.show()

# Create dictionary class
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value




def main():
    
    # Load decision matrix with performance values
    df = pd.read_csv('dataset/data_general.csv', index_col='Auto')
    df_data = df.iloc[:len(df) - 1, :]

    # Load criteria types
    types = df.iloc[len(df) - 1, :].to_numpy()
    
    # matrix
    matrix = df_data.to_numpy()

    # calculate criteria weights using CRITIC weighting method
    weights = mcda_weights.critic_weighting(matrix)

    # sustainability coefficient from matrix calculated based on standard deviation from normalized matrix
    n_matrix = norms.minmax_normalization(matrix, types)
    s = np.sqrt(np.sum(np.square(np.mean(n_matrix, axis = 0) - n_matrix), axis = 0) / n_matrix.shape[0])

    alt_names = [r'$A_{' + str(el) + '}$' for el in range(1, matrix.shape[0] + 1)]
    col_names = [r'$C_{' + str(el) + '}$' for el in range(1, matrix.shape[1] + 1)]

    # Save CRITIC weights to CSV file
    df_weights_CRITIC = pd.DataFrame(weights.reshape(1, -1), index = ['Weight CRITIC'], columns = col_names)
    df_weights_CRITIC.to_csv('results/weights_decision_matrix_CRITIC.csv')

    # Initialize TOSS method object
    toss = TOSS(normalization_method=norms.minmax_normalization)

    # Save s coefficient for each criterion to CSV file
    df_s = pd.DataFrame(s.reshape(1, -1), index = ['Sust coeff'], columns = col_names)
    df_s.to_csv('results/sust_coeff.csv')

    # Create dataframe for results
    rank_results = pd.DataFrame(index=alt_names)
    
    # TOSS
    pref_toss = toss(matrix, weights, types, s_coeff = s)
    rank_toss = rank_preferences(pref_toss, reverse = True)

    # TOPSIS - for comparison
    topsis = TOPSIS(normalization_method=norms.minmax_normalization, distance_metric=dists.euclidean)
    pref_t = topsis(matrix, weights, types)
    rank_t = rank_preferences(pref_t, reverse = True)

    rank_results['TOSS pref'] = pref_toss
    rank_results['TOPSIS pref'] = pref_t
    rank_results['TOSS rank'] = rank_toss
    rank_results['TOPSIS rank'] = rank_t

    rank_results = rank_results.rename_axis('Ai')
    rank_results.to_csv('./results/results_1.csv')

    # column figure 2-colored
    rank_results = rank_results.drop(columns = ['TOSS pref', 'TOPSIS pref'])
    rank_results = rank_results.rename(columns={"TOSS rank": "TOSS", "TOPSIS rank": "TOPSIS"})

    plot_barplot(rank_results, 'MCDA methods')

    plot_radar(rank_results, 'MCDA methods')
    


if __name__ == '__main__':
    main()