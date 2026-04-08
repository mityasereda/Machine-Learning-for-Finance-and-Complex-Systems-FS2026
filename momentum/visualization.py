import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

def plot_strategy_performance(strat, config, save_path=None, strat_with_impact=None, labels=None):
    """Plot strategy performance and save if path is provided"""
    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    # Set default labels if not provided
    if labels is None:
        labels = ['Strategy', 'With Market Impact'] if strat_with_impact is not None else ['Strategy']
    
    # Plotting the AUM of the strategies and the passive S&P 500 exposure
    ax1.plot(strat.index, strat['AUM'], label=labels[0], linewidth=2, color='blue')
    if strat_with_impact is not None:
        ax1.plot(strat.index, strat_with_impact['AUM'], label=labels[1], linewidth=2, color='red')
    ax1.plot(strat.index, strat['AUM_SPX'], label='S&P 500', linewidth=1, color='gray', alpha=0.7)

    # Plotting the returns comparison
    ax2.plot(strat.index, strat['ret'].cumsum(), label=f'{labels[0]} Returns', linewidth=2, color='blue')
    if strat_with_impact is not None:
        ax2.plot(strat.index, strat_with_impact['ret'].cumsum(), label=f'{labels[1]} Returns', linewidth=2, color='red')
    ax2.plot(strat.index, strat['ret_spy'].cumsum(), label='S&P 500 Returns', linewidth=1, color='gray', alpha=0.7)

    # Formatting the plots
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle=':')
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0%}'))
    
    # Additional formatting for AUM plot
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.set_ylabel('AUM ($)')
    
    # Additional formatting for returns plot
    ax2.set_ylabel('Cumulative Returns (%)')
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    
    # Add titles
    plt.suptitle('Strategy Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_title('Asset Under Management', fontsize=12)
    ax2.set_title('Cumulative Returns', fontsize=12)
    
    # Add commission info
    plt.figtext(0.02, 0.02, f'Commission = ${config["backtesting"]["commission"]}/share', 
                fontsize=9, style='italic')

    # Adjust layout and save if path is provided
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def print_statistics(stats):
    """Print strategy statistics in a formatted way"""
    print("\nStrategy Statistics:")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key:.<30} {value:>10}")
    print("=" * 50) 