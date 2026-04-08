import os
import pandas as pd
import yaml
from data import get_data
from backtest import BacktestEngine
from visualization import plot_strategy_performance, print_statistics


def final_backtest_momentum(ticker, from_date, until_date):
    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)


    # Fetch and prepare data
    print("Fetching data...") 
    df_intra, df_daily = get_data(ticker, from_date, until_date) 
    print("Data preparation complete.")

    # Run backtest without market impact
    print("Running backtest without market impact...")
    config['backtesting']['market_impact']['enabled'] = False
    engine_no_impact = BacktestEngine(config)
    strat_no_impact = engine_no_impact.run(ticker, df_intra, df_daily)
    print("Backtest complete.")

    # Calculate and print statistics for no market impact
    stats_no_impact = engine_no_impact.calculate_statistics(strat_no_impact)
    print("\nStatistics without market impact:")
    print_statistics(stats_no_impact) 
    # Run backtest with market impact
    print("\nRunning backtest with market impact...")
    config['backtesting']['market_impact']['enabled'] = True
    engine_with_impact = BacktestEngine(config)
    strat_with_impact = engine_with_impact.run(ticker, df_intra, df_daily)
    print("Backtest complete.")

    # Calculate and print statistics for with market impact
    stats_with_impact = engine_with_impact.calculate_statistics(strat_with_impact)
    print("\nStatistics with market impact:")
    print_statistics(stats_with_impact)

    # Plot results together
    print("\nGenerating comparison plot...")
    plot_strategy_performance(
        strat_no_impact, 
        config, 
        save_path=f'results/{ticker}_momentum_strategy_performance_comparison.png',
        strat_with_impact=strat_with_impact,
        labels=['Without Market Impact', 'With Market Impact']
    )
    print(f"Plot saved as '{ticker}_momentum_strategy_performance_comparison.png'")

    # Save stats to csv
    strat_no_impact.to_csv(f'results/{ticker}_momentum_stats_no_impact.csv', index=True)
    strat_with_impact.to_csv(f'results/{ticker}_momentum_stats_with_impact.csv', index=True)
    return strat_no_impact, strat_with_impact

if __name__ == "__main__":
    import os

 
    # ETFs and stocks: Momentum-friendly and Momentum-resistant
    tickers = [
        # ETFs
        "SPY", "QQQ", "XLK", "MTUM", "XLF",  # momentum-friendly
        "TLT", "GLD", "XLP", "VYM", "USO",   # momentum-challenged

        # Stocks
        "NVDA", "AAPL", "MSFT", "META", "TSLA",  # momentum-friendly
        "JNJ", "KO", "T", "INTC", "XOM"          # momentum-challenged
    ]

    from_date  = "2022-05-09"
    until_date = "2022-11-09"

    for ticker in tickers:
        if os.path.exists(f'results/{ticker}_momentum_stats_no_impact.csv'):
            print(f"Skipping {ticker} because it already exists in results folder")
            continue
        print(f"\n===== {ticker} =====")
        final_backtest_momentum(ticker, from_date, until_date)

