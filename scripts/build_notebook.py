"""Build and execute the walkthrough notebook (logic lives in the package).

    python scripts/build_notebook.py            # Yahoo data (cached after first run)
    python scripts/build_notebook.py synthetic  # offline
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "yahoo"

CELLS = [
    ("md", "# VaR / ES model comparison: walkthrough\n"
           "Eight models, rolling 500-day window, back-tests and consistent ranking. "
           "All logic lives in the `marketrisk` package."),
    ("code", f'SOURCE = "{SOURCE}"\n'
             "import warnings; warnings.filterwarnings('ignore')\n"
             "import numpy as np, pandas as pd\n"
             "from marketrisk import data as D\n"
             "pd.set_option('display.precision', 3)\n"
             "R = (D.to_returns(D.load_prices(cache_dir='../data/raw')) if SOURCE == 'yahoo'\n"
             "     else D.make_synthetic(3000))\n"
             "w = D.equal_weights(R.shape[1])\n"
             "R.describe().T[['mean', 'std', 'min', 'max']].mul(100).round(2)"),
    ("md", "## Stylised facts: fat tails and volatility clustering"),
    ("code", "from scipy import stats\n"
             "r = D.portfolio_returns(R, w)\n"
             "print(f'excess kurtosis {stats.kurtosis(r):.1f}, skew {stats.skew(r):.2f}')\n"
             "print('autocorr of r  :', round(r.autocorr(), 3))\n"
             "print('autocorr of r^2:', round((r**2).autocorr(), 3))"),
    ("md", "## Rolling out-of-sample forecasts"),
    ("code", "from marketrisk.backtest import run_rolling, summary_table, period_exceptions, "
             "STRESS_PERIODS\n"
             "from marketrisk.models import default_models\n"
             "res = run_rolling(R, w, default_models(), window=500)\n"
             "summary_table(res)"),
    ("code", "period_exceptions(res, STRESS_PERIODS)"),
    ("md", "## Ranking with consistent scoring functions\n"
           "FZ0 is the joint (VaR, ES) loss; Diebold-Mariano compares each model with the best."),
    ("code", "from marketrisk.compare import loss_table, ranking\n"
             "ranking(loss_table(res))"),
    ("md", "## Custom portfolio: 60/40 equities/Treasuries"),
    ("code", "w6040 = np.zeros(R.shape[1]); cols = list(R.columns)\n"
             "w6040[cols.index('SPY')], w6040[cols.index('TLT')] = 0.6, 0.4\n"
             "res6040 = run_rolling(R, w6040, default_models(mc_sims=5000), window=500)\n"
             "summary_table(res6040)[['exceptions', 'expected', 'kupiec_p', 'christ_cc_p', "
             "'days_red_%']]"),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s)
                for k, s in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    NotebookClient(nb, timeout=1200, resources={"metadata": {"path": str(out)}}).execute()
    nbf.write(nb, out / "walkthrough.ipynb")
    print("built notebooks/walkthrough.ipynb")


if __name__ == "__main__":
    main()
