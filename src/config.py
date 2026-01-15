from pathlib import Path
Base = Path(__file__).parent.parent
data_dir = Base / 'data'
Results_dir = Base / 'results'
Results_dir.mkdir(exist_ok=True)
plots_dir = Results_dir / 'plots'
plots_dir.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)