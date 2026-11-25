"""raglab: the shared code behind the RAG-lab notebook series."""

import warnings

# tqdm warns in notebooks without ipywidgets; the progress bars aren't needed.
warnings.filterwarnings("ignore", message="IProgress not found")
