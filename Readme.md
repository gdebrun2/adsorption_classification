**Environment Setup**

Install python 3.11.12 in a fresh environment:

```bash
conda create -n py311 python=3.11.12
```

Install ovito:

```bash
conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.13.0
```

The rest of the required packages can be installed via ```pip install -r requirements.txt```. It is necessary to do the python and ovito installs using conda and then switch to pip, as the ovito package is incredibly sensitive to environment setup.

There are a couple packages that must be manually built found in deps/. These aren't necessary for the production analysis, but were used in feature selection analysis that did not end up being useful.

To reproduce plots, a TeX distribution must be installed and discoverable by matplotlib. If not already installed, the easiest way to do this without sudo is to install the full-scheme TinyTeX distribution using curl, add it to your path and set the following env variables:

```bash
export TINYTEX_INSTALLER=TinyTeX-2
curl -fsSL https://yihui.org/tinytex/install-bin-unix.sh | sh
echo 'export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"' >> "$HOME/.bashrc"
echo 'export TEXMFHOME="$HOME/texmf"' >> "$HOME/.bashrc"
```

```export TINYTEX_INSTALLER=TinyTeX-2``` is supposed to install the full scheme but did not work for me. To install all packages do:

```bash
tlmgr install scheme-full # or scheme-medium
```

You may need to install ghostscript if it's not on the system:

```bash
conda install -c conda-forge ghostscript
```

This may be unresolvable with the defualt solver so switch to the classic solver via ```conda config --set solver classic ```


The requirements.txt can get messy with versioning and pointing to local wheels so here are the required packages:

```bash
pip install jupyterlab scikit-learn scipy pandas matplotlib tqdm scienceplots numba joblib plotly dash seaborn pillow kaleido
```

Plotly is awful so to save figures it generates you have to have chrome installed. Do:

```bash
mkdir ~/chrome
plotly_get_chrome --path ~/chrome
```

and add ```export BROWSER_PATH="~/chrome/chrome-linux64/chrome"``` to your .bashrc