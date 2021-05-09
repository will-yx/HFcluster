README

Requirements
- Windows OS
- Python 3.7 environment

Installation
- copy notebook, HFcluster.py, my_fcswrite.py, xkcd_hexcol.csv, dependencies.txt into the same folder
- in command prompt: pip install -r dependencies.txt

Please also download the following python modules from this website:
https://www.lfd.uci.edu/~gohlke/pythonlibs/

Python-igraph:
python_igraph-0.7.1.post6-cp37-cp37m-win_amd64.whl

louvain-igraph:
louvain-0.6.1-cp37-cp37m-win_amd64.whl

leidenalg:
leidenalg-0.7.0-cp37-cp37m-win_amd64.whl


CSV file structure
~~~~
The script by default in takes .csv format *tsv is compatible
Each csv file should represent one tissue region with cells as rows and parameters as columns

The default file structure is the output format of the Akoya or CRISP CODEX processor
Processed CODEX Runs should be saved in the same parent folder and designated in the Setup field as 'path'
Folder names will be read from the 'runs' as a list i.e. runs = ['run1 folder name', 'run2 folder name', ...]
Typically, the output file structure saves the compensated .csv files in the subdirectory /processed/segm/segm-1/fcs/compensated
- this can be changed by modifying the 'subdir' variable

Output directory ('outdir') can NOT be the same as the input directory

X, Y, Z should be exact to the column headers in your csv

IncludeList is a list of channels used for clustering.
- looks for column headers that contain the marker

Column headers of autofluorescence channels should contain "blank" or "Blank"
~~~~

