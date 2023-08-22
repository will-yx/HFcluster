README

Requirements
- Python 3.6+ environment
- TissueSimGPU requires Windows or Linux with CUDA 10.1+

Installation
- git clone https://github.com/will-yx/HFcluster
- conda create -n HFcluster python=3.9
- conda activate HFcluster
- pip install -e HFcluster

For Windows if you are having problems compiling the the following modules (python-igraph, louvain, leidenalg) try downloading pre-compiled wheels from this website
https://www.lfd.uci.edu/~gohlke/pythonlibs/
or use conda to install them

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

