"""
main(sub_dir)
    input:
        sub_dir : some subdirectory of ./net_results/ that contains a config.ini file (see README or ./HPC/config_template.ini for documentation)
            config.ini : see ./HPC/config_template.ini and README.md for more information
    steps:
        1) seed the random number generator
        2) extract config information from ./net_results/sub_dir
        3) reset max_workers to the number on the machine (might want to comment this out)
        4) load in data for the patient/seizure number
        5) downsample the data
        6) filter the data into the frequency band
        7) cut the data into all segments outlined in the low/high lists for the patient/seizure
        8) calculate univariate (and maybe multivariate) surrogates for that data
        9) chop the data into subsegments of time length sub_size*original_time_length (nseg subsegments minimally overlappying)
        10) Run the functional connectivity for each of the methods in the frequency band
        6b) repeat (6-10) for each frequency band in the config file 'a', 'b' etc.
        4b) repeat (1-10+6b) for each patient/seizure number in 'pats' in the config file
    Functional Connectivity:
        For more details, see Run() in ./MEGpy/connectivity.py
if __name__ == "__main__":
    sys.argv[1] : a subdirectory of ./net_results/, where the config.ini file is located (./HPC/config_template.ini or README.md for details on the config file)
    run main(sys.argv[1])
"""

# import some modules
# useful stuff
import os
from pathlib import Path
import sys
import time as clock
import multiprocessing as mp
import subprocess

# data science
import numpy as np
import scipy.fftpack

# save/load
from scipy.io import savemat
from scipy.io import loadmat
import hdf5storage as hdf5
import pickle
# config file (need pyyaml)
import yaml

# repository code
import preprocess as prep
import connectivity as conn

def main(sub_dir):
    """
    main(sub_dir)
    input:
        sub_dir : some subdirectory of ./net_results/ that contains a config.ini file (see README or ./HPC/config_template.ini for documentation)
            config.ini : see ./HPC/config_template.ini and README.md for more information
    steps:
        1) seed the random number generator
        2) extract config information from ./net_results/sub_dir
        3) reset max_workers to the number on the machine (might want to comment this out)
        4) load in data for the patient/seizure number
        5) downsample the data
        6) filter the data into the frequency band
        7) cut the data into all segments outlined in the low/high lists for the patient/seizure
        8) calculate univariate (and maybe multivariate) surrogates for that data
        9 chop the data into subsegments of time length sub_size*original_time_length (nseg subsegments minimally overlappying)
        10) Run the functional connectivity for each of the methods in the frequency band
        6b) repeat (6-10) for each frequency band in the config file 'a', 'b' etc.
        4b) repeat (1-10+6b) for each patient/seizure number in 'pats' in the config file
    Functional Connectivity:
        For more details, see Run() in ./MEGpy/connectivity.py
    """

    # Seed the random number generator
    np.random.seed()
    seed = np.random.randint(1, 2147483647)
    np.random.seed(seed)

    #input directory is always patients
    pat_dir= 'patients'
    # output directory (and location of config file) is always ./net_results/sub_dir
    res_dir= os.path.join('.','net_results',sub_dir)

    # Extract inputs from the config.ini file
    print(res_dir)

    methods = ['TE']
    nsurr=19
    TH=False
    BH=True
    freq_new = 1000
    sub_size=0.25
    nseg=10
    # Reset max workers to the number on the machine (this overrides the config.ini file max_workers)
    # This might not be desirable for your situation
    max_workers = mp.cpu_count()
    print('max_workers reset to :',max_workers)

    # Some of the methods print out some crap
    np.warnings.filterwarnings('ignore')


    # Save the seed in the same directory as the config.ini input
    # and the functional connectivity network output




        # Input and output file names (both have the same filename)
        # input is in ./patients
        # output is in ./net_results/sub_dir/
    fin = './patient1/original_data.mat'
    fout = './net_results/sub_dir/patient1/'

    network = {}
    # iterate over frequency bands 'a','b','c' (see config file and documentation)
    for cname in methods:

            # Load in patient data from ./patients
            # see documentation for proper format
            din    = hdf5.loadmat(fin)
            sfreq  = np.squeeze(din['sfreq'])
            times  = np.squeeze(din['times']) # array (time_points,)
            series = np.squeeze(din['virtual_grid_seizure_time_series'].T) # array (time_points, num_sources)
            srange = np.squeeze(din['seizure_range'])

            # Downsample
            #series, times = prep.downsample(series, times, sfreq, freq_new)

            # Bandstop filter the line noise (Aussie noise is 50 Hz)
            series = prep.butterworth_filter(series, freq_new, 49, 51, 'bandstop')

            # Bandpass filter in the range specified by inputs
            #lfreq  = 0.1
            #hfreq  = 30
            #series = prep.butterworth_filter(series, freq_new, lfreq, hfreq, 'band')

            # cut series and times into the segments outlined in the config file
            # series and times are a list the size of the low and high lists in the config file
            # series[epc] and times[epc] is the sources and times in the range [ low[epc] , high[epc] ] from config file
            low = []
            high = []
            low.append(0)
            high.append(1000)
            series, times = prep.time_segment(series, times, low, high)

            # Univariate and Multivariate Surrogates
            dataset_uni = []
            dataset_mul = []

            t0 = clock.time()
            # Univariate Surrogates (different ones for each epoch)
            surrogates_uni = prep.generate_iAAFT(series, nsurr, max_workers)
            # Iterate over time segments to analyse (different surrogates for each)
            for epc in range(len(series)):
                # dataset_uni has both real data and surrogates
                # dataset_uni[epc] is epoch epc, dataset_uni[epc] = array (time points x sources x number of surrogates + 1 [ the 1 being real data ])
                # dataset_uni[epc][:,:,0] is real data
                dataset_aux = np.concatenate((np.expand_dims(series[epc], axis= 2), surrogates_uni[epc]),axis= 2)
                dataset_uni.append(dataset_aux)

            # In this method, the data is chopped into nseg minimally overalapping subsegments of sub_size (a fraction) of the full epoch length
            # it does this for both real data and surrogates
            # dataset_uni[epc] = array (sub_size*time_points x sources x nseg subsegments x number of surrogates + 1 [the 1 being real data])
            # dataset_uni[epc][:,:,:,0] is real data
            dataset_uni = prep.chop_segment(dataset_uni, nseg, sub_size)
            print('Univariate Surrogates Created: ', "%.2f" %(clock.time()-t0),' s')

            # t0 = clock.time()
            # # Multivariate Surrogates (different ones for each epoch)
            # # We only need multivariate surrogates for MI (easy) or MI5 (MILCA) mutual information connectivity methods
            # if ('MI' in methods) or ('TE' in methods):
            #     surrogates_mul = prep.generate_multivariate_iAAFT(series,nsurr, max_workers)
            #     # Iterate over time segments to analyse to concatenate data and surrogates for each epoch
            #     for epc in range(len(series)):
            #         # dataset_mul has both real data and multivariate surrogates
            #         # dataset_mul[epc] is epoch epc, dataset_mul[epc] = array (time_points x sources x number of surrogates + 1 [the 1 being real data])
            #         # dataset_mul[epc][:,:,0] is real data
            #         dataset_aux = np.concatenate((np.expand_dims(series[epc], axis= 2), surrogates_mul[epc]),axis= 2)
            #         dataset_mul.append(dataset_aux)
            #
            #
            #     # In this method, the data is chopped into nseg minimally overalapping subsegments of sub_size (a fraction) of the full epoch length
            #     # it does this for both real data and surrogates
            #     # dataset_mul[epc] = array (sub_size*time_points x sources x nseg subsegments x number of surrogates + 1 [the 1 being real data])
            #     # dataset_mul[epc][:,:,:,0] is real data
            #     dataset_mul = prep.chop_segment(dataset_mul, nseg,sub_size)
            #     print('Multivariate Surrogates Created: ', "%.2f" %(clock.time()-t0),' s')
            #
            # t0 = clock.time()
            # Use the chopped up filtered data and chopped up filtered surrogates and calculate functional connectivity for each
            # This gets done for each functional connectivity method in the current frequency band list: Example:  methods['a']['method'] methods['b']['method']
            # output network['a'] is an array of (sources x sources x number of functional connectivity methods in frequency band 'a')
            network[cname] = conn.Run(dataset_uni, dataset_mul, methods, TH, BH, os.path.join(res_dir,'aux'), max_workers)
            print('Completed Network Calculations: ', "%.2f" %(clock.time()-t0),' s')

        # save the result (one for each patient/seizure in "pats" in the config file)
    savemat(fout+'te_0_1.mat', network)


if __name__ == "__main__":
    """
    if __name__ == "__main__":
    sys.argv[1] : a subdirectory of ./net_results/, where the config.ini file is located (./HPC/config_template.ini or README.md for details on the config file)
    run main(sys.argv[1])
    """
    sub_dir='res'
    main(sub_dir)