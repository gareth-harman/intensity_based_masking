
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import os 
from scipy.stats import kde
from nibabel import cifti2
import sys 


########################################################
# Main class 
#   Assumes we are using a dtseries
#   Can be modified to work with any time series object
########################################################

class DTseries:
    
    def __init__(self, pathname, sub, outdir, scanner):
        
        # Path to file
        self.pathname = pathname
        self.sub = sub
        self.outdir = outdir
        self.scanner = scanner
        
        #Load time series and get temporal mean
        self.cifti = cifti2.load(self.pathname)
        self.mat = self.cifti.get_data()
        self.mat = np.array(self.mat.tolist())
        self.mat = np.transpose(self.mat)
        
        # Get the mean
        self.mat_mu = np.mean(self.mat, axis = 1)
        
        # Create gaussian kernel density estimate
        self.density = kde.gaussian_kde(self.mat_mu)

    # Identify the minima and thus threshold for intensity        
    def createMask(self, plot=True, interp_fact=1024):
        
        # Create a linspace vers of our data
        self.xgrid = np.linspace(self.mat_mu.min(), 
                                 self.mat_mu.max(), 
                                 interp_fact)
        
        # Recreate Y in this space 
        self.ygrid = self.density(self.xgrid)

        # Find the minima and corresponding X value
        self.found_min = argrelextrema(self.ygrid, np.less)
        self.x_thresh = np.max(self.xgrid[self.found_min[0]])
        
        # Plot if you want
        if plot:
            fig = plt.figure()
            plt.hist(self.mat_mu, 100)
            plt.vlines(self.x_thresh, 0, 2000)
            plt.title(os.path.basename(self.pathname.split('.')[0]))
            plt.savefig(self.outdir + '/allPlots/' + self.sub + '_' + self.scanner + '.png')
            plt.close(fig)

        # Creat our empty mask
        self.mask = np.zeros(self.mat_mu.shape)
        
        # Fill it where our thresh condition is met
        self.mask[np.where(self.mat_mu <= self.x_thresh)[0]] = 1.0
    
    # Write numpy mask to file
    def writeOut(self):
        
        self.matout = self.outdir + '/allMasks/' + self.sub + '_' + self.scanner + '_mask.txt'
        np.savetxt(self.matout, self.mask, fmt = '%i')

        

######################################################
# RUN
######################################################        
        
if __name__ == "__main__":
    
    # Validate argument length
    if len(sys.argv) != 5:
        sys.exit('Incorrect number of inputs: exiting...')
    
    dt_file = sys.argv[1]   # Path to DT file
    sub = sys.argv[2]       # Subject
    outdir = sys.argv[3]    # Outpath
    scanner = sys.argv[4]   # Scanner
    
    # Run 
    subDT = DTseries(dt_file, sub, outdir, scanner)
    subDT.createMask(plot=False)
    subDT.writeOut()
    
