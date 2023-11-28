#!/usr/bin/env python
#
#

# for headless runs
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, getopt



def main():

    # Get command line arguments
    if( len(sys.argv) < 4 ):
        print("Usage: {0} chart_title output.png file0.csv [file1.csv file2.csv ...]".format(sys.argv[0]))
        sys.exit(2)
        

    chart_title=sys.argv[1]
    fig_fn=sys.argv[2]
    csv_file_list= sys.argv[3:]

    x_label="Size (p=m0)"
    y_label="Performance (GFLOP/s)"
    
    ymax = 1
    fig, ax = plt.subplots()


    for fn in csv_file_list:
        df = pd.read_csv(fn)
        xsize = df['m0']    
        res   = df['result']
        ax.plot(xsize, res,label=fn)
        ymax=max(ymax,max(res))

    ax.set(ylim=(0, ymax*1.5),
           xlabel=x_label, ylabel=y_label,
           title=chart_title)
    

    plt.legend()
    fig.savefig(fig_fn)
    #plt.show() # comment out for headless run

if __name__ == '__main__':
    main()
