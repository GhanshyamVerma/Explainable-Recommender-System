import pandas as pd
import numpy as np
import os

import argparse



# os.chdir('TuckER/')
print("Current working directory : ",os.getcwd())

# Initializing Parser
parser = argparse.ArgumentParser()

def dir_path(directory):
    os.makedirs(directory, exist_ok = True)
    
def is_file(f_path):
    if not os.path.isfile(f_path):
        raise FileNotFoundError(f_path + "was not found in the directory")
        
parser.add_argument('--kg_path', type=str,help = "path to the safron generated kg .nt file")

parser.add_argument('--out_path', type=str,help = "output path inside tucker/data/{} " )

args = parser.parse_args()

dir_path(args.out_path)
is_file(args.kg_path)


df = pd.read_csv(args.kg_path)

train, validate, test = np.split(df.sample(frac=1), [int(.8*len(df)),int(.9*len(df))])


np.savetxt(args.out_path + '/train.txt',train.values, fmt='%s')
np.savetxt(args.out_path + '/test.txt', test.values, fmt='%s')
np.savetxt(args.out_path + '/valid.txt', validate.values, fmt='%s')