import os 
import glob
if os.path.exists('Results/BIN'):
    os.system('rm -r Results/BIN')
os.makedirs('Results/BIN', exist_ok=True)
for csv in glob.glob('output/base/*/*.csv'):
    os.system(f'cp {csv} Results/BIN')

for bin_folder in glob.glob('output/base/*/*/gsbin'):
    new_bin_folder = bin_folder.replace('output/base/','Results/BIN/')
    os.makedirs(os.path.dirname(new_bin_folder), exist_ok=True)
    os.system(f'cp -r {bin_folder} {new_bin_folder}')

for yaml in glob.glob('output/base/*/*/*.yaml'):
    new_yaml = yaml.replace('output/base/','Results/BIN/')
    os.system(f'cp -r {yaml} {new_yaml}')
    
os.system('zip -r Results/BIN.zip Results/BIN')
