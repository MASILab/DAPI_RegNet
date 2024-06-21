import pandas as pd
import numpy as np
import argparse
import os

#This is for GammaGateR classification:

#Get the csv files as arguments using argparse

parser = argparse.ArgumentParser(description='Classify cells using GammaGateR')
parser.add_argument('tissue_name', type=str, help='Name of the tissue')
path='/fs5/p_masi/rudravg/MxIF_Vxm_Registered_V2/'


args = parser.parse_args()

#Read the csv files
tissue_name = args.tissue_name
registered = f'{path}{tissue_name}/{tissue_name}_post_all_V2_registered.csv'
registered = pd.read_csv(registered)
unregistered = f'{path}{tissue_name}/{tissue_name}_post_all_V2_unregistered.csv'
unregistered = pd.read_csv(unregistered)

#For columns starting from 3 and ending in last column -1 set True if the value is greater than 0.5, False otherwise
for i in range(0, len(registered.columns)-1):
    registered[registered.columns[i]] = registered[registered.columns[i]] > 0.5
    unregistered[unregistered.columns[i]] = unregistered[unregistered.columns[i]] > 0.5

#For consistency
registered.columns=registered.columns.str.upper()
unregistered.columns=unregistered.columns.str.upper()

registered.columns = registered.columns.str.replace('NZNORM_MEAN', 'Mean')
unregistered.columns = unregistered.columns.str.replace('NZNORM_MEAN', 'Mean')

#Add a column to the left called Instances and start the second row from 1 to the end
registered.insert(0, 'Instances', range(1, 1 + len(registered)))
unregistered.insert(0, 'Instances', range(1, 1 + len(unregistered)))

#Group Epi+
#Make a new column called Epi+ and set it to True if NAKATPASE or PANCK or CGA
registered['Epi+'] = registered['Mean_NAKATPASE'] | registered['Mean_PANCK'] | registered['Mean_CGA']
unregistered['Epi+'] = unregistered['Mean_NAKATPASE'] | unregistered['Mean_PANCK'] | unregistered['Mean_CGA']

#Group Stroma+
#Make a new column called Stroma+ and set it True if Vimentin or SMA True
registered['Stroma+'] = registered['Mean_VIMENTIN'] | registered['Mean_SMA']
unregistered['Stroma+'] = unregistered['Mean_VIMENTIN'] | unregistered['Mean_SMA']

#Exclude nuclei that are both Epi+ and Stroma+
#Remove rows where Epi+ and Stroma+ are both True
registered = registered[~(registered['Epi+'] & registered['Stroma+'])]
unregistered = unregistered[~(unregistered['Epi+'] & unregistered['Stroma+'])]

#Group Immune+ 
#Make a new column called Immune+ and set it to True if CD3 or CD8 or CD20 or CD68 or CD45 or CD4 or CD11B or LYSOZYME or CD3D True
registered['Immune+'] =  registered['Mean_CD8'] | registered['Mean_CD20'] | registered['Mean_CD68'] | registered['Mean_CD45'] | registered['Mean_CD4'] | registered['Mean_CD11B'] | registered['Mean_LYSOZYME'] | registered['Mean_CD3D']
unregistered['Immune+'] = unregistered['Mean_CD8'] | unregistered['Mean_CD20'] | unregistered['Mean_CD68'] | unregistered['Mean_CD45'] | unregistered['Mean_CD4'] | unregistered['Mean_CD11B'] | unregistered['Mean_LYSOZYME'] | unregistered['Mean_CD3D']

#Remove Immune conflicts for Macrophages
#Remove rows where (CD68+ and CD3d+), (CD68+ and CD20+),(CD68+ and CD4+), (CD68+ and CD8+), or (CD68+ and CD11B+)
conflict_conditions = (registered['Mean_CD68'] & registered['Mean_CD3D']) | (registered['Mean_CD68'] & registered['Mean_CD20']) | (registered['Mean_CD68'] & registered['Mean_CD4']) | (registered['Mean_CD68'] & registered['Mean_CD8']) | (registered['Mean_CD68'] & registered['Mean_CD11B'])
registered = registered[~conflict_conditions]

conflict_conditions = (unregistered['Mean_CD68'] & unregistered['Mean_CD3D']) | (unregistered['Mean_CD68'] & unregistered['Mean_CD20']) | (unregistered['Mean_CD68'] & unregistered['Mean_CD4']) | (unregistered['Mean_CD68'] & unregistered['Mean_CD8']) | (unregistered['Mean_CD68'] & unregistered['Mean_CD11B'])
unregistered = unregistered[~conflict_conditions]


#Remove Immune conflicts for Monocytes
#Remove rows where (CD11B+ and CD3d+), (CD11B+ and CD20+),(CD11B+ and CD4+), (CD11B+ and CD8+), or (CD11B+ and CD68+)
conflict_conditions = (registered['Mean_CD11B'] & registered['Mean_CD3D']) | (registered['Mean_CD11B'] & registered['Mean_CD20']) | (registered['Mean_CD11B'] & registered['Mean_CD4']) | (registered['Mean_CD11B'] & registered['Mean_CD8']) | (registered['Mean_CD11B'] & registered['Mean_CD68'])
registered = registered[~conflict_conditions]

conflict_conditions = (unregistered['Mean_CD11B'] & unregistered['Mean_CD3D']) | (unregistered['Mean_CD11B'] & unregistered['Mean_CD20']) | (unregistered['Mean_CD11B'] & unregistered['Mean_CD4']) | (unregistered['Mean_CD11B'] & unregistered['Mean_CD8']) | (unregistered['Mean_CD11B'] & unregistered['Mean_CD68'])
unregistered = unregistered[~conflict_conditions]

#Remove Immune conflicts for B cells
#Remove rows where (CD20+ and CD3d+), (CD20+ and CD4+), or (CD20+ and CD8+)
conflict_conditions = (registered['Mean_CD20'] & registered['Mean_CD3D']) | (registered['Mean_CD20'] & registered['Mean_CD4']) | (registered['Mean_CD20'] & registered['Mean_CD8'])
registered = registered[~conflict_conditions]

conflict_conditions = (unregistered['Mean_CD20'] & unregistered['Mean_CD3D']) | (unregistered['Mean_CD20'] & unregistered['Mean_CD4']) | (unregistered['Mean_CD20'] & unregistered['Mean_CD8'])
unregistered = unregistered[~conflict_conditions]

#Remove Immune conflicts for helper T cells and cytotoxic T cells
#Remove rows where  (CD3d- and CD45- and CD4+), (CD3d- and CD45- and CD8+), or (CD4+ and CD8+)
conflict_conditions = (registered['Mean_CD3D'] & registered['Mean_CD45'] & registered['Mean_CD4']) | (registered['Mean_CD3D'] & registered['Mean_CD45'] & registered['Mean_CD8']) | (registered['Mean_CD4'] & registered['Mean_CD8'])
registered = registered[~conflict_conditions]

conflict_conditions = (unregistered['Mean_CD3D'] & unregistered['Mean_CD45'] & unregistered['Mean_CD4']) | (unregistered['Mean_CD3D'] & unregistered['Mean_CD45'] & unregistered['Mean_CD8']) | (unregistered['Mean_CD4'] & unregistered['Mean_CD8'])
unregistered = unregistered[~conflict_conditions]

#Group Progenitor Cells
#Make a new column called Progenitor and set it True if SOX9+ or OLFM4+
registered['Progenitor'] = registered['Mean_SOX9'] | registered['Mean_OLFM4']
unregistered['Progenitor'] = unregistered['Mean_SOX9'] | unregistered['Mean_OLFM4']


#Exclude instances that are not in either Epi or Stroma
#Remove rows where Epi+ and Stroma+ are both False
registered = registered[(registered['Epi+'] | registered['Stroma+'])]
unregistered = unregistered[(unregistered['Epi+'] | unregistered['Stroma+'])]


#Remove conflicts for enteroendocrine across all instances
#Remove rows where (CgA+ and Immune+), (CgA+ and SMA+), (CgA+ and Progenitor+)
conflict_conditions = (registered['Mean_CGA'] & registered['Immune+']) | (registered['Mean_CGA'] & registered['Stroma+']) | (registered['Mean_CGA'] & registered['Progenitor'])
registered = registered[~conflict_conditions]

conflict_conditions = (unregistered['Mean_CGA'] & unregistered['Immune+']) | (unregistered['Mean_CGA'] & unregistered['Stroma+']) | (unregistered['Mean_CGA'] & unregistered['Progenitor'])
unregistered = unregistered[~conflict_conditions]


#Remove conflicts for fibroblasts across all instances
#Remove rows where  (SMA+ and Immune+)
conflict_conditions = (registered['Mean_SMA'] & registered['Immune+'])
registered = registered[~conflict_conditions]

conflict_conditions = (unregistered['Mean_SMA'] & unregistered['Immune+'])
unregistered = unregistered[~conflict_conditions]

#Remove conflicts for progenitors across all instances
#Remove rows where (Progenitor+ and Immune+)
conflict_conditions = (registered['Progenitor'] & registered['Immune+'])
registered = registered[~conflict_conditions]

conflict_conditions = (unregistered['Progenitor'] & unregistered['Immune+'])
unregistered = unregistered[~conflict_conditions]

#Remove instances that are negative for all large groupings
#Remove rows where Epi+, Stroma+, Immune+, Progenitor+ are all False
registered = registered[(registered['Epi+'] | registered['Stroma+'] | registered['Immune+'] | registered['Progenitor'])]
unregistered = unregistered[(unregistered['Epi+'] | unregistered['Stroma+'] | unregistered['Immune+'] | unregistered['Progenitor'])]

#Remove any immune instances from Epi+ groups
#Remove rows where Epi+ and Immune+ are both True
registered = registered[~(registered['Epi+'] & registered['Immune+'])]
unregistered = unregistered[~(unregistered['Epi+'] & unregistered['Immune+'])]

#Final annotation for enteroendocrine
#Make a new column called Enteroendocrine and set it True if CgA+ and Progenitor- and Epi+ is True
registered['final_Enteroendocrine'] = registered['Mean_CGA'] & ~registered['Progenitor'] & registered['Epi+']
unregistered['final_Enteroendocrine'] = unregistered['Mean_CGA'] & ~unregistered['Progenitor'] & unregistered['Epi+']


#Final annotation for enterocytes
#Make a new column called Enterocytes and set it True if CgA- and Progenitor- and Epi+ is True
registered['final_Enterocytes'] = ~registered['Mean_CGA'] & ~registered['Progenitor'] & registered['Epi+']
unregistered['final_Enterocytes'] = ~unregistered['Mean_CGA'] & ~unregistered['Progenitor'] & unregistered['Epi+']

#Group fibroblasts/stromal(undetermined)
#Make a new column called Fibroblasts/Stromal(undetermined) and set it True if Stroma+ and Immune-
registered['Fibroblasts/Stromal(undetermined)'] = registered['Stroma+'] & ~registered['Immune+']
unregistered['Fibroblasts/Stromal(undetermined)'] = unregistered['Stroma+'] & ~unregistered['Immune+']

#Final annotation for fibroblasts
#Make a new column called Fibroblasts and set it True if Fibroblasts/Stromal(undetermined)+ and SMA+ and Progenitor-
registered['final_Fibroblasts'] = registered['Fibroblasts/Stromal(undetermined)'] & registered['Mean_SMA'] & ~registered['Progenitor']
unregistered['final_Fibroblasts'] = unregistered['Fibroblasts/Stromal(undetermined)'] & unregistered['Mean_SMA'] & ~unregistered['Progenitor']

#Final annotation for stromal(undetermined)
#Make a new column called Stromal and set it True if Fibroblasts/Stromal(undetermined)+ and SMA- and Progenitor-
registered['final_Stromal'] = registered['Fibroblasts/Stromal(undetermined)'] & ~registered['Mean_SMA'] & ~registered['Progenitor']
unregistered['final_Stromal'] = unregistered['Fibroblasts/Stromal(undetermined)'] & ~unregistered['Mean_SMA'] & ~unregistered['Progenitor']

#Final annotation for myleoid
#Make a new column called Myeloid and set it True if Immune+ and (Lysozyme+ and CD68- and CD11B- and Progenitor- and CD20-) and CD3d- and CD8- and CD4-)
registered['final_Myeloid'] = registered['Immune+'] & (registered['Mean_LYSOZYME'] & ~registered['Mean_CD68'] & ~registered['Mean_CD11B'] & ~registered['Mean_CD20']) & ~registered['Mean_CD3D'] & ~registered['Mean_CD8'] & ~registered['Mean_CD4']
unregistered['final_Myeloid'] = unregistered['Immune+'] & (unregistered['Mean_LYSOZYME'] & ~unregistered['Mean_CD68'] & ~unregistered['Mean_CD11B'] & ~unregistered['Mean_CD20']) & ~unregistered['Mean_CD3D'] & ~unregistered['Mean_CD8'] & ~unregistered['Mean_CD4']

#Final annotation for helper T
#Make a new column called Helper T and set it True if Immune+ and (CD4+ and Progenitor-)
registered['final_Helper T'] = registered['Immune+'] & (registered['Mean_CD4'] & ~registered['Progenitor'])
unregistered['final_Helper T'] = unregistered['Immune+'] & (unregistered['Mean_CD4'] & ~unregistered['Progenitor'])

#Final annotation for cytotoxic T
#Make a new column called Helper T and set it True if Immune+ and (CD8+ and Progenitor-)
registered['final_Cytotoxic T'] = registered['Immune+'] & (registered['Mean_CD8'] & ~registered['Progenitor'])
unregistered['final_Cytotoxic T'] = unregistered['Immune+'] & (unregistered['Mean_CD8'] & ~unregistered['Progenitor'])

#Final annotation for T-cell receptor
#Make a new column called T-cell receptor and set it True if Immune+ and (CD3d+ and CD4- and CD8-)
registered['final_T-cell receptor'] = registered['Immune+'] & (registered['Mean_CD3D'] & ~registered['Mean_CD4'] & ~registered['Mean_CD8'])
unregistered['final_T-cell receptor'] = unregistered['Immune+'] & (unregistered['Mean_CD3D'] & ~unregistered['Mean_CD4'] & ~unregistered['Mean_CD8'])

#Final annotation for monocyte
#Make a new column called Monocyte and set it True if Immune+ and (CD11b+ and CD3d- and Progenitor- and CD4- and CD8-)
registered['final_Monocyte'] = registered['Immune+'] & (registered['Mean_CD11B'] & ~registered['Mean_CD3D'] & ~registered['Progenitor'] & ~registered['Mean_CD4'] & ~registered['Mean_CD8'])
unregistered['final_Monocyte'] = unregistered['Immune+'] & (unregistered['Mean_CD11B'] & ~unregistered['Mean_CD3D'] & ~unregistered['Progenitor'] & ~unregistered['Mean_CD4'] & ~unregistered['Mean_CD8'])

#Final annotation for macrophage
#Make a new column called Macrophage and set it True if Immune+ and  (CD68+ and CD3d- and Progenitor- and CD4- and CD8-)
registered['final_Macrophage'] = registered['Immune+'] & (registered['Mean_CD68'] & ~registered['Mean_CD3D'] & ~registered['Progenitor'] & ~registered['Mean_CD4'] & ~registered['Mean_CD8'])
unregistered['final_Macrophage'] = unregistered['Immune+'] & (unregistered['Mean_CD68'] & ~unregistered['Mean_CD3D'] & ~unregistered['Progenitor'] & ~unregistered['Mean_CD4'] & ~unregistered['Mean_CD8'])

#Final annotation for B cell
#Make a new column called B cell and set it True if Immune+ and  (CD20+ and CD68- and CD3d- and Progenitor- and CD4- and CD8-)
registered['final_B cell'] = registered['Immune+'] & (registered['Mean_CD20'] & ~registered['Mean_CD68'] & ~registered['Mean_CD3D'] & ~registered['Progenitor'] & ~registered['Mean_CD4'] & ~registered['Mean_CD8'])
unregistered['final_B cell'] = unregistered['Immune+'] & (unregistered['Mean_CD20'] & ~unregistered['Mean_CD68'] & ~unregistered['Mean_CD3D'] & ~unregistered['Progenitor'] & ~unregistered['Mean_CD4'] & ~unregistered['Mean_CD8'])

#Final annotation for leukocyte
#Make a new column called Leukocyte and set it True if Immune+ and  (CD45+ and CD20- and CD68- and CD3d- and Progenitor- and CD4- and CD8- and CD11B- and Lysozyme-)
registered['final_Leukocyte'] = registered['Immune+'] & (registered['Mean_CD45'] & ~registered['Mean_CD20'] & ~registered['Mean_CD68'] & ~registered['Mean_CD3D'] & ~registered['Progenitor'] & ~registered['Mean_CD4'] & ~registered['Mean_CD8'] & ~registered['Mean_CD11B'] & ~registered['Mean_LYSOZYME'])
unregistered['final_Leukocyte'] = unregistered['Immune+'] & (unregistered['Mean_CD45'] & ~unregistered['Mean_CD20'] & ~unregistered['Mean_CD68'] & ~unregistered['Mean_CD3D'] & ~unregistered['Progenitor'] & ~unregistered['Mean_CD4'] & ~unregistered['Mean_CD8'] & ~unregistered['Mean_CD11B'] & ~unregistered['Mean_LYSOZYME'])

#Final annotation for Progenitor
#Make a new column called Progenitor and set it True if Progenitor+
registered['final_Progenitor'] = registered['Progenitor']
unregistered['final_Progenitor'] = unregistered['Progenitor']

#Print the number of cells in each category
# print("Number of cells in registered:")
# print(registered['final_Enteroendocrine'].value_counts())
# print(registered['final_Enterocytes'].value_counts())
# print(registered['final_Fibroblasts'].value_counts())
# print(registered['final_Stromal'].value_counts())
# print(registered['final_Myeloid'].value_counts())
# print(registered['final_Helper T'].value_counts())
# print(registered['final_Cytotoxic T'].value_counts())
# print(registered['final_T-cell receptor'].value_counts())
# print(registered['final_Monocyte'].value_counts())
# print(registered['final_Macrophage'].value_counts())
# print(registered['final_B cell'].value_counts())
# print(registered['final_Leukocyte'].value_counts())
# print(registered['final_Progenitor'].value_counts())

#Print the total number of cells classified
print("Total number of cells classified registered:")
print(registered.shape[0])


# print("Number of cells in unregistered:")
# print(unregistered['final_Enteroendocrine'].value_counts())
# print(unregistered['final_Enterocytes'].value_counts())
# print(unregistered['final_Fibroblasts'].value_counts())
# print(unregistered['final_Stromal'].value_counts())
# print(unregistered['final_Myeloid'].value_counts())
# print(unregistered['final_Helper T'].value_counts())
# print(unregistered['final_Cytotoxic T'].value_counts())
# print(unregistered['final_T-cell receptor'].value_counts())
# print(unregistered['final_Monocyte'].value_counts())
# print(unregistered['final_Macrophage'].value_counts())
# print(unregistered['final_B cell'].value_counts())
# print(unregistered['final_Leukocyte'].value_counts())
# print(unregistered['final_Progenitor'].value_counts())

#Print the total number of cells classified
print("Total number of cells classified unregistered:")
print(unregistered.shape[0])

#Save the classified cells to a csv file 
registered.to_csv('GCA020TIB_registered_classified.csv', index=False)

#Save the unclassified cells to a csv file
unregistered.to_csv('GCA020TIB_unregistered_classified.csv', index=False)