import pandas as pd
import numpy as np
import argparse
import os

# This is for GammaGateR classification:

# Get the csv files as arguments using argparse

parser = argparse.ArgumentParser(description='Classify cells using GammaGateR')
parser.add_argument('registered_csv', type=str, help='Path to the registered file')
parser.add_argument('unregistered_csv', type=str, help='Path to the unregistered file')
parser.add_argument('name', type=str, help='Name of tissue')
args = parser.parse_args()

# Read the csv files
registered = pd.read_csv(args.registered_csv)
unregistered = pd.read_csv(args.unregistered_csv)
name=args.name

# For columns starting from 3 and ending in last column -1 set True if the value is greater than 0.5, False otherwise
for i in range(0, len(registered.columns)-1):
    registered[registered.columns[i]] = registered[registered.columns[i]] > 0.5
    unregistered[unregistered.columns[i]] = unregistered[unregistered.columns[i]] > 0.5

# For consistency
registered.columns = registered.columns.str.upper()
unregistered.columns = unregistered.columns.str.upper()

registered.columns = registered.columns.str.replace('NZNORM_MEAN', 'Mean')
unregistered.columns = unregistered.columns.str.replace('NZNORM_MEAN', 'Mean')

# Add a column to the left called Instances and start the second row from 1 to the end
registered.insert(0, 'Instances', range(1, 1 + len(registered)))
unregistered.insert(0, 'Instances', range(1, 1 + len(unregistered)))

# Initialize the "Undefined" column
registered['Undefined'] = False
unregistered['Undefined'] = False

# Group Epi+
# Make a new column called Epi+ and set it to True if NAKATPASE or PANCK or CGA
registered['Epi+'] = registered['Mean_NAKATPASE'] | registered['Mean_PANCK'] | registered['Mean_CGA']
unregistered['Epi+'] = unregistered['Mean_NAKATPASE'] | unregistered['Mean_PANCK'] | unregistered['Mean_CGA']

# Group Stroma+
# Make a new column called Stroma+ and set it True if Vimentin or SMA True
registered['Stroma+'] = registered['Mean_VIMENTIN'] | registered['Mean_SMA']
unregistered['Stroma+'] = unregistered['Mean_VIMENTIN'] | unregistered['Mean_SMA']

# Exclude nuclei that are both Epi+ and Stroma+
# Set Undefined to True where Epi+ and Stroma+ are both True
registered.loc[registered['Epi+'] & registered['Stroma+'], 'Undefined'] = True
unregistered.loc[unregistered['Epi+'] & unregistered['Stroma+'], 'Undefined'] = True

# Group Immune+ 
# Make a new column called Immune+ and set it to True if CD3 or CD8 or CD20 or CD68 or CD45 or CD4 or CD11B or LYSOZYME or CD3D True
registered['Immune+'] =  registered['Mean_CD8'] | registered['Mean_CD20'] | registered['Mean_CD68'] | registered['Mean_CD45'] | registered['Mean_CD4'] | registered['Mean_CD11B'] | registered['Mean_LYSOZYME'] | registered['Mean_CD3D']
unregistered['Immune+'] = unregistered['Mean_CD8'] | unregistered['Mean_CD20'] | unregistered['Mean_CD68'] | unregistered['Mean_CD45'] | unregistered['Mean_CD4'] | unregistered['Mean_CD11B'] | unregistered['Mean_LYSOZYME'] | unregistered['Mean_CD3D']

# Remove Immune conflicts for Macrophages
conflict_conditions = (registered['Mean_CD68'] & registered['Mean_CD3D']) | (registered['Mean_CD68'] & registered['Mean_CD20']) | (registered['Mean_CD68'] & registered['Mean_CD4']) | (registered['Mean_CD68'] & registered['Mean_CD8']) | (registered['Mean_CD68'] & registered['Mean_CD11B'])
registered.loc[conflict_conditions, 'Undefined'] = True

conflict_conditions = (unregistered['Mean_CD68'] & unregistered['Mean_CD3D']) | (unregistered['Mean_CD68'] & unregistered['Mean_CD20']) | (unregistered['Mean_CD68'] & unregistered['Mean_CD4']) | (unregistered['Mean_CD68'] & unregistered['Mean_CD8']) | (unregistered['Mean_CD68'] & unregistered['Mean_CD11B'])
unregistered.loc[conflict_conditions, 'Undefined'] = True

# Remove Immune conflicts for Monocytes
conflict_conditions = (registered['Mean_CD11B'] & registered['Mean_CD3D']) | (registered['Mean_CD11B'] & registered['Mean_CD20']) | (registered['Mean_CD11B'] & registered['Mean_CD4']) | (registered['Mean_CD11B'] & registered['Mean_CD8']) | (registered['Mean_CD11B'] & registered['Mean_CD68'])
registered.loc[conflict_conditions, 'Undefined'] = True

conflict_conditions = (unregistered['Mean_CD11B'] & unregistered['Mean_CD3D']) | (unregistered['Mean_CD11B'] & unregistered['Mean_CD20']) | (unregistered['Mean_CD11B'] & unregistered['Mean_CD4']) | (unregistered['Mean_CD11B'] & unregistered['Mean_CD8']) | (unregistered['Mean_CD11B'] & unregistered['Mean_CD68'])
unregistered.loc[conflict_conditions, 'Undefined'] = True

# Remove Immune conflicts for B cells
conflict_conditions = (registered['Mean_CD20'] & registered['Mean_CD3D']) | (registered['Mean_CD20'] & registered['Mean_CD4']) | (registered['Mean_CD20'] & registered['Mean_CD8'])
registered.loc[conflict_conditions, 'Undefined'] = True

conflict_conditions = (unregistered['Mean_CD20'] & unregistered['Mean_CD3D']) | (unregistered['Mean_CD20'] & unregistered['Mean_CD4']) | (unregistered['Mean_CD20'] & unregistered['Mean_CD8'])
unregistered.loc[conflict_conditions, 'Undefined'] = True

# Remove Immune conflicts for helper T cells and cytotoxic T cells
conflict_conditions = (registered['Mean_CD3D'] & registered['Mean_CD45'] & registered['Mean_CD4']) | (registered['Mean_CD3D'] & registered['Mean_CD45'] & registered['Mean_CD8']) | (registered['Mean_CD4'] & registered['Mean_CD8'])
registered.loc[conflict_conditions, 'Undefined'] = True

conflict_conditions = (unregistered['Mean_CD3D'] & unregistered['Mean_CD45'] & unregistered['Mean_CD4']) | (unregistered['Mean_CD3D'] & unregistered['Mean_CD45'] & unregistered['Mean_CD8']) | (unregistered['Mean_CD4'] & unregistered['Mean_CD8'])
unregistered.loc[conflict_conditions, 'Undefined'] = True

# Group Progenitor Cells
# Make a new column called Progenitor and set it True if SOX9+ or OLFM4+
registered['Progenitor'] = registered['Mean_SOX9'] | registered['Mean_OLFM4']
unregistered['Progenitor'] = unregistered['Mean_SOX9'] | unregistered['Mean_OLFM4']

# Exclude instances that are not in either Epi or Stroma
registered.loc[~(registered['Epi+'] | registered['Stroma+']), 'Undefined'] = True
unregistered.loc[~(unregistered['Epi+'] | unregistered['Stroma+']), 'Undefined'] = True

# Remove conflicts for enteroendocrine across all instances
conflict_conditions = (registered['Mean_CGA'] & registered['Immune+']) | (registered['Mean_CGA'] & registered['Stroma+']) | (registered['Mean_CGA'] & registered['Progenitor'])
registered.loc[conflict_conditions, 'Undefined'] = True

conflict_conditions = (unregistered['Mean_CGA'] & unregistered['Immune+']) | (unregistered['Mean_CGA'] & unregistered['Stroma+']) | (unregistered['Mean_CGA'] & unregistered['Progenitor'])
unregistered.loc[conflict_conditions, 'Undefined'] = True

# Remove conflicts for fibroblasts across all instances
conflict_conditions = (registered['Mean_SMA'] & registered['Immune+'])
registered.loc[conflict_conditions, 'Undefined'] = True

conflict_conditions = (unregistered['Mean_SMA'] & unregistered['Immune+'])
unregistered.loc[conflict_conditions, 'Undefined'] = True

# Remove conflicts for progenitors across all instances
conflict_conditions = (registered['Progenitor'] & registered['Immune+'])
registered.loc[conflict_conditions, 'Undefined'] = True

conflict_conditions = (unregistered['Progenitor'] & unregistered['Immune+'])
unregistered.loc[conflict_conditions, 'Undefined'] = True

# Remove instances that are negative for all large groupings
registered.loc[~(registered['Epi+'] | registered['Stroma+'] | registered['Immune+'] | registered['Progenitor']), 'Undefined'] = True
unregistered.loc[~(unregistered['Epi+'] | unregistered['Stroma+'] | unregistered['Immune+'] | unregistered['Progenitor']), 'Undefined'] = True

# Remove any immune instances from Epi+ groups
registered.loc[registered['Epi+'] & registered['Immune+'], 'Undefined'] = True
unregistered.loc[unregistered['Epi+'] & unregistered['Immune+'], 'Undefined'] = True

# Final annotation for enteroendocrine
registered['final_Enteroendocrine'] = registered['Mean_CGA'] & ~registered['Progenitor'] & registered['Epi+']
unregistered['final_Enteroendocrine'] = unregistered['Mean_CGA'] & ~unregistered['Progenitor'] & unregistered['Epi+']

# Final annotation for enterocytes
registered['final_Enterocytes'] = ~registered['Mean_CGA'] & ~registered['Progenitor'] & registered['Epi+']
unregistered['final_Enterocytes'] = ~unregistered['Mean_CGA'] & ~unregistered['Progenitor'] & unregistered['Epi+']

# Group fibroblasts/stromal(undetermined)
registered['Fibroblasts/Stromal(undetermined)'] = registered['Stroma+'] & ~registered['Immune+']
unregistered['Fibroblasts/Stromal(undetermined)'] = unregistered['Stroma+'] & ~unregistered['Immune+']

# Final annotation for fibroblasts
registered['final_Fibroblasts'] = registered['Fibroblasts/Stromal(undetermined)'] & registered['Mean_SMA'] & ~registered['Progenitor']
unregistered['final_Fibroblasts'] = unregistered['Fibroblasts/Stromal(undetermined)'] & unregistered['Mean_SMA'] & ~unregistered['Progenitor']

# Final annotation for stromal(undetermined)
registered['final_Stromal'] = registered['Fibroblasts/Stromal(undetermined)'] & ~registered['Mean_SMA'] & ~registered['Progenitor']
unregistered['final_Stromal'] = unregistered['Fibroblasts/Stromal(undetermined)'] & ~unregistered['Mean_SMA'] & ~unregistered['Progenitor']

# Final annotation for myleoid
registered['final_Myeloid'] = registered['Immune+'] & (registered['Mean_LYSOZYME'] & ~registered['Mean_CD68'] & ~registered['Mean_CD11B'] & ~registered['Mean_CD20']) & ~registered['Mean_CD3D'] & ~registered['Mean_CD8'] & ~registered['Mean_CD4']
unregistered['final_Myeloid'] = unregistered['Immune+'] & (unregistered['Mean_LYSOZYME'] & ~unregistered['Mean_CD68'] & ~unregistered['Mean_CD11B'] & ~unregistered['Mean_CD20']) & ~unregistered['Mean_CD3D'] & ~unregistered['Mean_CD8'] & ~unregistered['Mean_CD4']

# Final annotation for helper T
registered['final_Helper T'] = registered['Immune+'] & (registered['Mean_CD4'] & ~registered['Progenitor'])
unregistered['final_Helper T'] = unregistered['Immune+'] & (unregistered['Mean_CD4'] & ~unregistered['Progenitor'])

# Final annotation for cytotoxic T
registered['final_Cytotoxic T'] = registered['Immune+'] & (registered['Mean_CD8'] & ~registered['Progenitor'])
unregistered['final_Cytotoxic T'] = unregistered['Immune+'] & (unregistered['Mean_CD8'] & ~unregistered['Progenitor'])

# Final annotation for T-cell receptor
registered['final_T-cell receptor'] = registered['Immune+'] & (registered['Mean_CD3D'] & ~registered['Mean_CD4'] & ~registered['Mean_CD8'])
unregistered['final_T-cell receptor'] = unregistered['Immune+'] & (unregistered['Mean_CD3D'] & ~unregistered['Mean_CD4'] & ~unregistered['Mean_CD8'])

# Final annotation for monocyte
registered['final_Monocyte'] = registered['Immune+'] & (registered['Mean_CD11B'] & ~registered['Mean_CD3D'] & ~registered['Progenitor'] & ~registered['Mean_CD4'] & ~registered['Mean_CD8'])
unregistered['final_Monocyte'] = unregistered['Immune+'] & (unregistered['Mean_CD11B'] & ~unregistered['Mean_CD3D'] & ~unregistered['Progenitor'] & ~unregistered['Mean_CD4'] & ~unregistered['Mean_CD8'])

# Final annotation for macrophage
registered['final_Macrophage'] = registered['Immune+'] & (registered['Mean_CD68'] & ~registered['Mean_CD3D'] & ~registered['Progenitor'] & ~registered['Mean_CD4'] & ~registered['Mean_CD8'])
unregistered['final_Macrophage'] = unregistered['Immune+'] & (unregistered['Mean_CD68'] & ~unregistered['Mean_CD3D'] & ~unregistered['Progenitor'] & ~unregistered['Mean_CD4'] & ~unregistered['Mean_CD8'])

# Final annotation for B cell
registered['final_B cell'] = registered['Immune+'] & (registered['Mean_CD20'] & ~registered['Mean_CD68'] & ~registered['Mean_CD3D'] & ~registered['Progenitor'] & ~registered['Mean_CD4'] & ~registered['Mean_CD8'])
unregistered['final_B cell'] = unregistered['Immune+'] & (unregistered['Mean_CD20'] & ~unregistered['Mean_CD68'] & ~unregistered['Mean_CD3D'] & ~unregistered['Progenitor'] & ~unregistered['Mean_CD4'] & ~unregistered['Mean_CD8'])

# Final annotation for leukocyte
registered['final_Leukocyte'] = registered['Immune+'] & (registered['Mean_CD45'] & ~registered['Mean_CD20'] & ~registered['Mean_CD68'] & ~registered['Mean_CD3D'] & ~registered['Progenitor'] & ~registered['Mean_CD4'] & ~registered['Mean_CD8'] & ~registered['Mean_CD11B'] & ~registered['Mean_LYSOZYME'])
unregistered['final_Leukocyte'] = unregistered['Immune+'] & (unregistered['Mean_CD45'] & ~unregistered['Mean_CD20'] & ~unregistered['Mean_CD68'] & ~unregistered['Mean_CD3D'] & ~unregistered['Progenitor'] & ~unregistered['Mean_CD4'] & ~unregistered['Mean_CD8'] & ~unregistered['Mean_CD11B'] & ~unregistered['Mean_LYSOZYME'])

# Final annotation for Progenitor
registered['final_Progenitor'] = registered['Progenitor']
unregistered['final_Progenitor'] = unregistered['Progenitor']

final_columns = [col for col in registered.columns if col.startswith('final_')]

# Set all 'final_' columns to False where 'Undefined' is True for registered and unregistered
registered.loc[registered['Undefined'], final_columns] = False
unregistered.loc[unregistered['Undefined'], final_columns] = False

# Save the classified cells to a csv file 
registered.to_csv('GCA020TIB_registered_classified.csv', index=False)

# Save the unclassified cells to a csv file
unregistered.to_csv('GCA020TIB_unregistered_classified.csv', index=False)
# class_labels = [col for col in registered.columns if col.startswith("final_")]

# # Initialize dictionaries for counts
# gained_cells = {label: 0 for label in class_labels}
# lost_cells = {label: 0 for label in class_labels}
# changed_cells = {label: 0 for label in class_labels}
# total_before = {label: 0 for label in class_labels}
# total_after = {label: 0 for label in class_labels}

# # Iterate through each row to calculate gains, losses, changes, and totals
# for (idx_reg, reg_row), (idx_unreg, unreg_row) in zip(registered.iterrows(), unregistered.iterrows()):
#     reg_classes = [label for label in class_labels if reg_row[label]]
#     unreg_classes = [label for label in class_labels if unreg_row[label]]

#     # Calculate total counts before and after registration
#     for label in class_labels:
#         if unreg_row[label]:
#             total_before[label] += 1
#         if reg_row[label]:
#             total_after[label] += 1

#     # Calculate gains, losses, and changes
#     if unreg_row['Undefined']:
#         for label in reg_classes:
#             gained_cells[label] += 1
#     if reg_row['Undefined']:
#         for label in unreg_classes:
#             lost_cells[label] += 1
#     if len(unreg_classes) == 1 and len(reg_classes) == 1 and unreg_classes[0] != reg_classes[0]:
#         changed_cells[unreg_classes[0]] += 1

# # Prepare summary DataFrame
# summary_df = pd.DataFrame({
#     'Tissue': [name]
# })

# for label in class_labels:
#     summary_df[f'gain_{label}'] = [gained_cells[label]]
#     summary_df[f'loss_{label}'] = [lost_cells[label]]
#     summary_df[f'change_{label}'] = [changed_cells[label]]
#     summary_df[f'total_before_{label}'] = [total_before[label]]
#     summary_df[f'total_after_{label}'] = [total_after[label]]

# # Append to CSV if exists, else create new
# output_filename = 'tissue_classification_changes.csv'
# if os.path.exists(output_filename):
#     summary_df.to_csv(output_filename, mode='a', header=False, index=False)
# else:
#     summary_df.to_csv(output_filename, index=False)

# print("Updated Gain/Loss/Change and Total Counts Data for each class:")
# print(summary_df)