import pandas as pd

# Load the datasets
registered_df = pd.read_csv('GCA020TIB_registered_classified.csv')
unregistered_df = pd.read_csv('GCA020TIB_unregistered_classified.csv')
gain_loss_change_df = pd.read_csv('gain_loss_change.csv')

# Identify class labels that start with 'final_'
class_labels = [col for col in registered_df.columns if col.startswith("final_")]

# Initialize dictionaries to hold counts of gained, lost, and changed cells for each class
gained_cells = {label: 0 for label in class_labels}
lost_cells = {label: 0 for label in class_labels}
changed_cells = {label: 0 for label in class_labels}

# Dictionaries to hold total counts of cells before and after registration for each class
total_before = {label: 0 for label in class_labels}
total_after = {label: 0 for label in class_labels}

# Iterate through each row to calculate gains, losses, changes, and totals
for index, (reg_row, unreg_row) in enumerate(zip(registered_df.iterrows(), unregistered_df.iterrows())):
    reg_classes = [label for label in class_labels if reg_row[1][label]]
    unreg_classes = [label for label in class_labels if unreg_row[1][label]]

    # Calculate total counts before and after registration
    for label in class_labels:
        if unreg_row[1][label]:
            total_before[label] += 1
        if reg_row[1][label]:
            total_after[label] += 1

    # Calculate gains, losses, and changes
    if unreg_row[1]['Undefined']:
        for label in reg_classes:
            gained_cells[label] += 1
    if reg_row[1]['Undefined']:
        for label in unreg_classes:
            lost_cells[label] += 1
    if len(unreg_classes) == 1 and len(reg_classes) == 1 and unreg_classes[0] != reg_classes[0]:
        changed_cells[unreg_classes[0]] += 1

# Update the DataFrame with new counts
for label in class_labels:
    gain_loss_change_df.loc['Gained', label] += gained_cells[label]
    gain_loss_change_df.loc['Lost', label] += lost_cells[label]
    gain_loss_change_df.loc['Changed', label] += changed_cells[label]
    gain_loss_change_df.loc['Total_before', label] += total_before[label]
    gain_loss_change_df.loc['Total_after', label] += total_after[label]

# Save the updated DataFrame to a new CSV file
gain_loss_change_df.to_csv('updated_gain_loss_change.csv', index=False)

# Print the updated DataFrame
print("Updated Gain/Loss/Change and Total Counts Data:")
print(gain_loss_change_df)
