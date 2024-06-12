import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
annotation_results = pd.read_csv('annotation_results.csv')

# Extract the relevant columns
deformable_cells = annotation_results['Registered cells annotated']
rigid_cells = annotation_results['Unregistered cells annotated']

# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'Deformable(Proposed Method)': deformable_cells,
    'Rigid': rigid_cells
})

# Melt the DataFrame for seaborn
plot_data_melted = plot_data.melt(var_name='Type of Registration Applied', value_name='Number of Cells Annotated')

# Define custom colors
deformable_color = '#0F4C75'
rigid_color = '#F08A5D'
line_color = '#B83B5E'

# Create the plot
plt.figure(figsize=(10, 6))

# Create box plot for Rigid
sns.boxplot(x='Type of Registration Applied', y='Number of Cells Annotated', data=plot_data_melted[plot_data_melted['Type of Registration Applied'] == 'Rigid'], 
            showfliers=False, boxprops=dict(facecolor='none', edgecolor=rigid_color, linewidth=2), 
            medianprops=dict(color=rigid_color, linewidth=2),
            whiskerprops=dict(color=rigid_color, linewidth=2),
            capprops=dict(color=rigid_color, linewidth=2))

# Create box plot for Deformable(Proposed Method)
sns.boxplot(x='Type of Registration Applied', y='Number of Cells Annotated', data=plot_data_melted[plot_data_melted['Type of Registration Applied'] == 'Deformable(Proposed Method)'], 
            showfliers=False, boxprops=dict(facecolor='none', edgecolor=deformable_color, linewidth=2), 
            medianprops=dict(color=deformable_color, linewidth=2),
            whiskerprops=dict(color=deformable_color, linewidth=2),
            capprops=dict(color=deformable_color, linewidth=2))

# Add points with custom colors
sns.swarmplot(x='Type of Registration Applied', y='Number of Cells Annotated', data=plot_data_melted, 
              palette=[deformable_color, rigid_color], alpha=0.6)

# Add correspondence lines with custom color
for i in range(len(deformable_cells)):
    plt.plot([1, 0], [deformable_cells[i], rigid_cells[i]], color=line_color, alpha=0.6, linestyle='--')

# Add bracket and p-value label
x1, x2 = 0, 1
y, h, col = max(deformable_cells.max(), rigid_cells.max()) + 7, 2, 'k'  # Adjust y and h as needed
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1 + x2) * .5, y + h + 0.5, "p<0.05", ha='center', va='bottom', color=col, fontsize=12)  # Adjust y+h+0.5 for higher position

plt.title('Comparison of Number of Annotated Cells between Deformable and Rigid Registrations')
plt.subplots_adjust(top=0.85)  # Add some gap after the title

plt.show()