import json
import os
import glob

def update_notebook(filepath, task_prefix):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if "save_plot(fig, '" in line:
                    # e.g., save_plot(fig, 'rating_distribution.png')
                    parts = line.split("save_plot(fig, '")
                    if "/" not in parts[1]:  # not already prefixed
                        line = f"{parts[0]}save_plot(fig, '{task_prefix}/{parts[1]}"
                new_source.append(line)
            cell['source'] = new_source

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

base_dir = "datamining part 1/notebooks"
update_notebook(os.path.join(base_dir, "01_data_understanding.ipynb"), "task_1")
update_notebook(os.path.join(base_dir, "02_clustering.ipynb"), "task_2")
update_notebook(os.path.join(base_dir, "03_classification_regression.ipynb"), "task_3")
update_notebook(os.path.join(base_dir, "04_pattern_mining.ipynb"), "task_4")
print("Notebooks updated.")
