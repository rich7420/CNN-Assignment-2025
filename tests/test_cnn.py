import nbformat
import pytest
import glob
import re

def load_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def test_file_name():
    files = [f for f in glob.glob("*_CNN_Assignment.ipynb") if not f.startswith('executed_')]
    assert len(files) == 1, "Exactly one notebook with format 'ClassNumber_CNN_Assignment.ipynb' required (excluding executed files)"
    assert re.match(r'[A-Z]{3}\d{6}_CNN_Assignment\.ipynb', files[0]), \
        "Notebook name must follow 'ClassNumber_CNN_Assignment.ipynb' (e.g., ACS109145_CNN_Assignment.ipynb)"

def test_task_1_model_changes():
    nb_file = [f for f in glob.glob("*_CNN_Assignment.ipynb") if not f.startswith('executed_')][0]
    nb = load_notebook(nb_file)
    model_code = ""
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'model = models.Sequential' in cell.source:
            model_code = cell.source
            break
    assert model_code, "Model definition not found"
    assert 'Conv2D' in model_code, "Task 1: Model must include at least one Conv2D layer"

def test_task_2_hyperparameters():
    nb_file = [f for f in glob.glob("*_CNN_Assignment.ipynb") if not f.startswith('executed_')][0]
    nb = load_notebook(nb_file)
    compile_code = ""
    fit_code = ""
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if 'model.compile' in cell.source:
                compile_code = cell.source
            if 'model.fit' in cell.source:
                fit_code = cell.source
    assert compile_code, "Model compilation not found"
    assert fit_code, "Model training not found"
    assert any(opt in compile_code for opt in ['SGD', 'RMSprop', 'Adam']), \
        "Task 2: Must specify an optimizer (e.g., SGD, RMSprop, Adam)"

def test_task_3_data_augmentation():
    nb_file = [f for f in glob.glob("*_CNN_Assignment.ipynb") if not f.startswith('executed_')][0]
    nb = load_notebook(nb_file)
    augmentation_code = ""
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'ImageDataGenerator' in cell.source:
            augmentation_code = cell.source
            break
    assert augmentation_code, "Task 3: ImageDataGenerator not found"
    assert any(param in augmentation_code for param in ['rotation_range', 'width_shift_range', 'height_shift_range', 'horizontal_flip']), \
        "Task 3: ImageDataGenerator must include at least one augmentation parameter"

def test_task_4_visualization():
    nb_file = [f for f in glob.glob("*_CNN_Assignment.ipynb") if not f.startswith('executed_')][0]
    nb = load_notebook(nb_file)
    vis_code = ""
    for cell in nb.cells:
        if cell.cell_type == 'code' and ('plt.plot' in cell.source or 'plt.imshow' in cell.source) and 'predictions' in cell.source:
            vis_code = cell.source
            break
    assert vis_code, "Task 4: Visualization code not found"
    assert any(range_str in vis_code for range_str in ['range(10)', 'range(1, 11)']), \
        "Task 4: Must visualize at least 10 test images"

def test_task_5_report():
    nb_file = [f for f in glob.glob("*_CNN_Assignment.ipynb") if not f.startswith('executed_')][0]
    nb = load_notebook(nb_file)
    report_found = False
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and any(title in cell.source for title in ['# Task 5:', '# Report', '# Conclusion']):
            report_found = True
            report_content = cell.source
            break
    assert report_found, "Task 5: Report section with '# Task 5:', '# Report', or '# Conclusion' not found"
    assert len([line for line in report_content.split('\n') if line.strip()]) > 3, "Task 5: Report section must contain meaningful content (more than 3 non-empty lines)"