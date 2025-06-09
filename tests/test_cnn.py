import nbformat
import pytest
import re
import json

def load_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def test_file_name():
    import glob
    files = glob.glob("*_CNN_Assignment.ipynb")
    assert len(files) == 1, "Exactly one notebook with format 'ClassNumber_CNN_Assignment.ipynb' required"
    assert re.match(r'[A-Z]{3}\d{6}_CNN_Assignment\.ipynb', files[0]), \
        "Notebook name must follow 'ClassNumber_CNN_Assignment.ipynb' (e.g., ACS109145_CNN_Assignment.ipynb)"

def test_task_1_model_changes():
    nb = load_notebook(glob.glob("*_CNN_Assignment.ipynb")[0])
    model_code = ""
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'model = models.Sequential' in cell.source:
            model_code = cell.source
            break
    assert model_code, "Model definition not found"
    # Check for additional Conv2D or Dropout layers
    assert 'Conv2D' in model_code and ('Dropout' in model_code or model_code.count('Conv2D') > 3), \
        "Task 1: Model must include additional Conv2D or Dropout layers"

def test_task_2_hyperparameters():
    nb = load_notebook(glob.glob("*_CNN_Assignment.ipynb")[0])
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
    # Check for optimizer change or epochs > 10
    assert any(opt in compile_code for opt in ['SGD', 'RMSprop']) or 'epochs=2' in fit_code or 'epochs=1' in fit_code, \
        "Task 2: Must change optimizer to SGD/RMSprop or increase epochs beyond 10"

def test_task_3_data_augmentation():
    nb = load_notebook(glob.glob("*_CNN_Assignment.ipynb")[0])
    augmentation_code = ""
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'ImageDataGenerator' in cell.source:
            augmentation_code = cell.source
            break
    assert augmentation_code, "Task 3: ImageDataGenerator not found"
    assert any(param in augmentation_code for param in ['rotation_range', 'width_shift_range', 'horizontal_flip']), \
        "Task 3: ImageDataGenerator must include rotation, shift, or flip"

def test_task_4_visualization():
    nb = load_notebook(glob.glob("*_CNN_Assignment.ipynb")[0])
    vis_code = ""
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'plt.subplot' in cell.source and 'predictions' in cell.source:
            vis_code = cell.source
            break
    assert vis_code, "Task 4: Visualization code not found"
    assert 'range(10)' in vis_code or 'range(1, 11)' in vis_code, \
        "Task 4: Must visualize at least 10 test images"

def test_task_5_report():
    nb = load_notebook(glob.glob("*_CNN_Assignment.ipynb")[0])
    report_found = False
    report_content = ""
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and '# Task 5:' in cell.source:
            report_found = True
            report_content = cell.source
            break
    assert report_found, "Task 5: Report section with '# Task 5:' not found"
    assert len(report_content.split('\n')) > 3, "Task 5: Report section must contain meaningful content (more than 3 lines)"