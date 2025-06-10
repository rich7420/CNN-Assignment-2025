# CNN-Assignment-2025
NTCU113-2  ｜   Machine Learning   ｜   賴冠州教授
## Submission Requirements
- Submit your work as a Jupyter Notebook (.ipynb).
- File name format: `ClassNumber_CNN_Assignment.ipynb` (e.g., `ACS109145_CNN_Assignment.ipynb`).
- Ensure the notebook includes all code, visualizations, and a report section answering Task 5.
- Upload to this repository via a pull request (PR).
## Steps
1. Fork this repo.
2. Do Submission Requirements base on cnn_assignment.ipynb in colab.
3. attach your file(.ipynb) to your github repo
4. Commit and Create PR.
## **5 Key Tasks to Complete:**
1. **Task 1: Model Architecture Enhancement**
   - Modify the CNN model structure
   - Must include `model = models.Sequential` and `Conv2D` layers
2. **Task 2: Hyperparameter Optimization** 
   - Implement `model.compile` and `model.fit`
   - Specify optimizer (SGD/RMSprop/Adam)
3. **Task 3: Data Augmentation**
   - Add `ImageDataGenerator` with augmentation parameters
   - Include: rotation_range, width_shift_range, height_shift_range, horizontal_flip
4. **Task 4: Visualization**
   - Create plots using `plt.plot`, `plt.subplot`, or `plt.imshow`
   - Generate `predictions` variable for model predictions
5. **Task 5: Report Section**
   - Add Markdown cell with heading containing "Task 5:", "Report", or "Conclusion"
   - Include meaningful analysis (more than 3 non-empty lines)

## Autograding
- File name format validation
- Code execution without errors
- Presence of required code components
- Visualization outputs
- Report section completeness