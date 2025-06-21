# CNN 影像分類任務報告

## 任務目標  
本次作業目的是使用 TensorFlow/Keras 建立與訓練一個卷積神經網路（CNN），用於分類 CIFAR-10 影像資料集。
須完成PPT中的五個Tasks 與提高A

---

## Task 1: Model Architecture Enhancement 
### Modify the CNN model structure 
### Must include model = models.Sequential and Conv2D layers

- 使用 `Sequential` 建立 CNN 模型。
- 模型架構重點：
  - 三層 `Conv2D` 卷積層（使用 ReLU 激活函數）
  - 兩層 `MaxPooling2D` 池化層
  - `Flatten` 與 `Dense` 全連接層，最後輸出為 10 類（使用 softmax）
- 設計目的是讓模型能逐層學習圖像的邊緣、形狀與物體特徵。

---

## Task 2: Hyperparameter Optimization 
### Implement model.compile and model.fit 
### Specify optimizer (SGD/RMSprop/Adam)

- 模型使用以下設定進行訓練：
  - 最佳化器：`Adam`
  - 損失函數：`sparse_categorical_crossentropy`
  - 評估指標：`accuracy`
- 訓練過程共執行 10 個 epochs，並加入驗證資料進行評估。
- 透過 `history` 物件儲存訓練與驗證的變化資訊，用於後續分析與繪圖。

---

## Task 3: Data Augmentation
### Add ImageDataGenerator with augmentation parameters
### Include: rotation_range, width_shift_range, height_shift_range, horizontal_flip
- 使用 `ImageDataGenerator` 進行圖像資料的隨機變化，提升模型泛化能力。
- 採用的增強方式包含：
  - `rotation_range=15`：隨機旋轉角度 ±15°
  - `width_shift_range=0.1`：左右平移最多 10%
  - `height_shift_range=0.1`：上下平移最多 10%
  - `horizontal_flip=True`：左右翻轉圖片
- 增強方式可模擬圖像在現實中的變化，讓模型更能適應多樣化輸入。

---

## Task 4: Visualization
### Create plots using plt.plot, plt.subplot, or plt.imshow
### Generate predictions variable for model predictions

- 可視化項目包含：
  - 顯示訓練集中隨機圖像與對應標籤
  - 繪製訓練與驗證的準確率與損失變化曲線
  - 顯示前幾筆測試圖像的預測結果與實際標籤
- 透過圖形觀察模型學習曲線，判斷是否有收斂、過擬合等現象。

---

## Task 5: Report Section
### Add Markdown cell with heading containing "Task 5:", "Report", or "Conclusion"
### Include meaningful analysis (more than 3 non-empty lines)

- 因為有學長姊先寫好的模板了，於是去查裡面可增強訓練結果的參數，了解其意思後再加以改變參數。
- 因前面的東西都不用改，遇事就直接加入Task2的rotation_range, width_shift_range, height_shift_range, horizontal_flip，慢慢調整
- 平移的部分，因為整個主體的位置差異不大，於是不用更動太多；而角度的部分，因為照片角度的確多變，於是嘗試了較多的角度。
- 圖示化的部分，最好還是要有漸趨平緩的上升曲線，避免近乎平整的上升曲線出現。


---

##  模型結果

- Test Accuracy：`0.7844`
- Test Loss：`0.6401`
- Final Training Accuracy：`0.7815`
- Final Validation Accuracy：`0.7844`
- Final Training Loss：`0.6310`
- Final Validation Loss：`0.6401`
- 模型總參數數量：`122570`
- 訓練輪數：10 Epochs
- 使用的資料增強：旋轉、平移、翻轉

---

📄 *本報告結束。*
