# Assignment-1--Image-Classification

在你開始訓練以前：

1. 安裝 anaconda
2. 開啟 Anaconda Prompt
3. 下指令安裝套件 conda install opencv/numpy/skimage/tqdm/matplotlib
4. 將 nn-svm-adaboost-classifier.py、train.txt、test.txt、val.txt 以及 images 資料夾放置於同一個路徑下
5. 執行 nn-svm-adaboost-classifier.py 開始訓練

最終結果輸出：
1. svm, ada, nn --> 各分類器訓練完的權重
2. svm_test_result, svm_val_result --> SVM TOP1 & TOP5 準確率
3. ada_test_result, ada_val_result --> AdaBoost TOP1 & TOP5 準確率
4. nn_test_result, nn_val_result --> nn TOP1 & TOP5 準確率

P.S 各模型的準確率會印在 console 裡面，而 curves of the training/test accuracy 會透過 plt 輸出成圖片
