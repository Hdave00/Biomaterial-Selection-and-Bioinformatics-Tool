## 预测结果提交

基于fingerprint特征的预测结果和基于RAC特征的预测结果需要分别给出，文件命名为`finger_prediction.csv`和`RAC_prediction.csv`，并放入一个zip压缩包`submission.zip`中提交至天池平台测评。

提交的预测结果文件`finger_prediction.csv`需要遵循下述格式。

```
mof,temperature,time
1,55,48
2,60,120
...
```

提交的预测结果文件`RAC_prediction.csv`需要遵循下述格式。

```
mof,temperature,time,param1,param2,param3,param4,param5,additive
1,55,48,1.1,2.2,3.3,4.4,5.5,0
2,60,120,1.1,2.2,3.3,4.4,5.5,2
...
```

**注意**：

- csv文件数据中不应该包含多余的空格。
- 提交的预测结果顺序应该与提供的test文件一致。
- 预测结果中的`additive`是对训练集中`additive_ategory`的预测结果



-----

## Prediction Result Submission

The prediction results based on fingerprint features and RAC features should be provided separately. The files should be named `finger_prediction.csv` and `RAC_prediction.csv`, respectively. Then, they should be placed in a zip archive named `submission.zip` and submitted to the Tianchi platform for evaluation.

The submitted prediction result file `finger_prediction.csv` should follow the following format:

```
mof,temperature,time
1,55,48
2,60,120
...
```

The submitted prediction result file `RAC_prediction.csv` should follow the following format:

```
mof,temperature,time,param1,param2,param3,param4,param5,additive
1,55,48,1.1,2.2,3.3,4.4,5.5,0
2,60,120,1.1,2.2,3.3,4.4,5.5,2
...
```

**Note:**

- There should be no extra spaces in the CSV file data.
- The order of the submitted prediction results should match the provided test file.
- The `additive` in the prediction results refers to the predicted result for the `additive_category` in the training set.