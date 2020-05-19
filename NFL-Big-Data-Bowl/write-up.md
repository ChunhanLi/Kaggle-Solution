### What I learned from this competition.

- `augment the data by copying all rows twice, once with Yards-1 and once with Yards+1` 非常有意思的点 没测试过实际效果 说对树模型有很大提升 学到了 对NN没效果下降 猜想原因 1. 数据量比较小 只有2W+train 2.CRPS本身对这个操作应该不敏感 要是MSE/MAE 这个操作不一定好用 后续：查了下这是一种soft label的思想 可以去描述出不同类别之间的联系 比如给五类编码 [0,0.2,0.6,0.2,0] 如果真实值是3 
- 对结果后处理 不仅可以对太大的处理/也可以对太小的处理 最差不会超过YardLine的负值 我把后者忘记了
- DATA cleaning 去处理2017年的数据  2017年A与yards的关系 和2018年不一样 `https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119314#latest-683036` `https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119326#latest-683047`