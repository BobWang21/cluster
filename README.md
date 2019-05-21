# cluster
k_init.py file is stolen from sklearn😳

   | 特征类别 | 特征 | 描述|
   | :---: | --- | :---: | 
   | gift| gift\_sum | 用户所送礼物总金额|
   | gift| gift\_count | 用户送礼物的 次数/个数 按session汇总|
   | gift| gift\_session\_min| 用户所送礼的最小金额, 按session汇总|
   | gift| gift\_session\_max| 用户所送礼的最大金额, 按session汇总|
   | gift| gift\_session\_mean | 用户每次所送礼物平均金额, 按session汇总|
   | gift| gift\_session\_std | 用户每次送礼物价格的方差, 按session汇总|
   | gift| gift\_day\_max | 用户所送礼物的最大值, 按天汇总|
   | gift| gift\_day\_min | 用户所送礼物的最小值,按天汇总|
   | time| recent | 用户最近一次购买距今时间|
   | time| frequency | 用户送礼物的天数|
   | time| max\_continue\_days | 用户最长连续送礼天数 例如：2017-05-01, 	2017-05-02, 2017-05-05, 2017-05-06, 2017-05-07 返回3天|
   | type| live\_tf\_idf|用户live消费 tf-idf 值|
   | type| media\_tf\_idf |用户media消费 tf-idf 值|
   | type| live_sum |用户live消费总金额|
   | type| live_per |用户live消费占其消费的百分比|
   | type| media_sum |用户media消费总金额|
   | type| media_per |用户media消费占其消费的百分比|
   | receive| rec\_num |用户送给多少不同的receiver|
   | receive | max\_rec\_per |最大receiver金额占比|
   | receive | rec\_0 |用户送给0类接受者占比|
   | receive | rec\_1 |用户送给1类接受者占比|
   | receive | rec\_2 |用户送给2类接受者占比|
   
