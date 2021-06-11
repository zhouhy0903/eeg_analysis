### 
path.conf, aimtrack_path.conf, eeg_path.conf, score_path.conf为配置文件
### aimtrack.py
#### drop_repeat
去除原轨迹文件中重复的部分
#### show_aimtrack
展示无情绪射击过程中的射击轨迹
#### save_aimtrack_feature
保存射击轨迹中提取的特征
#### get_aimtrack_feature
读取各瞄准过程中的特征

### score.py
#### get_score 
获得每个人的成绩，init配置成绩文件的路径，配置文件在score_path.conf中，需要修改path_score的路径。

### eegprocess.py
将去除眼电后得到的.fif文件放入一个目录中，提取需要的epoch。这里需要修改提取epoch的标签，提取的时间段长度，将原始的.fif文件转化为.h5或.csv文件，这些数据主要包括在标签前一段时间内的各导联的数据。
#### preprocess
从原始数据中提取epoch，转化为.h5或.csv文件。
#### get_epoch_eeg
从.h5文件中提取第i个人的数据。