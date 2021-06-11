### python + mne环境运行
#### 安装相关依赖包
pip3 install -r requirements.txt
#### 运行代码
python ica.py
#### 参数设置
data_path为原始脑电信号数据
save_path为ICA以及去除眼电后的数据保存位置
low_pass和high_pass为滤波的低通和高通频率
componum为ICA的成分个数，一般设置为大于通道数
#### 运行细节
第一次运行可以先拷贝一个人的数据到data_path下，然后观察ICA的结果，选取其中为眼电的成分，赋值给pattern_subject_id（实验者编号，一般为0）和pattern_ica_id（眼电成分的编号）。
然后再将所有人的数据拷贝到data_path下，进行ICA去除眼电。
