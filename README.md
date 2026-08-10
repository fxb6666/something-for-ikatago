# ikatago Jupyter Notebook
目前仅针对 Tesla T4 GPU 预设了搜索线程，对于其它GPU请自行测试。

## 用法

### Run In Colab
可任选一种方式打开：
* 打开最新版 ：[Open in Colab](https://colab.research.google.com/github/fxb6666/something-for-ikatago/blob/main/ikatago-for-colab.ipynb)
* 从 [Releases](https://github.com/fxb6666/something-for-ikatago/releases) 下载特定版本的 ipynb 文件，然后打开 [Colab](https://colab.research.google.com) 并上传该文件。

### Run In Cloud Studio
腾讯云 Cloud Studio [产品页面](https://cloud.tencent.com/product/cloudstudio)。

操作步骤：
1. 打开 Cloud Studio [首页](https://cloudstudio.net/)并登录
2. 进入“个人中心”-“应用管理”，点击“创建应用”。
3. 点击顶部的“从 Git 仓库导入”，地址框输入`https://github.com/fxb6666/something-for-ikatago.git`，标题框写个标题，然后点击“创建”。或者选择 Jupyter Notebook 模板再手动上传 ipynb 文件。
4. 开机后打开里面的 ipynb 文件。
5. 点击右上角“选择内核”选择带星号的 Python3 内核，然后在顶部切换GPU。

用完后，在顶部切换GPU的窗口点击“停止”进行关机。
从“应用管理”中再次打开笔记本时无需重新安装，可直接从“启动”部分开始。
