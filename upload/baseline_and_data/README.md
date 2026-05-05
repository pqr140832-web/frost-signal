# 第四届世界科学智能大赛——中学生赛道

## Baseline 安装

### 1. 安装anaconda:

* 可在[anaconda官网](https://www.anaconda.com/download)或[清华tuna镜像站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/?C=M&O=D)选择合适的版本安装。
* 建议勾选 "Add Anaconda3 to my PATH envirionment variable"
 
### 2. 安装vscode

* 可在[vscode官网](https://code.visualstudio.com/download)选择合适的版本安装。

### 3. 配置vscode

* 以管理员权限启动vscode
* 顶部菜单-File-Open Folder-打开baseline所在文件夹
* 左侧菜单点击Extension，安装python插件
* Ctrl+Shiftp+P（同时按）打开命令面板，输入"Python: Select Interpreter"，选择该选项，选择已安装的anaconda中的python解释器（如XXX/anaconda3/python.exe）
* 打开baseline文件，点击右上角运行键，此时应能运行程序，并产生一个paint_pred.csv文件

### 其他说明
- 晋级复赛的选手需提交初赛的预测模型及代码，包括：
  - 训练代码、模型权重、推理代码
  - README文件：简要说明策略及方法，如何启动训练及推理
  - 环境配置：在README中说明使用了哪些python包，或environment.yml文件
- 初赛结束后，以排行榜成绩作为初赛成绩依照，要求团队提交代码审核
- 如存在作弊、提交非模型运行结果等不合理行为，组委会将取消团队比赛资格
- 如果训练模型中使用了数据增强，请在提交程序的README中写明增强方法及数据来源
- 每天限制提交两次预测结果文件