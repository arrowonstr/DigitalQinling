## 秦岭水质反演模型库

### 环境
```
python=3.9
```

```
pip install -r requirements.txt
```
### git开发
#### 下载git

#### git关联github账户
```
git config --global user.name "注册名"
git config --global user.email "注册邮箱"
ssh-keygen -t rsa -C "自己的邮箱"

进入c/user/用户名/.ssh/id_rsa.pub 复制文本
到github中 setting ssh粘贴ssh
```

#### 初始化本地分支
```
//本地仓库初始化
git init
git clone git@github.com:arrowonstr/DigitalQinling.git

// 将需要改动的文件添加到缓存区
git add file_name
// 根据缓存区内容生成一次提交 并注释
git commit -m "xxx"
// 新建并切换到master分支（-b参数）
git checkout -b master
// 关联仓库
git remote add origin git@github.com:arrowonstr/DigitalQinling.git
// 将本地内容复制到远端，关联并新建仓库分支（-u参数）
git push -u origin master

// 其它操作
// 添加所有文件进入缓存区
git add .
// 查看缓存区状态
git status
// 移除缓存区文件
git rm --cached file_name

// git push会将所有commit版本都push
```

#### 开发
```
// 检查本地版本和远程版本是否一致
git pull origin master
```

对于冲突
<<<<<<< HEAD
aaa
=======
b45678910
>>>>>>> 6853e5ff961e684d3a6c02d4d06183b5ff330dcc
