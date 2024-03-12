# XJTLU_ICS_Y2S2_Course-notes_23-24
Ciallo～(∠・ω< )⌒★!

欢迎来到本仓库~本仓库的内容是23-24学年西交利物浦大学信息与计算科学（ICS）专业大二第二学期课程笔记的汇总。

Welcome to our repository! The content of this repository is the summary of course notes for the second semester of Information and Computing Science (ICS) Sophomores at XJTLU in the 23-24 academic year.

## 建立本仓库的目的
为促进学习小组更方便地分享管理所学课程笔记、心得，本仓库被建立为公用的，可被大家提交的合作学习平台，还请大家多多参与捏。

## 涵盖课程

1. **CPT102 - Data Structures**
2. **CPT104 - Operating Systems Concepts**
3. **EAP111 - English Language and Study Skills for Advanced Technology**
4. **INT102 - Algorithmic Foundations and Problem Solving**
5. **INT104 - Artificial Intelligence**

## 目标

我们的目标是建立一个积极、协作的学习社群。每个人认领对应的课程/对应的学习部分，在要求内，发挥主观能动性，有责任感地完成对课程的梳理和对小组其他成员的讲述，使得个人和小组成员相互学习督促，良性循环。

具体来说，我们的目标包括：

- **课程笔记**：每位成员通过个人复习和笔记整理，达到对课程内容的深度理解。
- **重点纪要**：在课程内容之中，还有有许多需要认真辨析或展开的概念，定义与实践操作。它们往往难以在课堂中良好地展开，我们应当将其记录。
- **习题试卷**：除了来自图书馆的往年试题，成员将收集与当前学习内容相适应或拓展的练习内容帮助组员巩固所学知识，并将知识与实践结合。
- **互助答疑**：通过讨论、解答疑问和相互支持，确保每个成员都能够取得最好的学习效果。

## 文件目录（目前）

```markdown
|------ 课程编号
    |------ 课程笔记
    |------ 重点纪要
    |------ 习题试卷
自述文件
重要节点时间表
```

## 如何贡献

欢迎每位成员积极参与社群活动！如果您尚未了解什么是github，如何使用github，以下有推荐的教程：

https://edu.csdn.net/skill/git/git-62c30f9c31f64a1d96af732c47c93f04?category=1413

如果您想要贡献或提出建议，请按照以下步骤操作：

1. **创作笔记**：按照自己的进度整理每门课程的笔记。
2. **Pull Request**：将你的笔记提交为一个 Pull Request，以便其他人可以查看和提出建议。
3. **讨论**：在 Issues 中分享你的疑问、建议或者想法，与社群成员一起讨论。

## 推送规范

在开始贡献之前，请确保你的本地主分支与远程主分支同步。你可以使用以下命令：

```bash
git pull origin main
```

创建一个新的分支，分支名应该采用以下格式：

```bash
branch@<名字>
```

例如，如果你的名字是 LiHua，那么你的分支名可以是：

```bash
branch@LiHua
```

 **文件命名规范（待定）**
```bash
对课程笔记等适合以周为粒度分类的文件可以使用周命名
<课程名><周数>_<文件名>
INT104W0_课程信息与时间线

对不适合以周命名的文件、纪要或杂项可以使用日期命名
<课程名>_<创建日期><文件名>
INT104_240310课程信息与时间线

如果文件经过值得注意的改动，可以在文件名后跟版本号（GNU风格），并烦请附上简要的更新说明
<课程名>_<创建日期><文件名><主版本号.子版本号[.修正版本号]>
INT104_240310课程信息与时间线v0.2
或
<课程名><周数>_<文件名><主版本号.子版本号[.修正版本号]>
INT104W0_课程信息与时间线v0.2
```

进入你所在课程的目录，推送你产出的文件，确保你的推送消息（commit message）清晰地描述了你的更改，具体参考提交规范（自己百度）。

```bash
git add .
git commit -m "feat: 你的推送描述……"
git push origin branch@LiHua
```

**描述前缀**
feature:    本次提交的代码用于开发新功能，新增业务
fix: 	    本次提交的代码用于修复bug问题、漏洞
docs: 	    本次提交的代码用于修改文档，注释等相关
style: 	    本次提交的代码用于修改代码格式，不影响代码逻辑，常见的代码规范：PEP8，PEP484
refactor: 	本次提交的代码用于项目/代码重构，理论上不影响现有功能(针对代码的重写，改造原来的模块/类/函数/方法)
perf: 	    本次提交的代码用于提升性能，代码优化
test: 	    本次提交的代码用于增加或修改测试用例代码
deps: 	    本次提交的代码用于升级项目依赖（更新依赖模块）

### 向分支提交代码的操作

基本命令：

1.查看所有分支

```bash
git branch -a （看看是否连接上远程的git）
```

2.创建分支

```bash
git branch xxx（为你的分支起名字）
```

3.切换分支

```bash
git checkout xxx（切换到你创建的分支，xxx为你要切换分支的名字）
```

4.添加修改代码到缓存（注意最后的"."前面有个空格）.是提交当前目录所有修改的

```bash
git add .
```

5.添加提交代码的备注

```bash
git commit -m "xxx"（xxx为本次提交代码的备注）
```

6.提交代码到指定的分支

```bash
git push origin xxx （xxx为要提交代码的分支名称）
```
7.如果git push这个步骤出现了错误，是因为是git服务器中的你提交的文件不在本地代码目录中，可以通过如下命令进行代码合并，然后在使用第6步

```bash
git pull --rebase origin xxx（xxx为要提交代码的分支名称）
```

更多详细内容：https://www.cnblogs.com/hanfe1/p/14807506.html

创建一个 Pull Request，标题和描述应该清晰、简明扼要。确保你的分支与主分支同步，没有冲突，并经过其他团队成员的审核。

如果你的 Pull Request 通过审查，并且没有冲突，由项目维护者进行合并。

重复以上步骤，确保每位成员的笔记都能够被整合到一个综合的学习资料中。


**特别鸣谢：@General_K1ng**

--------End of File--------
