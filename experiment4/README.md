<<<<<<< HEAD
﻿该实验我调用的是deepseek-v4-pro完成的基于大模型的代码审查
具体配置见项目中的env文件
然后定义了上下文类型：
1 仅包含 Diff；
2 Diff + Pull Request 描述；
3 Diff + Commit Message；
4 Diff + Pull Request 描述，Commit Message，修改文件内容

prompt类型：
 Zero-shot Prompt；
 Few-shot Prompt；
Chain-of-Thought Prompt；
 Role-based Prompt。
生成了16个csv文件来保存代码审查的结果 格式为llm_review_result_ctx(上下文类型序号_pt(zero/few/cot/role).csv
=======
# 2023113078-杨夕--AI4SE
 1.实验一使用了5个python脚本 完成了数据挖掘 脚本在src里  实验结果result里
>>>>>>> 120e0d50257d57c2e6b7f79eb9218f20a959bb70
