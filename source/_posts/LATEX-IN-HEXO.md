---
title: 渲染 Hexo 博客里的 latex 公式
date: 2021-09-13 10:35:58
tags: ['Others']
categories: ['Blog','Hexo']
---



## markdown渲染器的不足

hexo的默认渲染器hexo-renderer-marked不支持latex语法，一些latex中的符号会被误认为是markdown语法，比如`\`、`_`和`*`，这让mathjax不能正确地渲染（katex需要渲染器支持，根本不能工作）。


## 替换pandoc

办法就是替换Hexo的渲染器，比如在博客目录下执行：


```shell
brew install pandoc
npm uninstall hexo-renderer-marked --save
npm install hexo-renderer-pandoc --save
```

