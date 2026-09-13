"""视图层：每个 Tab 一个模块，通过 render(ctx) 挂载。

Tab 之间不互相 import，只依赖 services/ 和 context，
新增一个 Tab 不需要改动其他任何文件。
"""
