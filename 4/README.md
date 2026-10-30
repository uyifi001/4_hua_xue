# deal.II 镜像复现

`dealii_mirror.cpp` 将仓库根目录的 `chemo_mechanical.py` 的核心分裂算法
搬到 C++：后向 Euler 氧扩散与反应汇项、解析 TGO 相场更新，以及生长应变
引起的应力反馈。为保证在没有 Python/FEniCSx 的环境中也能验证，当前文件
使用一维竖向有限体积数组作为 deal.II `DoFHandler`/向量的可替换参考实现；
在 deal.II 镜像中可直接由同一个 `CMakeLists.txt` 编译运行。

```bash
cmake -S 4 -B 4/build
cmake --build 4/build -j2
4/build/chemo_mechanical_dealii on 4/results
```

输出：

- `results/history.csv`：TGO 厚度、平均应力和收敛过程；
- `results/final_fields.csv`：最终的 `n`、氧浓度和应力场；
- `results/result.svg`：无需绘图库即可查看的结果图；
- `results/params.txt`：本次运行参数。

第一个参数改为 `off` 可复现无 TBC 工况。
