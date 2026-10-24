# 安装超时与模块缺失的修复说明

## 本次错误的原因

pip 下载 files.pythonhosted.org 上的依赖时发生 ReadTimeoutError，使 python -m pip install -e . 中断。项目环境随后缺少 torch，也没有安装 academic-hybrid-rewriter，因此运行 academic_rewriter.prepare、run、demo 时会出现 ModuleNotFoundError。

pip 的升级提示不是本次失败的原因。异常堆栈中的 D:\conda\envs\INFSCI\Lib 是原虚拟环境使用的基础 Python 标准库路径，单凭该路径不表示标准库损坏。数据下载已成功，无需因这次依赖安装错误重新下载。

## 再次出现 torch 缺失的原因

原 README 仍包含无条件执行的 python -m venv .venv。修复后再次执行该命令，会把 include-system-site-packages 重新设为 false，让项目虚拟环境再次无法读取 tee 中的 torch。此次配置记录中的创建命令正是 P:\MLProject\.venv\Scripts\python.exe -m venv P:\MLProject\.venv。项目包仍然安装着，因此只剩依赖 torch 找不到。

现已恢复共享选项，并将中英文 README 改为调用 scripts/setup.ps1。脚本仅在解释器不存在时创建环境，已有环境会直接复用；安装或检查失败即停止。

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Offline
~~~

上面适用于本机已具备依赖的共享环境。其他电脑首次安装使用不带 -Offline 的 setup.ps1。已有环境不需要再运行 python -m venv .venv。

## 本机采用的修复

1. 原有且未装完依赖的 .venv 保存在 .venv.before-repair，未删除。
2. 使用已验证的 D:\conda\envs\tee\python.exe 创建新的 .venv，并通过 --system-site-packages 共享基础环境中的包。
3. 创建时使用 --without-pip，不重复安装本地 pip 和 setuptools，而是使用基础环境提供的构建工具。
4. 离线执行 editable 安装，仅将本项目安装到 .venv，不修改 tee 中的依赖。
5. VS Code 工作区的默认 Python 路径指向 .venv\Scripts\python.exe，原有其他工作区偏好保留。

本机关键版本为 PyTorch 2.2.0+cu121、Transformers 4.44.2、Tokenizers 0.19.1、NumPy 1.26.4。CUDA 可用。

共享依赖意味着 D:\conda\envs\tee 需要继续存在；在其他电脑上不能直接复制此 .venv。该电脑应使用自己的 Python 和依赖重新建立环境。

## 本次复发修复验证

已连续执行两次 setup.ps1 -Offline，两次均保留 pyvenv.cfg 原样，PyTorch 与 CUDA 检查通过；四项单元测试全部通过，Hybrid 还完成了两步实际 CUDA 训练。日志位于 .build/environment-repair-repeat。

## 修复后的验证

使用项目 .venv 完成了四项单元测试、真实数据预处理、Transformer 与 Hybrid 各两步 CUDA 训练以及 Hybrid 检查点加载和演示入口检查。预处理仍保留 676 对文本，划分仍为 491 / 98 / 87。检查日志位于 .build/environment-repair，原有 runs 中的里程碑结果未覆盖。

## 现在如何运行

在项目根目录使用明确的解释器路径，不需要先激活环境，也不依赖 PYTHONPATH。

~~~powershell
.\.venv\Scripts\python.exe -c "import torch, academic_rewriter; print(torch.__version__); print(torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m academic_rewriter.prepare
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

训练时可指定新目录，保留原有里程碑运行证据。

~~~powershell
.\.venv\Scripts\python.exe -m academic_rewriter.run --tiny --model transformer --output runs/transformer_local --steps 30
.\.venv\Scripts\python.exe -m academic_rewriter.run --tiny --model hybrid --backend reference --fusion gated --output runs/hybrid_local --steps 30
~~~

这些小模型用于流程验证，不能当作可用学术改写模型。后续安装本项目仍应使用 python -m pip，不要用可能属于其他解释器的裸 pip 命令。

如需在当前已修复环境中重新注册本项目，可运行：

~~~powershell
.\.venv\Scripts\python.exe -m pip --isolated install --no-index --no-deps --no-build-isolation -e .
~~~

上述离线命令要求依赖和构建工具已经齐全，不能用于尚无依赖的全新环境。

## 在其他电脑上处理下载超时

先使用完整项目的标准安装流程，并适当增加读取超时和重试次数。仅当安装成功后才继续运行数据处理或测试。

~~~powershell
python -m pip install --timeout 120 --retries 8 -e .
if ($LASTEXITCODE -ne 0) { throw "安装失败，请先处理下载或依赖问题" }
~~~

如果所用网络长期无法访问下载站点，需要处理网络连接或使用自己信任、适用的包源。

## 官方文档

- [Python venv 文档](https://docs.python.org/3.11/library/venv.html)：基础解释器、共享包选项与重新创建环境。
- [pip 选项文档](https://pip.pypa.io/en/stable/cli/pip/#general-options)：timeout 和 retries 参数。
