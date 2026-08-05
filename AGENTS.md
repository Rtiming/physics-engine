# physics-engine — AI 协作规则

> 本项目遵循 rtime-project 跨设备范式。任何 AI 助手开工前先读本文件。

## 项目定位

- 一句话: winding-deviation-sim与fts-digital-twin的共同引擎仓，规范先行、代码后置
- 开发设备: Windows + macOS(平级);运行/部署: 本地库，无服务部署
- git 远程: ts-orangepi:physics-engine.git（/home/orangepi/，与fts-digital-twin.git、case2-digital-twin.git同处）（开源时另加 github.com/Rtiming/physics-engine） <!-- rtime-project: allow-abs -->

## 路径策略(硬性)

- 禁止新增写死的用户主目录/盘符绝对路径(`C:\Users\...`、`/Users/...`、`/home/...`、`/mnt/...`)。 <!-- rtime-project: allow-abs -->
- 路径从仓库根/当前文件计算: Python `pathlib.Path(__file__)`,Node `path`/`import.meta.dirname`,文档用相对路径。
- 真正的外部工程路径(如 Motion Perfect / WorkVisual)才保留绝对路径,并在 `.rtime-project-allow` 或行内 `rtime-project: allow-abs` 豁免。

## 同步与 git

- 代码只走 git(push/pull/fetch 或 .bundle),不靠文件夹同步搬 `.git`。
- 不经用户明确指示,不 `git push` 远程。
- 缓存、虚拟环境、`node_modules`、构建产物、AI 缓存、按机器跑出的产物不进版本控制,也不同步。

## 行尾与编码

- 仓库已配 `.gitattributes`(`* text=auto eol=lf`)与 `.editorconfig`(UTF-8/LF)。新增文件遵循之,Windows 脚本除外(CRLF)。

## 多 AI 协作

- AI 缓存(`.codex`/`.claude`/`.playwright-mcp`)在仓库内但 git/sync 均排除。
- 子代理只作建议,主代理拍板。
- 结构性大改动在 `<docs 或 00_project_docs>` 留审计痕迹。
- 提交署名 `[ai:名@设备]`。

## 改完自检

- 跑 `python tools/rtime-project-check.py . --strict`,清掉所有 [E]:硬编码路径、断链、超长 Windows 路径;并确认行尾归一、无意外脏 git。

## 设备安全(若涉及外部系统时填写)

- <对生产服务器/外部系统的写操作需用户明确授权;优先只读诊断、离线验证>
