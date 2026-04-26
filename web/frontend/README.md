# Lionet Frontend

## 本地开发

```bash
cd /Users/wenglongao/work_repo/lionet/web/frontend
npm install
npm run dev
```

## GitHub Pages 托管（已接入自动部署）

本项目已添加工作流：`.github/workflows/deploy-frontend-gh-pages.yml`。

### 1) 设置后端公网地址（非常关键）

GitHub Pages 是静态站点，前端要访问你 Mac 上运行的后端，必须配置可访问的后端 URL。

在 GitHub 仓库设置中添加变量：

- 路径：`Settings` -> `Secrets and variables` -> `Actions` -> `Variables`
- 名称：`VITE_API_BASE_URL`
- 值示例：`https://your-backend.example.com`

> 如果你的后端只监听 `localhost:8000`，外网设备无法访问。你需要把后端映射到公网（如反向代理/隧道/DDNS）。

### 2) 开启 GitHub Pages

1. 进入仓库：`https://github.com/leoweng111/lionet`
2. 打开 `Settings` -> `Pages`
3. `Build and deployment` 里选择：
   - `Source: GitHub Actions`

### 3) 触发部署

以下任一方式会触发部署：

- push 到 `main` 且涉及 `web/frontend/**`
- 在 `Actions` 页面手动运行 `Deploy Frontend To GitHub Pages`

### 4) 访问地址

部署成功后访问：

- `https://leoweng111.github.io/lionet/`

## 备注

- Vite 的 `base` 已自动适配 GitHub Pages 仓库子路径。
- 已生成 `404.html` 用于 Vue Router 刷新回退。
- 已添加 `.nojekyll`，避免静态资源被 Jekyll 处理。
