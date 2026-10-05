# ABC Flow Orbit Explorer — Dash

這個專案把原本 MATLAB GUI 改成瀏覽器版 Dash App。

## 功能

- 左側顯示預先計算的 `400×400` front-speed color map。
- 直接點 front-speed 圖選擇 `(y0,z0)`。
- `epsilon`, `y0`, `z0`, `T` 輸入框開啟時全部留白。
- 按 **Update** 後，用 Python/SciPy 積分一條 ABC flow trajectory。
- 3D Plotly 圖可直接旋轉、縮放、平移。
- 如果目前 trajectory 的 `epsilon` 或 `T` 與背景 front-speed map 不同，介面會顯示警告。

ABC flow 方程與目前 MATLAB 程式一致：

```text
dx/dt = epsilon*sin(z) + cos(y)
dy/dt = sin(x) + epsilon*cos(z)
dz/dt = sin(y) + cos(x)
```

網頁版使用 SciPy `solve_ivp(..., method="DOP853")` 做互動式數值積分。

## 1. 放入 front-speed MAT 檔

把 MATLAB 算好的檔案放到：

```text
data/frontspeed_N400_T100_eps_1.mat
```

至少需要：

```text
speed_abc_dir
y
z
```

建議也保留：

```text
epsi
T
N
x0
```

普通 MAT 檔與 MATLAB `-v7.3` HDF5 MAT 檔都支援。

## 2. 建立 Python 環境

macOS / Linux：

```bash
cd abc_dash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
cd abc_dash
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. 本機執行

```bash
python app.py
```

瀏覽器開啟：

```text
http://127.0.0.1:8050
```

## 4. 部署到雲端

Linux hosting 的 production start command 可用：

```bash
gunicorn app:server
```

例如 Render / Railway / Fly.io 等支援 Python 的平台都可部署。

## 專案結構

```text
abc_dash/
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   └── style.css
├── data/
│   └── frontspeed_N400_T100_eps_1.mat   <- 你自己放進來
└── matlab/
    └── frontspeed_large_batch_400.m
```

## 研究用途提醒

這個網頁版適合找候選點、看 front-speed 區域與展示 trajectory。長時間混沌軌跡可能因 solver 與浮點誤差逐漸分離，因此正式 existence / shooting / continuation / INTLAB 驗證仍建議使用原本研究流程。
