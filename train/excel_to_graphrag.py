"""
excel_to_graphrag.py
====================
将 /root/autodl-tmp/train-time-table/ 目录下的 Excel 时刻表文件
转换为：
  1. schedule_kb/excel_schedules.json  （供 MultiSourceKB 直接加载）
  2. graphrag/triples.json             （三元组，供 load_graph 使用）
  3. graphrag/graph.pkl                （NetworkX 图，供 load_graph 使用）

【调用方式】在 main() 的 init_sample_knowledge_files() 之后添加一行：
    import_excel_schedules_to_kb(EXCEL_TIMETABLE_DIR)

【对原代码的改动】
  - 顶部新增常量：EXCEL_TIMETABLE_DIR
  - main() 中插入一行调用
  - 本文件作为独立模块 import，原文件其余部分完全不变
"""

import os, re, json, pickle, warnings
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────
# 列名别名映射（兼容各种中英文表头）
# ──────────────────────────────────────────────
_COL_ALIASES = {
    "train":    ["车次", "列车编号", "车号", "train", "train_no", "trainno", "编号"],
    "station":  ["站名", "车站", "站点", "停靠站", "station", "stop", "站"],
    "arrive":   ["到达时间", "到达时刻", "到达", "到站时间", "到站", "arr", "arrive", "arrival", "到"],
    "depart":   ["发车时间", "发车时刻", "发车", "出发时间", "出发", "开车时间", "dep", "depart", "departure", "发"],
    "time":     ["时间", "时刻", "停靠时刻", "time"],
    "km":       ["里程", "公里", "km", "距离", "mileage", "dist"],
    "color":    ["颜色", "color", "线色"],
    "seq":      ["顺序", "序号", "order", "seq", "no", "编号"],
}

_TRAIN_COLORS = [
    "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
    "#1abc9c","#e67e22","#e91e63","#00bcd4","#8bc34a",
]


def _match_col(df_cols, aliases):
    """在 df_cols 中找第一个匹配 aliases 的列名（忽略大小写/空格）"""
    norm = {c.strip().lower(): c for c in df_cols}
    for a in aliases:
        if a.strip().lower() in norm:
            return norm[a.strip().lower()]
    return None


def _parse_time(val):
    """
    将各种格式的时间值统一转为 'HH:MM' 字符串。
    支持：
      - datetime / time 对象
      - float (Excel序列时间)
      - '9:00' / '09:00' / '21:09 (当日)' / '06:49 (次日)'
      - '9时00分' / '090000'
      - '----' / '--' → None（始发终到站无时间）
    次日时间自动 +24，使时序单调递增，绘图不折回。
    """
    import math
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None

    # datetime / time 对象
    if hasattr(val, "hour"):
        return f"{val.hour:02d}:{val.minute:02d}"

    # Excel 小数时间（0~1）
    if isinstance(val, float):
        if 0.0 <= val < 1.0:
            total = round(val * 1440)
            return f"{total // 60:02d}:{total % 60:02d}"
        frac = val - int(val)
        if frac > 0:
            total = round(frac * 1440)
            return f"{total // 60:02d}:{total % 60:02d}"
        return None

    s = str(val).replace('\xa0', ' ').replace('\u3000', ' ').strip()

    # 空值 / 无时间标记（---- 或纯横线）
    if not s or s.lower() in ("nan", "none", ""):
        return None
    if re.fullmatch(r"[-—－＊\s]+", s):
        return None

    # 次日判断（+24h 偏移，保持时序单调）
    next_day = bool(re.search(r"次日|翌日|next.?day|\+1", s, re.IGNORECASE))

    # HH:MM 或 H:MM（含 (当日)/(次日) 后缀）
    m = re.search(r"(\d{1,2})[：:](\d{2})", s)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if next_day:
            h += 24
        return f"{h:02d}:{mi:02d}"

    # 纯6位数字：090000 → 09:00
    if re.fullmatch(r"\d{6}", s):
        return f"{s[:2]}:{s[2:4]}"

    # 中文：9时00分
    m = re.search(r"(\d{1,2})时(\d{2})分?", s)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if next_day:
            h += 24
        return f"{h:02d}:{mi:02d}"

    return None


def _infer_km(stops):
    """
    若里程列全为0或缺失，按均匀分布估算（假设全程1318km，京沪高铁默认）。
    仅供无里程数据时的降级处理。
    """
    n = len(stops)
    if n < 2:
        return stops
    all_zero = all(s.get("km", 0) == 0 for s in stops)
    if all_zero:
        total = 1318  # 默认全程，实际使用时会被真实数据覆盖
        for i, s in enumerate(stops):
            s["km"] = round(total * i / (n - 1))
    return stops


# ──────────────────────────────────────────────
# 核心：解析单个 Sheet
# ──────────────────────────────────────────────
def _parse_sheet(data: list, default_train_name: str) -> list:
    """
    解析列表格式数据（每个元素是字典），返回 schedule 列表。
    data: 从CSV读取的列表，每个元素是 {"列名1": 值1, "列名2": 值2, ...}
    """
    if not data:
        return []

    # 1. 获取所有列名（从第一行字典中提取）
    if not data[0]:
        return []
    cols = list(data[0].keys())

    # 2. 匹配关键列（与原逻辑一致）
    train_col   = _match_col(cols, _COL_ALIASES["train"])
    station_col = _match_col(cols, _COL_ALIASES["station"])
    time_col    = _match_col(cols, _COL_ALIASES["time"])
    arr_col     = _match_col(cols, _COL_ALIASES["arrive"])
    dep_col     = _match_col(cols, _COL_ALIASES["depart"])
    km_col      = _match_col(cols, _COL_ALIASES["km"])
    color_col   = _match_col(cols, _COL_ALIASES["color"])

    schedules = {}

    # 3. 逐行解析数据（遍历字典列表）
    for row in data:
        # 站名（必须非空）
        station = row.get(station_col)
        if not station or str(station).lower() == "nan":
            continue
        station = str(station).strip()

        # 车次（有列用列值，无列用默认名）
        if train_col:
            train_name = row.get(train_col)
            train_name = str(train_name).strip() if train_name is not None else default_train_name
            if not train_name or train_name.lower() == "nan":
                train_name = default_train_name
        else:
            train_name = default_train_name

        # 时间解析（优先发车时间，其次到达时间）
        t_val = None
        for src_col in [dep_col, arr_col, time_col]:
            if src_col is None:
                continue
            cell_val = row.get(src_col)
            if cell_val is None:
                continue
            t_val = _parse_time(cell_val)
            if t_val:
                break
        if not t_val:
            continue

        # 里程解析
        km = 0.0
        if km_col:
            km_val = row.get(km_col)
            if km_val is not None:
                try:
                    km = float(km_val)
                except (ValueError, TypeError):
                    km = 0.0

        # 颜色解析
        color = ""
        if color_col:
            color_val = row.get(color_col)
            if color_val is not None:
                color = str(color_val).strip()

        # 加入时刻表字典
        if train_name not in schedules:
            schedules[train_name] = {"train": train_name, "color": color, "stops": []}
        if not schedules[train_name]["color"] and color:
            schedules[train_name]["color"] = color

        schedules[train_name]["stops"].append({
            "station": station,
            "time": t_val,
            "km": km,
        })

    # 4. 过滤无效数据、补全颜色和里程
    result = []
    for i, (name, sch) in enumerate(schedules.items()):
        if len(sch["stops"]) < 2:
            continue  # 至少2个站点才有效
        # 补全颜色（无颜色时用默认色表）
        if not sch["color"]:
            sch["color"] = _TRAIN_COLORS[i % len(_TRAIN_COLORS)]
        # 补全里程（无里程时均匀分配）
        sch["stops"] = _infer_km(sch["stops"])
        result.append(sch)

    return result


def _find_soffice() -> str:
    """动态查找 soffice/libreoffice 可执行文件，找不到则抛出清晰错误。"""
    import shutil, subprocess
    # 1. PATH 里直接找
    for name in ("soffice", "libreoffice"):
        path = shutil.which(name)
        if path:
            return path
    # 2. 常见固定路径
    candidates = [
        "/usr/bin/soffice", "/usr/bin/libreoffice",
        "/usr/lib/libreoffice/program/soffice",
        "/opt/libreoffice/program/soffice",
        "/snap/bin/libreoffice",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # 3. find 兜底
    try:
        r = subprocess.run(["find", "/usr", "-name", "soffice", "-type", "f"],
                           capture_output=True, text=True, timeout=10)
        for line in r.stdout.strip().splitlines():
            if line.strip():
                return line.strip()
    except Exception:
        pass
    raise FileNotFoundError(
        "未找到 LibreOffice，请运行安装命令：\n"
        "  apt-get install -y libreoffice-calc"
    )


def _read_xls_native(filepath: str) -> dict:
    """
    用 LibreOffice 将 .xls 转为 CSV，再用原生csv模块读取。
    完全绕开 pandas/numpy 兼容性问题，返回 {sheet_name: 列表格式数据}。
    """
    import subprocess, tempfile, shutil, glob, csv

    tmpdir = tempfile.mkdtemp(prefix="xls2csv_")
    try:
        # 1. 调用 LibreOffice 转换 .xls 为 CSV
        cmd = [
            _find_soffice(), "--headless", "--norestore",
            "--convert-to", "csv:Text - txt - csv (StarCalc):44,34,76,1,,1033,true,true",
            "--outdir", tmpdir,
            filepath
        ]
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if ret.returncode != 0:
            raise RuntimeError(f"LibreOffice 转换失败: {ret.stderr[:200]}")

        # 2. 查找转换后的 CSV 文件
        csv_files = sorted(glob.glob(os.path.join(tmpdir, "*.csv")))
        if not csv_files:
            raise RuntimeError("LibreOffice 未生成 CSV 文件")

        result = {}
        base = os.path.splitext(os.path.basename(filepath))[0]

        # 3. 逐个读取 CSV 文件（原生csv模块）
        for csv_path in csv_files:
            csv_name = os.path.splitext(os.path.basename(csv_path))[0]
            # 解析 sheet 名称（兼容 LibreOffice 命名规则）
            if csv_name == base:
                sheet_name = "Sheet1"
            elif csv_name.startswith(base + "-"):
                sheet_name = csv_name[len(base)+1:]
            else:
                sheet_name = csv_name

            # 4. 尝试多种编码读取 CSV
            all_rows = []
            for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
                try:
                    with open(csv_path, encoding=enc, newline="") as f:
                        reader = csv.reader(f)
                        all_rows = [row for row in reader]  # 原生列表，无numpy类型
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            else:
                print(f"  [!] 无法解码 {os.path.basename(csv_path)}，跳过")
                continue

            if not all_rows:
                continue

            # 5. 处理列名（去重、清理空白）
            raw_header = [str(c).replace("\xa0", " ").replace("\u3000", " ").strip() for c in all_rows[0]]
            header = []
            seen = {}
            for h in raw_header:
                h = h or "unnamed"
                if h in seen:
                    seen[h] += 1
                    h = f"{h}_{seen[h]}"
                else:
                    seen[h] = 0
                header.append(h)

            # 6. 处理数据行（清理空白、统一空值）
            data = []
            for row in all_rows[1:]:
                # 补齐/截断到列数，避免行长度不统一
                padded_row = (row + [""] * len(header))[:len(header)]
                # 清理每个单元格，空字符串转为 None
                cleaned_row = [str(cell).replace("\xa0", " ").strip() or None for cell in padded_row]
                data.append(dict(zip(header, cleaned_row)))  # 字典格式，便于后续解析

            result[sheet_name] = data  # 存储为列表+字典，无pandas/numpy类型

        return result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_excel_file(filepath: str) -> list:
    """读取一个 Excel 文件，返回所有 sheet 解析后的 schedule 列表"""
    all_schedules = []
    is_xls = filepath.lower().endswith(".xls")
    base_name = os.path.splitext(os.path.basename(filepath))[0]

    # ── .xls 文件：使用原生解析（返回 list 格式）──
    if is_xls:
        try:
            sheets_dict = _read_xls_native(filepath)
        except Exception as e:
            print(f"  [!] 无法读取 {os.path.basename(filepath)}: {e}")
            return []

        for sheet_name, data in sheets_dict.items():
            # 列表判空，不用 .empty
            if not data:
                print(f"  [!] Sheet '{sheet_name}' 为空，跳过")
                continue
            default_train = base_name if len(sheets_dict) == 1 else sheet_name
            try:
                schedules = _parse_sheet(data, default_train)
            except Exception as e:
                print(f"  [!] Sheet '{sheet_name}' 解析失败: {e}")
                continue
            if schedules:
                print(f"  [✓] {os.path.basename(filepath)} / {sheet_name}: 解析到 {len(schedules)} 趟列车")
            all_schedules.extend(schedules)
        return all_schedules

    # ── .xlsx 文件：用 openpyxl 读取，转成字典列表 ──
    try:
        xl = pd.ExcelFile(filepath, engine="openpyxl")
    except Exception as e:
        print(f"  [!] 无法读取 {os.path.basename(filepath)}: {e}")
        return []

    for sheet_name in xl.sheet_names:
        try:
            df = xl.parse(sheet_name, header=0)
        except Exception as e:
            print(f"  [!] Sheet '{sheet_name}' 解析失败: {e}")
            continue

        # 转字典列表，统一格式
        dict_data = df.to_dict('records')
        default_train = f"{base_name}" if len(xl.sheet_names) == 1 else f"{sheet_name}"
        schedules = _parse_sheet(dict_data, default_train)

        if schedules:
            all_schedules.extend(schedules)
            print(f"  [✓] {os.path.basename(filepath)} / {sheet_name}: 解析到 {len(schedules)} 趟列车")

    return all_schedules

# ──────────────────────────────────────────────
# 三元组 & 图谱构建
# ──────────────────────────────────────────────
def _build_triples(schedules: list) -> list:
    """
    从时刻表生成面向绘图的三元组，包含：
      - (列车, 停靠, 车站)
      - (列车, 始发, 首站)
      - (列车, 终到, 末站)
      - (车站A, 区间前序, 车站B)        按实际运行顺序
      - (列车, 到达时刻, HH:MM@车站)
      - (列车, 车型, G/D/K/...)
      - (列车, 颜色, #hex)
    """
    triples = []
    seen = set()

    def add(h, r, t):
        key = (h, r, t)
        if key not in seen:
            seen.add(key)
            triples.append({"h": h, "r": r, "t": t})

    for sch in schedules:
        train = sch["train"]
        stops = sch.get("stops", [])
        color = sch.get("color", "")

        if not stops:
            continue

        # 车型
        m = re.match(r"([A-Za-z]+)", train)
        train_type = m.group(1).upper() if m else "未知"
        add(train, "车型", train_type)

        # 颜色（供绘图直接使用）
        if color:
            add(train, "颜色", color)

        # 始发/终到
        add(train, "始发", stops[0]["station"])
        add(train, "终到", stops[-1]["station"])

        for i, stop in enumerate(stops):
            station = stop["station"]
            time_str = stop["time"]

            # 停靠关系
            add(train, "停靠", station)
            # 到达时刻（用于绘图定位）
            add(train, "到达时刻", f"{time_str}@{station}")
            # 里程
            km = stop.get("km", 0)
            if km:
                add(train, "里程标", f"{km}km@{station}")

            # 区间顺序（相邻站对）
            if i + 1 < len(stops):
                next_station = stops[i + 1]["station"]
                add(station, "区间前序", next_station)
                add(train, "区间运行", f"{station}→{next_station}")

    return triples


def _build_graph(triples: list) -> nx.DiGraph:
    """将三元组构建为有向图"""
    G = nx.DiGraph()
    for tri in triples:
        h, r, t = tri["h"], tri["r"], tri["t"]
        G.add_node(h)
        G.add_node(t)
        G.add_edge(h, t, relation=r)
    return G


# ──────────────────────────────────────────────
# 对外接口：主入口函数
# ──────────────────────────────────────────────
def import_excel_schedules_to_kb(
    excel_dir: str,
    schedule_kb_dir: str,
    triple_path: str,
    graph_path: str,
    force_reimport: bool = False,
):
    """
    扫描 excel_dir 下所有 .xlsx/.xls 文件，解析后：
      1. 写入 schedule_kb_dir/excel_schedules.json
      2. 写入 triple_path（triples.json）—— 追加模式，不覆盖已有非Excel三元组
      3. 写入 graph_path（graph.pkl）

    参数：
        excel_dir        Excel 时刻表所在目录
        schedule_kb_dir  schedule_kb 目录（与原代码 SCHEDULE_KB_DIR 一致）
        triple_path      triples.json 路径（与原代码 TRIPLE_PATH 一致）
        graph_path       graph.pkl 路径（与原代码 GRAPH_PATH 一致）
        force_reimport   True 则忽略缓存强制重新解析
    """
    out_json = os.path.join(schedule_kb_dir, "excel_schedules.json")

    # ── 找 Excel 文件 ──
    if not os.path.isdir(excel_dir):
        print(f"  [!] Excel目录不存在: {excel_dir}，跳过Excel导入")
        return

    excel_files = [
        os.path.join(excel_dir, f)
        for f in os.listdir(excel_dir)
        if f.lower().endswith((".xlsx", ".xls")) and not f.startswith("~$")
    ]

    if not excel_files:
        print(f"  [!] {excel_dir} 中未找到 Excel 文件，跳过Excel导入")
        return

    # ── 检查是否需要重新解析 ──
    if not force_reimport and os.path.exists(out_json):
        out_mtime = os.path.getmtime(out_json)
        newest_excel = max(os.path.getmtime(f) for f in excel_files)
        if newest_excel <= out_mtime:
            print(f"  [*] Excel缓存有效，跳过重新解析（共{len(excel_files)}个文件）")
            print(f"      如需强制重新导入，删除: {out_json}")
            return

    print(f"\n{'─'*60}")
    print(f"  [*] 开始导入 Excel 时刻表（{len(excel_files)} 个文件）...")

    # ── 解析所有文件 ──
    all_schedules = []
    for fp in sorted(excel_files):
        print(f"  [>] {os.path.basename(fp)}")
        all_schedules.extend(_parse_excel_file(fp))

    if not all_schedules:
        print("  [!] 所有 Excel 文件均未解析出有效数据，请检查表头格式")
        print("      支持的列名：车次/station/train + 站名/站点/stop + 时间/time/到达/发车 等")
        return

    # ── 写入 schedule_kb ──
    os.makedirs(schedule_kb_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_schedules, f, ensure_ascii=False, indent=2)
    print(f"  [✓] 时刻表已写入: {out_json}（{len(all_schedules)} 趟列车）")

    # ── 构建三元组 ──
    new_triples = _build_triples(all_schedules)

    # 追加模式：若已有 triples.json（非Excel来源），保留并去重合并
    existing_triples = []
    if os.path.exists(triple_path):
        try:
            existing_triples = json.load(open(triple_path, encoding="utf-8"))
        except Exception:
            existing_triples = []

    # 去重合并（以 h+r+t 为键）
    existing_keys = {(x["h"], x["r"], x["t"]) for x in existing_triples}
    merged = list(existing_triples)
    added = 0
    for tri in new_triples:
        key = (tri["h"], tri["r"], tri["t"])
        if key not in existing_keys:
            merged.append(tri)
            existing_keys.add(key)
            added += 1

    os.makedirs(os.path.dirname(triple_path), exist_ok=True)
    with open(triple_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"  [✓] 三元组已写入: {triple_path}（新增 {added} 条，共 {len(merged)} 条）")

    # ── 构建图谱 ──
    G = _build_graph(merged)
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
    print(f"  [✓] 图谱已写入: {graph_path}（{G.number_of_nodes()} 节点，{G.number_of_edges()} 边）")
    print(f"{'─'*60}\n")


# ──────────────────────────────────────────────
# 独立运行调试入口
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    BASE_DIR        = "/root/autodl-tmp/graphrag"
    SCHEDULE_KB_DIR = os.path.join(BASE_DIR, "schedule_kb")
    TRIPLE_PATH     = os.path.join(BASE_DIR, "triples.json")
    GRAPH_PATH      = os.path.join(BASE_DIR, "graph.pkl")
    EXCEL_DIR       = "/root/autodl-tmp/train-time-table"

    force = "--force" in sys.argv

    import_excel_schedules_to_kb(
        excel_dir=EXCEL_DIR,
        schedule_kb_dir=SCHEDULE_KB_DIR,
        triple_path=TRIPLE_PATH,
        graph_path=GRAPH_PATH,
        force_reimport=force,
    )

    # 打印预览
    out = os.path.join(SCHEDULE_KB_DIR, "excel_schedules.json")
    if os.path.exists(out):
        data = json.load(open(out, encoding="utf-8"))
        print(f"\n预览（前3趟）:")
        for sch in data[:3]:
            stops_str = " → ".join(
                f"{s['station']}({s['time']})" for s in sch["stops"][:4]
            )
            print(f"  {sch['train']} [{sch['color']}]: {stops_str} ...")