"""
GraphRAG 智能铁路运行图系统 v4.0
=====================================
分层知识库设计：
  schedule_kb/  - 列车时刻表 JSON/CSV（最核心）
  route_kb/     - 线路拓扑 JSON
  drawing_kb/   - 绘图规范 MD/TXT
  rule_kb/      - 运行规则 MD/TXT
  train_picture.md - 原有绘图知识文档
"""

import os, re, sys, json, csv, torch, pickle
import chromadb, numpy as np, networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from excel_to_graphrag import import_excel_schedules_to_kb

rcParams['font.family'] = ['WenQuanYi Zen Hei', 'sans-serif']
rcParams['axes.unicode_minus'] = False

# ================================
# 路径配置
# ================================
MODEL_PATH       = "/root/autodl-tmp/Qwen2.5-7B-finetuned"
EMBED_MODEL_PATH = "/root/autodl-tmp/bge-large-zh-v1.5"
DOCUMENT_PATH    = "/root/autodl-tmp/train picture1.md"
EXCEL_TIMETABLE_DIR = "/root/autodl-tmp/train-time-table"

BASE_DIR         = "/root/autodl-tmp/graphrag"
TRIPLE_PATH      = os.path.join(BASE_DIR, "triples.json")
GRAPH_PATH       = os.path.join(BASE_DIR, "graph.pkl")
DIAGRAM_DIR      = os.path.join(BASE_DIR, "diagrams")
EVAL_DIR         = os.path.join(BASE_DIR, "evaluations")

# 四类知识库目录
SCHEDULE_KB_DIR  = os.path.join(BASE_DIR, "schedule_kb")  # 时刻表 *.json/*.csv
ROUTE_KB_DIR     = os.path.join(BASE_DIR, "route_kb")     # 线路拓扑 *.json
DRAWING_KB_DIR   = os.path.join(BASE_DIR, "drawing_kb")   # 绘图规范 *.md/*.txt
RULE_KB_DIR      = os.path.join(BASE_DIR, "rule_kb")      # 运行规则 *.md/*.txt

for d in [BASE_DIR, DIAGRAM_DIR, EVAL_DIR,
          SCHEDULE_KB_DIR, ROUTE_KB_DIR, DRAWING_KB_DIR, RULE_KB_DIR]:
    os.makedirs(d, exist_ok=True)

DEEPSEEK_API_KEY  = "sk-1bd4a524b8e04859a68afb7a02d53467"
deepseek_client   = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# ================================
# 彩色打印
# ================================
class C:
    RESET="\033[0m"; BOLD="\033[1m"; CYAN="\033[96m"; GREEN="\033[92m"
    YELLOW="\033[93m"; RED="\033[91m"; BLUE="\033[94m"; GRAY="\033[90m"
    WHITE="\033[97m"; MAGENTA="\033[95m"

def ps(m): print(f"{C.BLUE}[*]{C.RESET} {m}")
def po(m): print(f"{C.GREEN}[✓]{C.RESET} {m}")
def pw(m): print(f"{C.YELLOW}[!]{C.RESET} {m}")
def pe(m): print(f"{C.RED}[✗]{C.RESET} {m}")
def pa(t): print(f"\n{C.WHITE}{C.BOLD}━━━ 回答 ━━━{C.RESET}\n{C.WHITE}{t}{C.RESET}\n")
def pev(t): print(f"\n{C.MAGENTA}{C.BOLD}━━━ DeepSeek评价 ━━━{C.RESET}\n{C.MAGENTA}{t}{C.RESET}\n")
def pdiv(): print(f"{C.GRAY}{'─'*60}{C.RESET}")

def banner():
    print(f"""{C.CYAN}{C.BOLD}
╔══════════════════════════════════════════════════════╗
║  🚄  GraphRAG 铁路运行图 v4.0 - 多源知识库版         ║
║  时刻表库+线路库+规范库+知识图谱+DeepSeek评价         ║
╚══════════════════════════════════════════════════════╝
{C.RESET}""")

# ================================
# 初始化示例知识文件（首次运行自动创建，可用真实数据替换）
# ================================
def init_sample_knowledge_files():
    """
    【文件放置说明】
    schedule_kb/ 放时刻表：
      JSON格式: [{"train":"G1","color":"#e74c3c","stops":[{"station":"北京南","time":"09:00","km":0},...]}]
      CSV格式:  train,station,time,km,color
    route_kb/ 放线路：
      JSON格式: [{"line":"京沪高铁","total_km":1318,"design_speed":350,
                  "stations":[{"name":"北京南","km":0},{"name":"上海虹桥","km":1318}]}]
    drawing_kb/ 放绘图规范文本
    rule_kb/ 放运行规则文本
    """

    sch_file = os.path.join(SCHEDULE_KB_DIR, "sample_schedules.json")
    if not os.path.exists(sch_file):
        sample = [
            {"train":"G1","color":"#e74c3c","stops":[
                {"station":"北京南","time":"09:00","km":0},
                {"station":"济南西","time":"10:42","km":406},
                {"station":"南京南","time":"12:55","km":1023},
                {"station":"上海虹桥","time":"14:28","km":1318}]},
            {"train":"G2","color":"#3498db","stops":[
                {"station":"上海虹桥","time":"09:00","km":0},
                {"station":"南京南","time":"10:32","km":295},
                {"station":"济南西","time":"12:45","km":912},
                {"station":"北京南","time":"14:28","km":1318}]},
            {"train":"G7","color":"#2ecc71","stops":[
                {"station":"北京南","time":"06:08","km":0},
                {"station":"济南西","time":"07:50","km":406},
                {"station":"徐州东","time":"09:00","km":693},
                {"station":"南京南","time":"09:55","km":1023},
                {"station":"上海虹桥","time":"11:18","km":1318}]},
            {"train":"G101","color":"#f39c12","stops":[
                {"station":"北京南","time":"07:00","km":0},
                {"station":"天津南","time":"07:27","km":122},
                {"station":"济南西","time":"08:48","km":406},
                {"station":"徐州东","time":"09:58","km":693},
                {"station":"南京南","time":"10:53","km":1023},
                {"station":"上海虹桥","time":"12:13","km":1318}]},
        ]
        json.dump(sample, open(sch_file,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        po(f"已创建示例时刻表: {sch_file}")

    route_file = os.path.join(ROUTE_KB_DIR, "sample_routes.json")
    if not os.path.exists(route_file):
        routes = [{"line":"京沪高铁","total_km":1318,"design_speed":350,"stations":[
            {"name":"北京南","km":0},{"name":"廊坊","km":60},{"name":"天津南","km":122},
            {"name":"沧州西","km":237},{"name":"德州东","km":323},{"name":"济南西","km":406},
            {"name":"泰安","km":463},{"name":"曲阜东","km":508},{"name":"滕州东","km":567},
            {"name":"枣庄","km":607},{"name":"徐州东","km":693},{"name":"宿州东","km":766},
            {"name":"蚌埠南","km":835},{"name":"定远","km":896},{"name":"滁州","km":940},
            {"name":"南京南","km":1023},{"name":"镇江南","km":1087},{"name":"丹阳北","km":1107},
            {"name":"常州北","km":1154},{"name":"无锡东","km":1190},{"name":"苏州北","km":1234},
            {"name":"上海虹桥","km":1318}]}]
        json.dump(routes, open(route_file,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        po(f"已创建示例线路: {route_file}")

    draw_file = os.path.join(DRAWING_KB_DIR, "drawing_spec.md")
    if not os.path.exists(draw_file):
        open(draw_file,"w",encoding="utf-8").write("""# 列车运行图绘制规范
横轴为时间（分钟），纵轴为车站顺序，每段折线代表一列车运行轨迹。
高铁G字头用实线2.5px，动车D字头用实线2px，快速K字头用虚线1.5px。
下行（奇数车次）用暖色，上行（偶数车次）用冷色。
时间刻度每30分钟一条网格线，站点处标注到发时刻，折线中部标注车次号。
""")
        po(f"已创建绘图规范: {draw_file}")

    rule_file = os.path.join(RULE_KB_DIR, "operation_rules.md")
    if not os.path.exists(rule_file):
        open(rule_file,"w",encoding="utf-8").write("""# 铁路运行规则
高铁正线最小追踪间隔3分钟（350km/h），4分钟（250km/h）。
G字头奇数下行（北京→上海方向），偶数上行。
高铁始发终到站停15-30分钟，区段站停2-5分钟，中间小站停1-2分钟。
京沪高铁G字头全程约4.5-5小时，D字头约6-7小时。
站间运行时间=站间距离/设计速度×60分钟，并加减速附加时分。
""")
        po(f"已创建运行规则: {rule_file}")


# ================================
# 多源知识库
# ================================
class MultiSourceKB:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.schedules: List[Dict] = []
        self.routes: List[Dict] = []
        self.drawing_docs: List[str] = []
        self.rule_docs: List[str] = []
        self._db = chromadb.Client()
        self._cols = {}

    def _try_get_col(self, name):
        try: return self._db.get_collection(name)
        except: return None

    def _build_col(self, name, texts):
        col = self._try_get_col(name)
        if col: return col
        col = self._db.create_collection(name)
        for i, t in enumerate(tqdm(texts, desc=f"  向量化{name}", leave=False)):
            emb = self.embed_model.encode(t)
            col.add(documents=[t], embeddings=[emb.tolist()], ids=[str(i)])
        return col

    def _load_json(self, d):
        items = []
        for f in os.listdir(d):
            if f.endswith(".json"):
                try:
                    data = json.load(open(os.path.join(d,f), encoding="utf-8"))
                    items.extend(data if isinstance(data, list) else [data])
                except Exception as e: pw(f"读取{f}失败:{e}")
        return items

    def _load_csv_schedules(self, d):
        mp = {}
        for f in os.listdir(d):
            if f.endswith(".csv"):
                try:
                    for row in csv.DictReader(open(os.path.join(d,f), encoding="utf-8")):
                        n = row.get("train","").strip()
                        if n not in mp:
                            mp[n] = {"train":n,"color":row.get("color","#3498db"),"stops":[]}
                        mp[n]["stops"].append({
                            "station":row.get("station","").strip(),
                            "time":row.get("time","00:00").strip(),
                            "km":float(row.get("km",0))})
                except Exception as e: pw(f"读取CSV{f}失败:{e}")
        return list(mp.values())

    def _load_texts(self, d, chunk=400):
        chunks = []
        for f in os.listdir(d):
            if f.endswith((".md",".txt")):
                try:
                    text = open(os.path.join(d,f), encoding="utf-8").read()
                    chunks += [text[i:i+chunk].strip() for i in range(0,len(text),chunk) if text[i:i+chunk].strip()]
                except: pass
        return chunks

    def _sch_text(self, s):
        stops = " → ".join(f"{st['station']}({st['time']})" for st in s.get("stops",[]))
        return f"车次{s.get('train','')} 运行路径: {stops}"

    def _route_text(self, r):
        sts = " ".join(st["name"] for st in r.get("stations",[]))
        return f"{r.get('line','')} {r.get('total_km','')}km 经过: {sts}"

    def load_all(self, main_doc_path):
        ps("加载时刻表知识库...")
        self.schedules = self._load_json(SCHEDULE_KB_DIR) + self._load_csv_schedules(SCHEDULE_KB_DIR)
        if self.schedules:
            self._cols["schedule"] = self._build_col("kb_sch", [self._sch_text(s) for s in self.schedules])
            po(f"时刻表: {len(self.schedules)} 条")
        else: pw("时刻表库为空，请在 schedule_kb/ 放置数据文件")

        ps("加载线路知识库...")
        self.routes = self._load_json(ROUTE_KB_DIR)
        if self.routes:
            self._cols["route"] = self._build_col("kb_route", [self._route_text(r) for r in self.routes])
            po(f"线路: {len(self.routes)} 条")

        ps("加载绘图规范库...")
        self.drawing_docs = self._load_texts(DRAWING_KB_DIR)
        if self.drawing_docs:
            self._cols["drawing"] = self._build_col("kb_draw", self.drawing_docs)
            po(f"绘图规范: {len(self.drawing_docs)} 块")

        ps("加载运行规则库...")
        self.rule_docs = self._load_texts(RULE_KB_DIR)
        if self.rule_docs:
            self._cols["rule"] = self._build_col("kb_rule", self.rule_docs)
            po(f"运行规则: {len(self.rule_docs)} 块")

        ps("加载主文档...")
        if os.path.exists(main_doc_path):
            text = open(main_doc_path, encoding="utf-8").read()
            chunks = [text[i:i+400] for i in range(0,len(text),400)]
            self._cols["main"] = self._build_col("kb_main", chunks)
            po(f"主文档: {len(chunks)} 块")

    def _query_col(self, name, query, k):
        col = self._cols.get(name)
        if not col: return []
        emb = self.embed_model.encode(query).tolist()
        n = min(k, col.count())
        if n == 0: return []
        return col.query(query_embeddings=[emb], n_results=n)["documents"][0]

    def by_train_name(self, names):
        return [s for s in self.schedules if s.get("train","") in names]

    def by_stations(self, stations):
        return [s for s in self.schedules
                if all(st in [x["station"] for x in s.get("stops",[])] for st in stations)]

    def query_schedules(self, query, k=4):
        docs = self._query_col("schedule", query, k)
        matched = []
        for doc in docs:
            for s in self.schedules:
                if s.get("train","") in doc and s not in matched:
                    matched.append(s); break
        return matched

    def query_routes(self, query, k=2):
        docs = self._query_col("route", query, k)
        matched = []
        for doc in docs:
            for r in self.routes:
                if r.get("line","") in doc and r not in matched:
                    matched.append(r); break
        return matched

    def query_drawing(self, query, k=3): return self._query_col("drawing", query, k)
    def query_rules(self, query, k=3): return self._query_col("rule", query, k)
    def query_main(self, query, k=3): return self._query_col("main", query, k)


# ================================
# 模型 & 知识图谱
# ================================

def load_models():
    ps("加载语言模型...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True).eval()
    po("语言模型完成")
    ps("加载向量模型...")
    emb = SentenceTransformer(EMBED_MODEL_PATH)
    po("向量模型完成")
    return tok, mdl, emb

# ================================
# 从 train_picture1.md 抽取知识图谱三元组
# ================================
def extract_triples_from_train_picture(md_path):
    if not os.path.exists(md_path):
        return []

    with open(md_path, encoding="utf-8") as f:
        text = f.read()

    triples = []
    patterns = [
        (r'([^\n]{2,8})是([^\n]{2,10})', '属于定义'),
        (r'([^\n]{2,8})指的是([^\n]{2,10})', '定义为'),
        (r'([^\n]{2,8})包括([^\n]{2,10})', '包括'),
        (r'([^\n]{2,8})包含([^\n]{2,10})', '包含'),
        (r'([^\n]{2,8})用于([^\n]{2,10})', '用于'),
        (r'([^\n]{2,8})表示([^\n]{2,10})', '表示'),
        (r'([^\n]{2,8})对应([^\n]{2,10})', '对应'),
        (r'([^\n]{2,8})称为([^\n]{2,10})', '称为'),
        (r'([^\n]{2,8})属于([^\n]{2,10})', '类别为'),
    ]

    sentences = text.replace("\n", " ").split("。")
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 8:
            continue

        for pat, rel in patterns:
            match = re.search(pat, sent)
            if match:
                h = match.group(1).strip()
                t = match.group(2).strip()
                if len(h) < 2 or len(t) < 2:
                    continue
                triples.append([h, rel, t])

        if "运行图" in sent and "横轴" in sent:
            triples.append(["列车运行图", "横轴表示", "时间"])
        if "运行图" in sent and "纵轴" in sent:
            triples.append(["列车运行图", "纵轴表示", "车站"])
        if "上下行" in sent:
            triples.append(["列车运行图", "包含", "上下行规则"])
        if "车次" in sent:
            triples.append(["列车运行图", "包含", "车次信息"])
        if "折线" in sent:
            triples.append(["运行线", "表示", "列车轨迹"])

    unique_triples = []
    seen = set()
    for tri in triples:
        key = str(tri)
        if key not in seen:
            seen.add(key)
            unique_triples.append(tri)

    return unique_triples

# ================================
# 加载图谱（合并 train picture1.md）
# ================================
def load_graph(embed_model):
    ps("加载知识图谱...")
    triples_data = json.load(open(TRIPLE_PATH, encoding="utf-8"))
    original_triples = [[x["h"], x["r"], x["t"]] for x in triples_data]
    G = pickle.load(open(GRAPH_PATH, "rb"))

    ps("从 train picture1.md 抽取知识图谱...")
    doc_triples = extract_triples_from_train_picture(DOCUMENT_PATH)
    po(f"从文档抽取三元组：{len(doc_triples)} 条")

    all_triples = original_triples + doc_triples

    for h, r, t in doc_triples:
        G.add_node(h)
        G.add_node(t)
        G.add_edge(h, t, relation=r)

    triple_texts = [f"{h} {r} {t}" for h, r, t in all_triples]
    triple_embs = embed_model.encode(triple_texts, show_progress_bar=True)
    po(f"最终图谱: {len(all_triples)} 三元组，{G.number_of_nodes()} 节点")

    return all_triples, G, np.array(triple_embs)

# ================================
# 图谱检索
# ================================
def graph_retrieval(query, embed_model, triples, G, triple_embs, top_k=5):
    qe = embed_model.encode(query)
    scores = np.dot(triple_embs, qe) / (np.linalg.norm(triple_embs, axis=1) * np.linalg.norm(qe) + 1e-8)
    selected = [triples[i] for i in scores.argsort()[-top_k:][::-1]]
    expanded = set()
    for h, r, t in selected:
        expanded.add((h, r, t))
        for node in [h, t]:
            if node in G:
                for nb in G.neighbors(node):
                    ed = G[node][nb]
                    if isinstance(ed, dict):
                        rel = ed.get("relation", "")
                        if rel:
                            expanded.add((node, rel, nb))
                        else:
                            for v in ed.values():
                                if isinstance(v, dict) and "relation" in v:
                                    expanded.add((node, v["relation"], nb))
    return list(expanded)[:20]
# ================================
# 工具函数
# ================================
def time_to_min(t):
    p = t.strip().split(":")
    try: return int(p[0])*60+int(p[1])+(int(p[2])/60 if len(p)>2 else 0)
    except: return 0.0

def min_to_hhmm(m):
    return f"{int(m)//60:02d}:{int(m)%60:02d}"

def is_drawing_request(q):
    for kw in ["什么是","如何","怎么","解释","说明","介绍","描述","定义"]:
        if kw in q: return False
    for kw in ["画","绘制","画出","作图","画图","绘图","生成.*图","制作.*图"]:
        if re.search(kw, q): return True
    return False

def extract_params(q):
    p = {"train_names": re.findall(r'([GDKCZT]\d{1,4})', q),
         "stations": list(dict.fromkeys(re.findall(r'([\u4e00-\u9fa5]{2,}(?:站|南|北|东|西|桥))', q)))[:8],
         "time_range": None, "style": "professional"}
    times = re.findall(r'(\d{1,2})[:：](\d{2})', q)
    if times: p["time_range"] = [f"{t[0]}:{t[1]}" for t in times]
    if "简约" in q or "简单" in q: p["style"] = "simple"
    elif "美观" in q or "漂亮" in q: p["style"] = "beautiful"
    return p

def enrich(schedules):
    colors = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22"]
    return [{"name": s.get("train",f"列车{i+1}"),
             "color": s.get("color", colors[i%len(colors)]),
             "stops": [{"station":st["station"],"time":st["time"],"km":st.get("km",0)}
                       for st in s.get("stops",[]) if st.get("station") and st.get("time")]}
            for i,s in enumerate(schedules)]

def fallback_schedule(params):
    trains = params["train_names"] or ["G1234","G5678"]
    stations = params["stations"] or ["北京南","济南西","南京南","上海虹桥"]
    colors = ["#e74c3c","#3498db","#2ecc71","#f39c12"]
    result = []
    for i,name in enumerate(trains[:3]):
        stops = [{"station":st,"time":f"{(8+i+j*2)%24:02d}:{j*10:02d}","km":j*300}
                 for j,st in enumerate(stations[:5])]
        result.append({"name":name,"color":colors[i%len(colors)],"stops":stops})
    return {"trains":result}


# ================================
# 时刻表构建（多源检索）
# ================================
def build_schedule(question, params, kb, triples, G, triple_embs, embed_model, tokenizer, model):
    # 1. 精确车次
    if params["train_names"]:
        exact = kb.by_train_name(params["train_names"])
        if exact:
            po(f"精确命中车次: {[s['train'] for s in exact]}")
            return {"trains": enrich(exact)}

    # 2. 途经站匹配
    if len(params["stations"]) >= 2:
        sm = kb.by_stations(params["stations"])
        if sm:
            po(f"站点匹配车次: {[s['train'] for s in sm[:4]]}")
            return {"trains": enrich(sm[:4])}

    # 3. 语义检索
    sem = kb.query_schedules(question, k=4)
    if sem:
        po(f"语义检索车次: {[s['train'] for s in sem]}")
        return {"trains": enrich(sem)}

    # 4. LLM生成（结合知识库上下文）
    pw("知识库无匹配，调用LLM生成...")
    graph_rel = graph_retrieval(question, embed_model, triples, G, triple_embs)
    graph_ctx = "\n".join(f"{h}-{r}->{t}" for h,r,t in graph_rel)
    route_ctx = "\n".join(
        f"{r.get('line','')}：{' → '.join(st['name'] for st in r.get('stations',[]))}"
        for r in kb.query_routes(question))
    rule_ctx = "\n".join(kb.query_rules(question))
    conds = []
    if params["train_names"]: conds.append(f"车次: {', '.join(params['train_names'])}")
    if params["stations"]: conds.append(f"途经: {', '.join(params['stations'])}")
    if params["time_range"]: conds.append(f"时间: {' 至 '.join(params['time_range'])}")

    prompt = f"""你是铁路时刻表专家，根据以下信息生成合理列车数据。

【条件】{chr(10).join(conds) if conds else '无特殊条件'}
【线路】{route_ctx}
【规则】{rule_ctx}
【图谱】{graph_ctx}
【原始问题】{question}

要求：时间合理（高铁北京南→上海虹桥约4.5小时），生成2-4趟列车，只输出JSON。

格式：
{{"trains":[{{"name":"G1","color":"#e74c3c","stops":[{{"station":"北京南","time":"09:00","km":0}},{{"station":"上海虹桥","time":"13:28","km":1318}}]}}]}}"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=800, temperature=0.1,
                             do_sample=False, eos_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    raw = re.sub(r"```json\s*|```\s*","",raw).strip()
    try:
        data = json.loads(raw[raw.index("{"):raw.rindex("}")+1])
        if data.get("trains"): return data
    except Exception as e: pw(f"LLM JSON解析失败: {e}")
    return fallback_schedule(params)


# ================================
# 绘图
# ================================
def draw_diagram(schedule, style="professional", title="列车运行图"):
    trains = schedule.get("trains",[])
    if not trains:
        fig,ax=plt.subplots(); ax.text(0.5,0.5,"无数据",ha="center",va="center"); return fig

    # 站点排序（按首次出现时间）
    st_first = {}
    for tr in trains:
        for stop in tr.get("stops",[]):
            s,t = stop["station"],time_to_min(stop["time"])
            if s not in st_first: st_first[s] = t
    sts = sorted(st_first, key=lambda x: st_first[x])
    sy = {s:i for i,s in enumerate(sts)}
    n = len(sts)

    all_t = [time_to_min(stop["time"]) for tr in trains for stop in tr.get("stops",[])]
    t0,t1 = max(0,min(all_t)-30), max(all_t)+30

    cfg = {"simple":{"bg":"#ffffff","grid":"#dddddd","txt":"#222222"},
           "beautiful":{"bg":"#1a1a2e","grid":"#16213e","txt":"#eeeeee"},
           "professional":{"bg":"#0d1117","grid":"#2d3748","txt":"#e2e8f0"}}
    c = cfg.get(style, cfg["professional"])
    bg,gc,tc = c["bg"],c["grid"],c["txt"]

    fig,ax = plt.subplots(figsize=(max(16,(t1-t0)/25), max(8,n*1.2)))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)

    for y in range(n): ax.axhline(y=y,color=gc,lw=0.8,alpha=0.5,zorder=1)
    t=((int(t0/30)+1)*30)
    while t<=t1: ax.axvline(x=t,color=gc,lw=0.5,ls="--",alpha=0.4,zorder=1); t+=30

    colors=["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c"]
    handles=[]

    for idx,tr in enumerate(trains):
        stops=[s for s in tr.get("stops",[]) if s["station"] in sy]
        if len(stops)<2: continue
        color=tr.get("color",colors[idx%len(colors)])
        name=tr.get("name",f"列车{idx+1}")
        xs=[time_to_min(s["time"]) for s in stops]
        ys=[sy[s["station"]] for s in stops]

        ax.plot(xs,ys,color=color,lw=2.5,zorder=3,solid_capstyle="round",alpha=0.9)
        ax.scatter(xs,ys,color=color,s=80,zorder=4,
                   edgecolors="white" if bg!="#ffffff" else "#555",lw=1.5,alpha=0.85)

        for x,y,s in zip(xs,ys,stops):
            ax.text(x, y+(-0.3 if y>0 else 0.3), s["time"],
                    color=color,fontsize=7.5,ha="center",va="center",fontweight="bold",zorder=5)

        mid=len(xs)//2
        ax.text(xs[mid]+4,ys[mid]-0.4,name,color=color,fontsize=9,fontweight="bold",zorder=5,
                bbox=dict(boxstyle="round,pad=0.2",facecolor=bg,edgecolor=color,lw=1,alpha=0.88))
        handles.append(mpatches.Patch(color=color,label=name))

    ax.set_yticks(list(sy.values()))
    ax.set_yticklabels(list(sy.keys()),fontsize=11,color=tc,fontweight="bold")
    ax.yaxis.set_tick_params(length=0,pad=8)

    xt,xl=[],[]
    t=((int(t0/30)+1)*30)
    while t<=t1: xt.append(t); xl.append(min_to_hhmm(t)); t+=30
    ax.set_xticks(xt); ax.set_xticklabels(xl,fontsize=9,color=tc,rotation=45,ha="right")
    ax.xaxis.set_tick_params(length=0,pad=5)

    ax.set_xlim(t0,t1); ax.set_ylim(-0.8,n-0.2); ax.invert_yaxis()
    for sp in ax.spines.values(): sp.set_edgecolor(gc); sp.set_linewidth(1)
    ax.set_title(title,fontsize=16,color=tc,pad=16,fontweight="bold")
    ax.set_xlabel("时间",fontsize=11,color=tc,labelpad=8)
    ax.set_ylabel("车站",fontsize=11,color=tc,labelpad=8)
    if handles:
        ax.legend(handles=handles,loc="upper right",framealpha=0.9,
                  facecolor=bg,edgecolor=gc,labelcolor=tc,fontsize=9)
    plt.tight_layout(pad=1.5)
    return fig


# ================================
# DeepSeek 评价
# ================================
def deepseek_evaluate(question, content, eval_type):
    ps("调用 DeepSeek 评价...")
    sys_prompts = {
        "qa": "你是AI问答质量评价专家，从准确性、完整性、专业性、可读性、实用性五维度评价。",
        "drawing": "你是铁路运行图评价专家，从合理性、完整性、专业性、可读性四维度评价。"
    }
    try:
        resp = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role":"system","content": sys_prompts.get(eval_type, sys_prompts["qa"])},
                {"role":"user","content":
                    f"【问题】\n{question}\n\n【内容】\n{content}\n\n"
                    f"输出：总体评分X/10，分项评价，优点，改进建议，综合结论"}
            ], temperature=0.3, max_tokens=800)
        text = resp.choices[0].message.content
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(EVAL_DIR, f"{eval_type}_{ts}.md")
        with open(save_path,"w",encoding="utf-8") as f:
            f.write(f"# 评价报告\n\n**问题**: {question}\n\n## 评价\n\n{text}\n\n## 内容\n\n{content}\n")
        return {"success":True,"evaluation":text,"save_path":save_path}
    except Exception as e:
        pe(f"DeepSeek失败: {e}")
        return {"success":False,"error":str(e),"evaluation":"评价服务暂时不可用"}


# ================================
# 问答流程
# ================================
def graphrag_qa(question, kb, triples, G, triple_embs, embed_model, tokenizer, model):
    graph_rel = graph_retrieval(question, embed_model, triples, G, triple_embs)
    graph_ctx = "\n".join(f"{h}-{r}->{t}" for h,r,t in graph_rel)
    sch_ctx = "\n".join(
        f"车次{s['train']}: " + " → ".join(f"{st['station']}({st['time']})" for st in s.get("stops",[]))
        for s in kb.query_schedules(question, k=3))
    route_ctx = "\n".join(
        f"{r.get('line','')}: " + " → ".join(st["name"] for st in r.get("stations",[]))
        for r in kb.query_routes(question))
    rule_ctx  = "\n".join(kb.query_rules(question))
    main_ctx  = "\n".join(kb.query_main(question))
    web_ctx = ""

    ctx_parts = []
    if sch_ctx.strip(): ctx_parts.append(f"【时刻表】\n{sch_ctx}")
    if route_ctx.strip(): ctx_parts.append(f"【线路】\n{route_ctx}")
    if rule_ctx.strip(): ctx_parts.append(f"【规则】\n{rule_ctx}")
    if main_ctx.strip(): ctx_parts.append(f"【知识库】\n{main_ctx}")
    if graph_ctx.strip(): ctx_parts.append(f"【知识图谱】\n{graph_ctx}")
    if web_ctx.strip(): ctx_parts.append(f"【网络参考（仅辅助）】\n{web_ctx}")
    ctx_block = "\n\n".join(ctx_parts) if ctx_parts else "（无额外上下文）"
    if not any([sch_ctx.strip(), route_ctx.strip(), rule_ctx.strip(), main_ctx.strip(), graph_ctx.strip()]):
        return "参考资料中未提供相关信息，无法回答该问题。\n\n【置信度】低置信度（无数据支撑）"
    prompt = f"""你是专业铁路领域知识助手。

【重要规则（必须严格遵守）】
1. 如果参考资料中没有相关信息，必须明确回答：
   “参考资料中未提供相关信息”，不得进行任何猜测或编造。
2. 严禁使用常识补全、经验推测或生成虚假数据。
3. 所有结论必须可以在参考资料中找到依据。
4. 如果判断为“未提供相关信息”，请只输出这一句话并立即结束，不得继续生成任何内容。
5. 如果输出包含“训练指令”“用户：”“#”等无关内容，判定为错误输出，必须停止生成。

【输出格式（强制）】

请严格按照以下结构回答：

【定义】
（1-2句话）

【作用】
（1-2句话）

生成文本依次编号。
禁止输出标题之外的内容。
禁止重复定义。

【回答风格】
- 简单问题：简洁回答
- 复杂问题：分点说明，逻辑清晰，避免重复

【去重要求】
- 禁止重复句子
- 禁止同义反复
- 每一点表达新的信息

【参考资料】
{ctx_block}

【问题】
{question}

【回答】
"""

    ps("模型推理...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            top_p=0.8,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
   # ✅ ① 截断异常生成（stop words）
    stop_words = ["# 训练指令", "用户：", "###", "```"]
    for sw in stop_words:
        if sw in raw:
            raw = raw.split(sw)[0]
    raw = raw.strip()

    # ✅ ② 清洗脏数据（比stop更宽）
    raw = clean_answer(raw)

    # ✅ ③ 去重（你原有的）
    answer = _deduplicate_answer(raw)

    # ✅ ④ 强制结构（非常关键）
    answer = enforce_format(answer)

    # ✅ ⑤ 未命中规则（必须在最后）
    if "未提供相关信息" in answer:
        answer = "参考资料中未提供相关信息。\n\n【置信度】低置信度（无数据支撑）"

    # ✅ ⑥ 置信度
    confidence = confidence_check(answer, ctx_block)
    answer += f"\n\n【置信度】{confidence}"
    return answer


def _deduplicate_answer(text: str) -> str:
    import re

    text = re.sub(r'```.*?```', '', text, flags=re.S)
    text = re.sub(r'`.*?`', '', text)

    sentences = re.split(r'[。！？\n]', text)

    seen = set()
    result = []

    for s in sentences:
        s = s.strip()
        if len(s) < 3:
            continue

        # ✅ 核心：前10字作为key（比你原来强很多）
        key = s[:10]

        if key in seen:
            continue
        seen.add(key)

        result.append(s)

    return "。\n".join(result[:5]) + "。"

def enforce_format(text):
    sections = ["【定义】", "【作用】", "【要素】"]
    result = []

    for sec in sections:
        if sec in text:
            part = text.split(sec)[1]
            part = part.split("【")[0]  # 截到下一个section
            result.append(sec + part.strip())

    return "\n".join(result)
def clean_answer(text):
    import re
    text = re.sub(r'#.*?#', '', text)
    text = re.sub(r'用户：.*', '', text)
    text = re.sub(r'训练指令.*', '', text)
    return text.strip()
# ================================
# 工具函数
# ================================
def confidence_check(answer, ctx_block):
    if "未提供相关信息" in answer or not answer.strip():
        return "低置信度（无数据支撑）"
    if len(ctx_block.strip()) < 50:
        return "中置信度（上下文较少）"
    return "高置信度"
# ================================
# 画图流程
# ================================
def graphrag_draw(question, kb, triples, G, triple_embs, embed_model, tokenizer, model):
    params = extract_params(question)
    if params["train_names"]: po(f"识别车次: {params['train_names']}")
    if params["stations"]: po(f"识别车站: {params['stations']}")

    ps("多源知识库检索...")
    schedule = build_schedule(question, params, kb, triples, G, triple_embs, embed_model, tokenizer, model)

    m = re.search(r'([\u4e00-\u9fa5A-Za-z0-9]+(?:线|路|至|—|-|到)+[\u4e00-\u9fa5A-Za-z0-9]*)', question)
    title = (m.group(0)+" 列车运行图") if m else "智能列车运行图"

    ps("绘图中...")
    fig = draw_diagram(schedule, style=params.get("style","professional"), title=title)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(DIAGRAM_DIR, f"diagram_{ts}.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    po(f"保存: {save_path}")

    print(f"\n{C.CYAN}{C.BOLD}━━━ 时刻表 ━━━{C.RESET}")
    for tr in schedule.get("trains",[]):
        print(f"\n{C.BOLD}{tr['name']}{C.RESET}")
        for st in tr.get("stops",[]): print(f"  {st['station']:10s}  {st['time']}")
    pdiv()
    return save_path, schedule


# ================================
# 主循环
# ================================
def main():
    banner()
    init_sample_knowledge_files()
    import_excel_schedules_to_kb(
    EXCEL_TIMETABLE_DIR, SCHEDULE_KB_DIR, TRIPLE_PATH, GRAPH_PATH
)

    tokenizer, model, embed_model = load_models()
    kb = MultiSourceKB(embed_model)
    kb.load_all(DOCUMENT_PATH)
    triples, G, triple_embs = load_graph(embed_model)

    print(f"\n{C.GREEN}{C.BOLD}✅ 系统就绪{C.RESET}")
    pdiv()
    print(f"{C.WHITE}知识库目录说明（把真实数据放入这些目录）：")
    print(f"  时刻表: {SCHEDULE_KB_DIR}/  (*.json / *.csv)  ← 最核心")
    print(f"  线路:   {ROUTE_KB_DIR}/     (*.json)")
    print(f"  规范:   {DRAWING_KB_DIR}/   (*.md / *.txt)")
    print(f"  规则:   {RULE_KB_DIR}/      (*.md / *.txt){C.RESET}")
    pdiv()
    print(f"{C.GRAY}示例: '画出G1到上海的运行图' / '什么是追踪间隔' / exit退出{C.RESET}\n")

    while True:
        pdiv()
        try: question = input(f"{C.BOLD}{C.GREEN}🚄 > {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt): print(f"\n{C.YELLOW}再见！{C.RESET}"); break
        if not question: continue
        if question.lower() in ("exit","quit","q","退出"): break
        if question.lower() == "clear": os.system("clear" if os.name=="posix" else "cls"); continue

        if is_drawing_request(question):
            print(f"{C.YELLOW}[🎨 画图模式]{C.RESET}")
            try:
                save_path, schedule = graphrag_draw(
                    question, kb, triples, G, triple_embs, embed_model, tokenizer, model)
                content = (f"生成{len(schedule.get('trains',[]))}趟列车运行图，路径:{save_path}\n" +
                    "\n".join(f"{t['name']}: "+
                        " → ".join(f"{s['station']}({s['time']})" for s in t.get("stops",[]))
                        for t in schedule.get("trains",[])))
                ev = deepseek_evaluate(question, content, "drawing")
                if ev["success"]: pev(ev["evaluation"]); po(f"评价: {ev['save_path']}")
                else: pw(f"评价失败: {ev.get('error')}")
            except Exception as e: pe(f"画图失败: {e}"); import traceback; traceback.print_exc()
        else:
            print(f"{C.YELLOW}[💬 问答模式]{C.RESET}")
            try:
                answer = graphrag_qa(question, kb, triples, G, triple_embs, embed_model, tokenizer, model)
                pa(answer)
                ev = deepseek_evaluate(question, answer, "qa")
                if ev["success"]: pev(ev["evaluation"]); po(f"评价: {ev['save_path']}")
                else: pw(f"评价失败: {ev.get('error')}")
            except Exception as e: pe(f"问答失败: {e}"); import traceback; traceback.print_exc()
        plt.close("all")

if __name__ == "__main__":
    try: import openai
    except ImportError: os.system("pip install openai -q")
    main()