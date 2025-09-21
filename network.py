import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib import font_manager as fm
import colorsys
import matplotlib.patches as mpatches

try:
    import community as community_louvain
except Exception:
    community_louvain = None


def _choose_font_family():
    candidates = ['Microsoft YaHei', 'DengXian', 'Noto Sans CJK SC', 'Noto Sans SC', 'Source Han Sans CN', 'PingFang SC', 'Hiragino Sans GB', 'WenQuanYi Micro Hei', 'Microsoft YaHei UI', 'SimHei', 'KaiTi', 'Kaiti SC', 'STKaiti', 'Arial Unicode MS', 'Segoe UI', 'DejaVu Sans']
    for name in candidates:
        try:
            _ = fm.findfont(mpl.font_manager.FontProperties(family=name), fallback_to_default=False)
            return name
        except Exception:
            continue
    return 'DejaVu Sans'


def _desaturate_hex(hex_color: str, target_s: float = 0.35, lightness_shift: float = 0.08, target_l: float = None) -> str:
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(0.0, min(1.0, target_s if target_s is not None else s))
    if target_l is not None:
        l = target_l
    l = max(0.0, min(1.0, l + lightness_shift))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return mcolors.to_hex((r2, g2, b2))


OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
TOL_12 = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377', '#BBBBBB', '#2288AA', '#66A61E', '#FFD92F', '#E78AC3', '#A6761D']
TOL_20 = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#661100', '#CC6677', '#882255', '#AA4499', '#DDDDDD', '#1F78B4', '#33A02C', '#FB9A99', '#E31A1C', '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A', '#B15928']


def _hsluv_palette(n: int, s: float = 60.0, l: float = 62.0):
    try:
        import hsluv as _hsluv
    except Exception:
        _hsluv = None
    if _hsluv is None:
        colors = []
        for i in range(max(1, n)):
            h = i / max(1, n)
            r, g, b = colorsys.hls_to_rgb(h, l/100.0, s/100.0)
            colors.append(mcolors.to_hex((r, g, b)))
        return colors
    return [_hsluv.hsluv_to_hex([i*360.0/max(1, n), s, l]) for i in range(n)]


def get_community_colors(n: int, theme: str = 'distinct', pastelize: float = 0.2, target_s: float = None, target_l: float = None):
    theme = (theme or 'auto').lower()
    if n <= 8:
        base = OKABE_ITO[:n]
    elif n <= 12:
        base = TOL_12[:n]
    elif n <= 20:
        base = TOL_20[:n]
    else:
        try:
            import distinctipy as _distinctipy
            cols = _distinctipy.get_colors(n, pastelize=pastelize)
            base = [_distinctipy.get_hex(c) for c in cols]
        except Exception:
            base = _hsluv_palette(n)
    if target_s is None and target_l is None:
        return base
    return [_desaturate_hex(c, target_s=target_s, lightness_shift=0.0, target_l=target_l) for c in base]


def draw_network(edge_csv: str, out_png: str, layout: str = 'auto', min_weight: int = 0, seed: int = 42):
    if community_louvain is None:
        raise RuntimeError('缺少依赖 python-louvain')

    FONT_FAMILY = _choose_font_family()
    mpl.rcParams['font.family'] = FONT_FAMILY
    mpl.rcParams['axes.unicode_minus'] = False

    df = pd.read_csv(edge_csv)
    df = df.copy()
    df['a'] = df[['source', 'target']].min(axis=1)
    df['b'] = df[['source', 'target']].max(axis=1)
    df_agg = (df.groupby(['a','b'], as_index=False)['weight'].sum().rename(columns={'a':'source','b':'target'}))
    if min_weight and min_weight>0:
        df_agg = df_agg[df_agg['weight']>=min_weight]

    G = nx.from_pandas_edgelist(df_agg, 'source','target', edge_attr='weight', create_using=nx.Graph())
    if G.number_of_nodes()==0:
        raise ValueError('图为空')

    partition = community_louvain.best_partition(G, weight='weight')

    if layout=='none':
        intra_mul, inter_mul, k_val, iter_val = 1.0,1.0,2.0,120
    elif layout=='mild':
        intra_mul, inter_mul, k_val, iter_val = 2.0,0.5,2.0,120
    elif layout=='strong':
        intra_mul, inter_mul, k_val, iter_val = 4.5,0.25,1.5,200
    else:
        intra_sum = sum(data.get('weight',1) for _,_,data in G.edges(data=True))
        inter_sum = sum(data.get('weight',1) for u,v,data in G.edges(data=True) if partition[u]!=partition[v])
        ratio = inter_sum/(intra_sum+1e-9)
        try:
            mod_score = community_louvain.modularity(partition, G, weight='weight')
        except Exception:
            mod_score = 0.0
        strength = min(1.0, max(0.0, 0.6*ratio + 0.4*max(0.0,(0.3-mod_score)/0.3)))
        intra_mul = 2.0 + strength*(4.0-2.0)
        inter_mul = 0.5 - strength*(0.5-0.25)
        k_val = 2.0 - strength*(2.0-1.5)
        iter_val = int(120 + strength*(220-120))

    for u,v,data in G.edges(data=True):
        w = data.get('weight',1)
        data['layout_weight'] = w*(intra_mul if partition[u]==partition[v] else inter_mul)

    num_communities = max(partition.values()) + 1
    colors = get_community_colors(num_communities, theme='distinct', pastelize=0.2, target_s=None, target_l=None)
    color_map = {node: colors[partition[node]] for node in G.nodes()}
    node_colors = [color_map[n] for n in G.nodes()]

    degrees_unweighted = dict(G.degree())
    size_min, size_max = 200, 3000
    gamma = 1.6
    min_d = min(degrees_unweighted.values()) if degrees_unweighted else 0
    max_d = max(degrees_unweighted.values()) if degrees_unweighted else 1
    if max_d == min_d:
        node_sizes = [(size_min+size_max)/2 for _ in G.nodes()]
    else:
        node_sizes = []
        for n in G.nodes():
            norm = (degrees_unweighted[n]-min_d)/(max_d-min_d)
            boosted = norm ** gamma
            node_sizes.append(size_min + (size_max-size_min)*boosted)

    pos = nx.spring_layout(G, k=k_val, iterations=iter_val, weight='layout_weight', seed=seed)

    plt.figure(figsize=(16,10))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.95, linewidths=1.0, edgecolors='white')
    max_weight = df_agg['weight'].max() if not df_agg.empty else 1
    edge_widths = [0.5 + 2*G[u][v]['weight']/max_weight for u,v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.35, edge_color='#9aa1a6')
    nx.draw_networkx_labels(G, pos, font_size=14, font_family=_choose_font_family())

    ax = plt.gca()
    legend_patches = [mpatches.Patch(color=colors[i], label=f'聚类 {i+1}') for i in range(num_communities)]
    ax.legend(handles=legend_patches, loc='lower right', frameon=False, fontsize=10, ncol=1)

    plt.title('语义网络图（Louvain社区检测）', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()

    info = {
        'num_communities': num_communities,
        'communities': [{
            'index': i+1,
            'color': colors[i],
            'nodes': [node for node in G.nodes() if partition[node]==i]
        } for i in range(num_communities)],
        'modularity': community_louvain.modularity(partition, G, weight='weight')
    }
    return out_png, info


