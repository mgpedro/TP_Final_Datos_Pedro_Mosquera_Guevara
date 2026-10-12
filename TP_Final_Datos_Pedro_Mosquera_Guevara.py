
# %% Parámetros
import json
import os
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.sparse.csgraph import connected_components
from scipy.special import zeta
from tqdm.auto import tqdm

#### IMPORTANTE ####
## Para poder cargar el trabajo se realiza un paso previo con la base completa para llegar a "Works_Filtrados.jsonl" Solamente se seleccionan las características relevantes y se imponenen las 
## limitaciones de la muestra, tipo de trabajo (solo artículos) y fecha de publicaciones. 

ruta_dir   = Path(r"d:\escritorio\DATOS") 
ruta_in    = ruta_dir / "works_filtrados.jsonl"
ruta_cache = ruta_dir / "matrices"          
ruta_sal   = ruta_dir / "salidas_tp"       
COL_FIELD  = "field"

DISCIPLINAS = {   # valor de "field" -> (nombre en tablas y gráficos, nombre corto para subcaptions, slug)
    "Medicine":              ("Medicina", "Medicina", "medicine"),
    "Physics and Astronomy": ("Física", "Física", "physics_and_astronomy"),
    "Computer Science":      ("Ciencias de la computación", "Computación", "computer_science"),
}


MAX_AUTORES = None

# Ajuste de ley de potencias (trabajos por autor)
XMIN_RANGO  = (1, 20)         
RANGO_ALPHA = [1.01, 8]       
EXCLUIR_IDS = set()            
TOP_N       = 15               
ANIOS_VENTANA = 26           

# Métricas de red
N_FUENTES  = 100              
N_BARRIDOS = 5                 
N_CUNAS    = 100_000           
CHUNK      = 20_000           
SEMILLA    = 42

rng = np.random.default_rng(SEMILLA)
ruta_cache.mkdir(exist_ok=True)
ruta_sal.mkdir(exist_ok=True)


def num(x, d):
    """Número con d decimales y coma decimal para LaTeX."""
    return f"{x:.{d}f}".replace(".", "{,}")


def miles(n):
    return f"{n:,}".replace(",", ".")


# %% Funciones de cálculo
def alpha_mle_discreta(x, xmin):
    """MLE exacta de alpha para una ley de potencia discreta (control independiente de powerlaw)."""
    x = x[x >= xmin]
    s = np.log(x).sum()
    nll = lambda a: len(x) * np.log(zeta(a, xmin)) + a * s   # -log L
    return minimize_scalar(nll, bounds=tuple(RANGO_ALPHA), method="bounded").x


def colaboradores_por_autor(B):
    """z_i = coautores distintos = elementos no nulos de la fila i de B·Bᵀ, menos uno (el propio autor)."""
    BT = B.T.tocsr()
    z = np.empty(B.shape[0], dtype=np.int64)
    for ini in tqdm(range(0, B.shape[0], CHUNK), desc="  colaboradores", leave=False):
        z[ini:ini + CHUNK] = (B[ini:ini + CHUNK] @ BT).getnnz(axis=1) - 1
    return z


def bfs(indptr, indices, origen, n):
    """BFS sobre el grafo bipartito; devuelve distancias (-1 = inalcanzable)."""
    dist = np.full(n, -1, dtype=np.int32)
    dist[origen] = 0
    frontera = np.array([origen], dtype=np.int64)
    d = 0
    while frontera.size:
        ini = indptr[frontera]
        largo = indptr[frontera + 1] - ini
        total = int(largo.sum())
        if total == 0:
            break
        pos = np.repeat(ini - np.cumsum(largo) + largo, largo) + np.arange(total)
        vecinos = indices[pos]
        vecinos = np.unique(vecinos[dist[vecinos] < 0])
        d += 1
        dist[vecinos] = d
        frontera = vecinos
    return dist


def clustering_muestreado(B, z, n_cunas):
    """
    C = 3 x triángulos / tripletes conectados (ec. 2 de Newman), estimado muestreando tripletes:
    un autor v con probabilidad proporcional a z_v (z_v - 1) / 2, dos de sus coautores al azar,
    y se verifica si esos dos también son coautores.
    """
    BT = B.T.tocsr()
    w = z.astype(np.float64) * (z - 1) / 2
    v_muestra = rng.choice(len(z), size=n_cunas, p=w / w.sum())
    centros, veces = np.unique(v_muestra, return_counts=True)
    cerrados = 0
    for v, m in tqdm(zip(centros, veces), total=len(centros), desc="  clustering", leave=False):
        trabajos_v = B.indices[B.indptr[v]:B.indptr[v + 1]]
        colab = np.unique(np.concatenate([BT.indices[BT.indptr[t]:BT.indptr[t + 1]]
                                          for t in trabajos_v]))
        colab = colab[colab != v]
        u = rng.integers(len(colab), size=m)
        x = rng.integers(len(colab) - 1, size=m)
        x[x >= u] += 1                                   # dos coautores distintos
        for a, b in zip(colab[u], colab[x]):
            ta = B.indices[B.indptr[a]:B.indptr[a + 1]]
            tb = B.indices[B.indptr[b]:B.indptr[b + 1]]
            cerrados += np.intersect1d(ta, tb, assume_unique=True).size > 0
    C = cerrados / n_cunas
    return C, np.sqrt(C * (1 - C) / n_cunas)


# %% 1) Lectura única: matriz autor x trabajo por disciplina (con caché)
#    B[i, j] = 1 si el autor i firma el trabajo j. La caché se regenera sola si cambia el JSONL.
ruta_meta = ruta_cache / "meta.json"
meta_actual = {"ruta_in": str(ruta_in), "mtime": os.path.getmtime(ruta_in)}
cache_valida = (ruta_meta.exists()
                and json.loads(ruta_meta.read_text(encoding="utf-8")) == meta_actual
                and all((ruta_cache / f"B_{slug}.npz").exists() for *_, slug in DISCIPLINAS.values()))

if cache_valida:
    print("Matrices cargadas desde caché.")
else:
    idx_autor = defaultdict(dict)                # campo -> {id entero: índice de fila}
    filas_B = defaultdict(lambda: array("i"))
    cols_B = defaultdict(lambda: array("i"))
    n_works = defaultdict(int)
    sin_autores = Counter()

    with open(ruta_in, encoding="utf-8") as f:
        for linea in tqdm(f, unit=" obras", mininterval=1.0):
            if not linea.strip():
                continue
            o = json.loads(linea)
            campo, ids = o.get(COL_FIELD), o.get("author_ids")
            if campo not in DISCIPLINAS:
                continue
            a = set()
            for u in ids if isinstance(ids, list) else []:
                if isinstance(u, str):
                    try:
                        a.add(int(u.rsplit("/", 1)[-1][1:]))      # "…/A123" -> 123
                    except ValueError:
                        pass
            if not a:                                            # obras sin autores: se descartan
                sin_autores[campo] += 1
                continue
            j = n_works[campo]
            n_works[campo] += 1
            d = idx_autor[campo]
            for aid in a:
                filas_B[campo].append(d.setdefault(aid, len(d)))
                cols_B[campo].append(j)

    for campo, (*_, slug) in DISCIPLINAS.items():
        r = np.frombuffer(filas_B[campo], dtype=np.int32)
        c = np.frombuffer(cols_B[campo], dtype=np.int32)
        B = sparse.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)),
                              shape=(len(idx_autor[campo]), n_works[campo]))
        sparse.save_npz(ruta_cache / f"B_{slug}.npz", B)
        np.save(ruta_cache / f"autores_{slug}.npy",
                np.fromiter(idx_autor[campo].keys(), dtype=np.int64, count=len(idx_autor[campo])))
        print(f"{campo}: {n_works[campo]:,} obras, {len(idx_autor[campo]):,} autores, "
              f"{sin_autores[campo]:,} obras sin autores descartadas")
    ruta_meta.write_text(json.dumps(meta_actual), encoding="utf-8")
    del idx_autor, filas_B, cols_B


def cargar_matriz(slug):
    """Carga B y los IDs de autor, y aplica el filtro MAX_AUTORES."""
    B = sparse.load_npz(ruta_cache / f"B_{slug}.npz").tocsr()
    ids = np.load(ruta_cache / f"autores_{slug}.npy")
    if MAX_AUTORES is not None:
        B = B[:, B.getnnz(axis=0) <= MAX_AUTORES]        # excluye trabajos con muchos autores
        con_trabajos = B.getnnz(axis=1) > 0              # excluye autores que quedaron sin trabajos
        B, ids = B[con_trabajos], ids[con_trabajos]
    B.sort_indices()
    return B, ids


# %% 2) Trabajos por autor: ajuste de ley de potencias (Figura 1 y Tabla 2)
COMPARACIONES = {   # nombre -> (distribución 1, distribución 2, etiqueta en la tabla)
    "lognormal":             ("power_law", "lognormal", r"Potencia vs.\ lognormal"),
    "exponencial":           ("power_law", "exponential", r"Potencia vs.\ exponencial"),
    "potencia_truncada":     ("power_law", "truncated_power_law", r"Potencia vs.\ potencia truncada"),
    "truncada_vs_lognormal": ("truncated_power_law", "lognormal", r"Truncada vs.\ lognormal"),
}
excluir = {int(s[1:]) for s in EXCLUIR_IDS}

filas_pl = []
for campo, (nombre, corto, slug) in DISCIPLINAS.items():
    print(f"\n=== {nombre}: trabajos por autor ===")
    B, ids = cargar_matriz(slug)
    x_todos = B.getnnz(axis=1)                           # trabajos por autor

    # Diagnóstico: autores con más trabajos (revisar posibles errores de desambiguación)
    top = np.argsort(-x_todos)[:TOP_N]
    pd.DataFrame({
        "author_id": [f"A{i}" for i in ids[top]],
        "trabajos": x_todos[top],
        "trabajos_por_anio": (x_todos[top] / ANIOS_VENTANA).round(1),
        "url": [f"https://openalex.org/A{i}" for i in ids[top]],
    }).to_csv(ruta_sal / f"top_autores_{slug}.csv", index=False, encoding="utf-8-sig")

    x = x_todos[~np.isin(ids, list(excluir))] if excluir else x_todos
    fit = powerlaw.Fit(x, discrete=True, xmin=XMIN_RANGO,
                       parameter_ranges={"alpha": RANGO_ALPHA})
    pl, tpl, ln = fit.power_law, fit.truncated_power_law, fit.lognormal
    n_cola = int((x >= fit.xmin).sum())
    alpha_exacta = alpha_mle_discreta(x, fit.xmin)
    if abs(alpha_exacta - pl.alpha) > 1e-3:
        print(f"  AVISO: alpha de powerlaw ({pl.alpha:.4f}) difiere de la MLE exacta ({alpha_exacta:.4f})")

    fila = {
        "campo": campo, "nombre": nombre, "corto": corto, "slug": slug,
        "n_autores": len(x), "n_excluidos": len(x_todos) - len(x), "max_trabajos": int(x.max()),
        "alpha": pl.alpha, "se_alpha": pl.sigma, "alpha_mle_exacta": alpha_exacta,
        "xmin": fit.xmin, "n_cola": n_cola, "pct_cola": 100 * n_cola / len(x), "KS": pl.D,
        "tpl_alpha": tpl.alpha, "tpl_lambda": tpl.Lambda, "ln_mu": ln.mu, "ln_sigma": ln.sigma,
    }
    # R > 0 favorece a la primera distribución del par; R < 0, a la segunda
    for clave, (d1, d2, _) in COMPARACIONES.items():
        fila[f"R_{clave}"], fila[f"p_{clave}"] = fit.distribution_compare(d1, d2, normalized_ratio=True)
    filas_pl.append(fila)

    # Panel de la Figura 1: CCDF con los ajustes (solo la cola, x >= x_min)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fit.plot_ccdf(ax=ax, color="C0", linewidth=2,
                  label=f"datos (x ≥ {fit.xmin:g}; n = {miles(n_cola)})")
    pl.plot_ccdf(ax=ax, color="C1", linestyle="--", label=f"potencia (α = {pl.alpha:.2f})")
    tpl.plot_ccdf(ax=ax, color="C2", linestyle="-.", label="potencia truncada")
    ln.plot_ccdf(ax=ax, color="C3", linestyle=":", label="lognormal")
    ax.set_ylim(bottom=0.5 / n_cola)                     # la CCDF empírica no baja de 1/n
    ax.set_xlabel("trabajos por autor")
    ax.set_ylabel("P(X ≥ x)")
    ax.set_title(nombre)
    ax.legend(fontsize=9)
    fig.savefig(ruta_sal / f"powerlaw_trabajos_{slug}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

pl_res = pd.DataFrame(filas_pl)
pl_res.round(4).to_csv(ruta_sal / "powerlaw_trabajos_resultados.csv", index=False, encoding="utf-8-sig")


# %% 3) Métricas de red por disciplina (Tabla 1)
filas = []
for campo, (nombre, corto, slug) in DISCIPLINAS.items():
    print(f"\n=== {nombre}: métricas de red ===")
    B, _ = cargar_matriz(slug)
    NA, NP = B.shape                                     # autores, trabajos
    pares = B.nnz                                        # pares obra-autor

    # Colaboradores por autor y ajuste de la ec. 1 de Newman: P(z) ~ z^-tau exp(-z/z_c)
    z = colaboradores_por_autor(B)
    fit = powerlaw.Fit(z[z > 0], discrete=True, xmin=1, parameter_ranges={"alpha": [0.01, 8]})
    tau, lam = fit.truncated_power_law.alpha, fit.truncated_power_law.Lambda
    zc = 1 / lam if lam > 0 else np.inf
    if tau < 0.02:
        print("  AVISO: tau quedó en el límite inferior; la distribución es prácticamente exponencial.")

    # Componentes conexas (sobre el bipartito, contando solo autores)
    print("  componentes conexas...")
    G = sparse.bmat([[None, B], [B.T, None]], format="csr")
    G.sort_indices()
    _, etiquetas = connected_components(G, directed=False)
    tam = np.bincount(etiquetas[:NA])                    # autores por componente
    orden = np.argsort(tam)[::-1]
    gigante = int(tam[orden[0]])
    segunda = int(tam[orden[1]]) if len(tam) > 1 else 0
    en_gigante = np.flatnonzero(etiquetas[:NA] == orden[0])

    # Distancias: BFS desde autores al azar de la componente gigante
    n = G.shape[0]
    medias, excentricidades = [], []
    lejano, d_lejano = None, -1
    fuentes = rng.choice(en_gigante, size=min(N_FUENTES, len(en_gigante)), replace=False)
    for s in tqdm(fuentes, desc="  distancias", leave=False):
        d_aut = bfs(G.indptr, G.indices, s, n)[en_gigante] // 2     # distancia entre autores
        medias.append(d_aut.sum() / (len(en_gigante) - 1))
        excentricidades.append(d_aut.max())
        if d_aut.max() > d_lejano:
            d_lejano, lejano = d_aut.max(), en_gigante[d_aut.argmax()]
    for _ in range(N_BARRIDOS):                          # double sweep para acotar el diámetro
        d_aut = bfs(G.indptr, G.indices, lejano, n)[en_gigante] // 2
        excentricidades.append(d_aut.max())
        lejano = en_gigante[d_aut.argmax()]
    del G

    C, se_C = clustering_muestreado(B, z, N_CUNAS)

    filas.append({
        "disciplina": nombre,
        "total_trabajos": NP, "total_autores": NA,
        "promedio_works_por_autor": pares / NA, "promedio_autores_por_work": pares / NP,
        "colaboradores_por_autor": z.mean(),
        "colaboradores_por_autor_con_coautores": z[z > 0].mean(),
        "mediana_colaboradores": np.median(z), "max_colaboradores": int(z.max()),
        "cutoff_zc": zc, "exponente_tau": tau,
        "componente_gigante": gigante, "pct_gigante": 100 * gigante / NA,
        "segunda_componente": segunda,
        "distancia_media": np.mean(medias),
        "se_distancia_media": np.std(medias, ddof=1) / np.sqrt(len(medias)),
        "distancia_maxima": int(max(excentricidades)),   # cota inferior del diámetro
        "clustering_C": C, "se_clustering_C": se_C,
    })

res = pd.DataFrame(filas)
print(res.round(3).set_index("disciplina").T.to_string())
res.round(4).to_csv(ruta_sal / "metricas_por_disciplina.csv", index=False, encoding="utf-8-sig")


# %% 4) Salidas LaTeX
nota_filtro = ("" if MAX_AUTORES is None else
               f" Se excluyen los trabajos con más de {miles(MAX_AUTORES)} autores.")


def tabla_latex(caption, label, encabezados, cuerpo, nota):
    lineas = [r"\begin{table}[htbp]", r"  \centering", f"  \\caption{{{caption}}}",
              f"  \\label{{{label}}}", r"  \begin{tabular}{l" + "c" * len(encabezados) + "}",
              r"    \toprule", "     & " + " & ".join(encabezados) + r" \\", r"    \midrule"]
    for item in cuerpo:
        # líneas ya formateadas (\midrule, \multicolumn) vienen con sangría; las filas de datos no
        lineas.append(item if item.startswith("    ") else f"    {item} \\\\")
    lineas += [r"    \bottomrule", r"  \end{tabular}", r"  \par\smallskip",
               f"  {{\\footnotesize {nota}}}", r"\end{table}"]
    return "\n".join(lineas) + "\n"


# --- Tabla 1: resumen por campo ---
def fila_t1(etiqueta, formato):
    return f"{etiqueta} & " + " & ".join(formato(r) for r in res.itertuples())


cuerpo_t1 = [
    fila_t1("Cantidad de trabajos",            lambda r: f"{r.total_trabajos}"),
    fila_t1("Cantidad de autores",             lambda r: f"{r.total_autores}"),
    fila_t1("Promedio de trabajos por autor",  lambda r: num(r.promedio_works_por_autor, 3)),
    fila_t1("Promedio de autores por trabajo", lambda r: num(r.promedio_autores_por_work, 3)),
    fila_t1("Colaboradores por autor",         lambda r: num(r.colaboradores_por_autor, 3)),
    fila_t1(r"\quad Cutoff $z_c$",             lambda r: r"$\infty$" if np.isinf(r.cutoff_zc) else num(r.cutoff_zc, 1)),
    fila_t1(r"\quad Exponente $\tau$",         lambda r: num(r.exponente_tau, 2)),
    r"    \midrule",
    fila_t1("Componente gigante",              lambda r: f"{r.componente_gigante}"),
    fila_t1(r"\quad Como porcentaje",          lambda r: num(r.pct_gigante, 1) + r"\%"),
    fila_t1("Segunda componente",              lambda r: f"{r.segunda_componente}"),
    r"    \midrule",
    fila_t1("Distancia media",                 lambda r: f"{num(r.distancia_media, 2)} ({num(r.se_distancia_media, 2)})"),
    fila_t1("Distancia máxima",                lambda r: f"{r.distancia_maxima}"),
    fila_t1(r"Coeficiente de clustering $C$",  lambda r: f"{num(r.clustering_C, 3)} ({num(r.se_clustering_C, 3)})"),
]
nota_t1 = ("Entre paréntesis, errores estándar. La distancia media se estima a partir de "
           f"{N_FUENTES} autores de origen elegidos al azar en la componente gigante, y $C$ a partir "
           f"de {miles(N_CUNAS)} tripletes conectados muestreados al azar. La distancia máxima es una "
           r"cota inferior del diámetro. $\tau$ y $z_c$ corresponden al ajuste por máxima "
           r"verosimilitud de $P(z) \sim z^{-\tau} e^{-z/z_c}$ con $z \geq 1$." + nota_filtro)
(ruta_sal / "tabla_resumen.tex").write_text(
    tabla_latex("Resumen por campo", "tab:resumen", list(res["disciplina"]), cuerpo_t1, nota_t1),
    encoding="utf-8")


# --- Tabla 2: ajuste de la distribución de trabajos por autor ---
def fila_t2(etiqueta, formato):
    return f"{etiqueta} & " + " & ".join(formato(r) for r in pl_res.itertuples())


sin_corte = pl_res["tpl_lambda"] < 1e-6
excepciones = []
cuerpo_t2 = [
    fila_t2(r"$\hat\alpha$ (e.e.)", lambda r: f"{num(r.alpha, 3)} ({num(r.se_alpha, 3)})"),
    fila_t2(r"$x_{\min}$", lambda r: f"{r.xmin:g}"),
    fila_t2(r"Autores en la cola (\%)", lambda r: num(r.pct_cola, 1)),
    fila_t2(r"$1/\hat\lambda$ (potencia truncada)",
            lambda r: "--" if r.tpl_lambda < 1e-6 else f"{1 / r.tpl_lambda:.0f}"),
    r"    \midrule",
    r"    \multicolumn{" + str(len(pl_res) + 1) + r"}{l}{\textit{Cociente de log-verosimilitudes normalizado ($R$)}} \\",
]
for clave, (*_, etiqueta) in COMPARACIONES.items():
    celdas = []
    for r in pl_res.itertuples():
        R, p = getattr(r, f"R_{clave}"), getattr(r, f"p_{clave}")
        celdas.append(f"${num(R, 1)}$")
        if p >= 0.001:
            excepciones.append(f"{r.nombre}, {etiqueta.replace(chr(92) + ' ', ' ').lower()} ($p={num(p, 3)}$)")
    cuerpo_t2.append(f"{etiqueta} & " + " & ".join(celdas))

nota_t2 = r"$R>0$ favorece a la primera distribución del par y $R<0$ a la segunda. "
nota_t2 += ("Todos los tests con $p<0{,}001$" +
            (", salvo: " + "; ".join(excepciones) + "." if excepciones else "."))
if sin_corte.any():
    nota_t2 += (" En " + " y ".join(pl_res.loc[sin_corte, "nombre"]) +
                r", $\hat\lambda \approx 0$ (sin corte exponencial).")
nota_t2 += nota_filtro
(ruta_sal / "tabla_powerlaw_trabajos.tex").write_text(
    tabla_latex("Ajuste de la distribución de trabajos por autor", "tab:powerlaw_trabajos",
                list(pl_res["nombre"]), cuerpo_t2, nota_t2),
    encoding="utf-8")


# --- Figura 1: trabajos por autor ---
paneles = []
for i, r in enumerate(pl_res.itertuples()):
    paneles.append(
        "  \\begin{subfigure}{0.32\\textwidth}\n"
        f"    \\includegraphics[width=\\textwidth]{{powerlaw_trabajos_{r.slug}.png}}\n"
        f"    \\caption{{{r.corto} ($\\hat\\alpha = {num(r.alpha, 2)}$)}}\n"
        "  \\end{subfigure}")
partes = [f"{r.xmin:g} en {r.nombre}" for r in pl_res.itertuples()]
xmins = ", ".join(partes[:-1]) + " y " + partes[-1] if len(partes) > 1 else partes[0]
figura = (
    "\\begin{figure}[htbp]\n  \\centering\n" + "\n  \\hfill\n".join(paneles) + "\n"
    "  \\caption{Distribución de trabajos por autor, por disciplina (2000--2026).\n"
    "  Se grafica la función de distribución complementaria acumulada\n"
    f"  $P(X \\geq x)$ en escala log-log para $x \\geq x_{{\\min}}$ ($x_{{\\min}}$ = {xmins}).\n"
    "  La línea continua corresponde a los datos de OpenAlex; las líneas punteadas, a los\n"
    "  ajustes por máxima verosimilitud de una ley de potencia (exponente $\\hat\\alpha$), una\n"
    "  ley de potencia con corte exponencial y una distribución lognormal.\n"
    "  Ver Tabla~\\ref{tab:powerlaw_trabajos}." + nota_filtro + "}\n"
    "  \\label{fig:pl_trabajos}\n\\end{figure}\n")
(ruta_sal / "figura_trabajos.tex").write_text(figura, encoding="utf-8")

print(f"\nListo. Archivos para el informe en {ruta_sal}")
