"""
SmartStorage 360 × NEXA · Simulador fusionado de almacenamiento solar y eficiencia energética

Dos tipos de usuario (se elige al empezar):
  · Autoconsumidor con placas (proyecto SmartStorage 360): regala excedentes y quiere almacenarlos.
  · Vivienda en eficiencia (aplicación NEXA): sin placas o con obra/presupuesto limitado.

Un único motor horario (24 h por mes y tipo de día), un catálogo unificado de medidas
(5 niveles de almacenamiento + kit solar, FV de tejado, aislamiento, VMC y ventanas),
un presupuesto que se recalcula en tiempo real y un entregable «Presupuesto personalizado».

Ejecutar:  python -m streamlit run smartstorage360_nexa.py
Requiere:  streamlit, numpy, pandas, plotly
"""
import datetime
import html as _html
import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 0. CONSTANTES Y PERFILES HORARIOS (hipótesis explícitas del modelo)
# ==============================================================================
DIAS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
HORAS = np.arange(24)
CP = 4.186 / 3600.0                    # kWh / (kg·K)  → Etérmica = m · cp · ΔT
IVA_ELEC = 1.21 * 1.0511               # IVA + impuesto eléctrico ≈ 1,271
IVA_GAS = 1.21
ETA_CALDERA = 0.90
FIJO_ELEC_ANUAL = 230.0                # € de término de potencia (para estimar kWh a partir de la factura)
FIJO_GAS_ANUAL = 99.0                  # € de término fijo del gas
GAS_TERMICO = 0.70                     # fracción del gas dedicada a calefacción (el resto ACS/cocina)
TOPE_EFICIENCIA = 0.50                 # máxima reducción de la demanda térmica por medidas de envolvente

SOL_SH = np.array([0.055, 0.065, 0.085, 0.095, 0.108, 0.118, 0.125, 0.115, 0.097, 0.078, 0.055, 0.049])
SOL_SH = SOL_SH / SOL_SH.sum()
LONG_DIA = [9.9, 10.8, 11.9, 13.1, 14.1, 14.6, 14.3, 13.4, 12.3, 11.2, 10.2, 9.7]
NUBLADOS = [0.40, 0.35, 0.35, 0.30, 0.25, 0.10, 0.05, 0.08, 0.15, 0.30, 0.38, 0.42]
FACTOR_NUBLADO = 0.30

CARGA = np.array([1.15, 1.10, 1.00, 0.95, 0.90, 0.95, 1.05, 1.05, 0.90, 0.95, 1.05, 1.15])
CARGA_NORM = float((CARGA * DIAS).sum())
CLIMA_SH = np.array([0.20, 0.15, 0.07, 0.02, 0.02, 0.07, 0.17, 0.17, 0.04, 0.02, 0.07, 0.20])
CLIMA_SH = CLIMA_SH / CLIMA_SH.sum()
T_RED = [10, 11, 13, 15, 17, 19, 21, 21, 19, 16, 13, 10]
T_ACS, T_TANQUE = 55.0, 65.0
LITROS_PERSONA = 40.0

F_APP = 0.85
F_EV_DENTRO, F_EV_FUERA = 0.80, 0.20
P_TIMER, P_RES, P_CLIMA = 1.5, 3.0, 2.5
F2C, F3C, PEN_CLIMA = 0.30, 0.60, 1.08
ETA_BAT, C_RATE = 0.95, 0.5

OBRA_BALCON = "Solo balcón / sin obra exterior"
OBRA_TEJADO = "Acceso a tejado / obra permitida"
TIPOS = ["Piso en Comunidad", "Unifamiliar / Chalet / Adosado", "Piso de Alquiler"]
RENDS = {"Andalucía": 1550, "Madrid": 1450, "Comunidad Valenciana": 1450, "Cataluña": 1350,
         "Otras CCAA (Sur)": 1500, "Otras CCAA (Norte)": 1150}


def _v(d):
    v = np.zeros(24)
    for h, w in d.items():
        v[h] = w
    return v / v.sum()


def _norm(v):
    v = np.asarray(v, float)
    return v / v.sum()


BASE_P = {True: _norm([2, 2, 2, 2, 2, 2.5, 4, 6, 4, 2.5, 2.5, 2.5, 3, 3, 2.5, 2.5, 2.5, 3, 4.5, 7, 8, 8, 6, 3.5]),
          False: _norm([2, 2, 2, 2, 2, 2.5, 3.5, 5, 5, 5, 5, 5.5, 6, 5.5, 5, 5, 5, 5, 5.5, 6.5, 7, 7, 5.5, 3])}
APP_P = {True: _v({19: 1, 20: 1, 21: 1, 22: 1}),
         False: _v({10: 1, 11: 1, 12: 1, 13: 1, 19: 1, 20: 1, 21: 1, 22: 1})}
ACS_P = {True: _v({7: .15, 8: .15, 21: .2, 22: .3, 23: .2}),
         False: _v({7: .15, 8: .10, 14: .15, 21: .25, 22: .25, 23: .10})}
CLIMA_P = {True: _v({7: .10, 8: .05, 19: .15, 20: .20, 21: .20, 22: .15, 23: .10, 0: .05}),
           False: _v({8: .05, 9: .05, 10: .05, 11: .10, 12: .10, 13: .10, 14: .10, 15: .10, 16: .05,
                      19: .10, 20: .10, 21: .10})}
EV_P = _v({22: 1, 23: 1, 0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
VENTANA = _v({11: 1, 12: 1, 13: 1, 14: 1, 15: 1})
HORAS_TIMER = (12, 13, 14)


def _forma_solar(m):
    larga = LONG_DIA[m]
    x = (HORAS + 0.5 - (12 - larga / 2)) / larga
    s = np.where((x > 0) & (x < 1), np.sin(np.pi * x), 0.0)
    return s / s.sum()


FORMA = [_forma_solar(m) for m in range(12)]


# ==============================================================================
# 1. MODELO DE DATOS (vivienda + escenario de medidas)
# ==============================================================================
@dataclass(frozen=True)
class Casa:
    # Generación existente
    pv_existente: bool = True
    kwp: float = 4.0
    rend: float = 1500.0           # kWh/kWp·año → 4 kWp × 1500 = 6.000 kWh/año (vivienda tipo del PDF)
    # Consumo eléctrico
    consumo_base: float = 2400.0   # kWh/año sin ACS, climatización, VE ni electrodomésticos programables
    app_kwh: float = 600.0
    personas: int = 3
    litros: float = 150.0
    termo: bool = True
    clima_elec: bool = False
    clima_kwh: float = 900.0
    gas: bool = False
    gas_kwh: float = 0.0
    modulo_gas: bool = False
    ev: bool = False
    ev_kwh: float = 1600.0
    fuera: bool = True
    # Vivienda y obra (NEXA)
    tipo: str = TIPOS[0]
    m2: float = 85.0
    obra: str = OBRA_BALCON
    presupuesto: float = 4000.0
    ded_pct: float = 0.0           # deducción fiscal / subvención sobre la inversión (0 por defecto)
    # Tarifa
    p_kwh: float = 0.142           # €/kWh sin impuestos (×1,271 ≈ 0,18 €/kWh, como en el PDF)
    p_exc: float = 0.07
    p_gas: float = 0.088
    cuota_vb: float = 4.0
    # Costes de las medidas
    c_n2: float = 85.0
    c_n3: float = 690.0
    c_modulo: float = 450.0
    bat_fijo: float = 100.0        # 5 kWh → 2.650 €; 10 kWh → 5.200 € (como en el PDF)
    bat_kwh_precio: float = 510.0
    c_kit: float = 550.0           # NEXA
    kit_kwp: float = 0.8
    kit_factor: float = 0.80       # rendimiento de un kit vertical de balcón respecto a una instalación óptima
    kwp_nuevo: float = 3.0
    coste_kwp: float = 1267.0      # 3.800 € / 3 kWp (NEXA)
    c_aisl_m2: float = 12.0
    c_aisl_min: float = 600.0
    c_vmc: float = 1100.0
    c_vent_m2: float = 35.0
    c_vent_min: float = 2500.0
    red_aisl: float = 0.22         # reducción de demanda térmica (valores de NEXA)
    red_vmc: float = 0.12
    red_vent: float = 0.25

    @property
    def acs_modulo(self):
        return self.gas and self.modulo_gas and not self.termo

    @property
    def sumidero_acs(self):
        return self.termo or self.acs_modulo

    @property
    def factible_hems(self):
        return self.sumidero_acs or self.clima_elec

    @property
    def techo_ok(self):
        return self.obra == OBRA_TEJADO and self.tipo != TIPOS[2]

    @property
    def envolvente_ok(self):
        return self.obra == OBRA_TEJADO and self.tipo != TIPOS[2]

    @property
    def demanda_termica(self):
        return (self.gas_kwh * GAS_TERMICO if self.gas else 0.0) + (self.clima_kwh if self.clima_elec else 0.0)


@dataclass(frozen=True)
class Esc:
    n1: bool = False
    n2: bool = False
    n3: bool = False
    n4: bool = False
    bat: float = 0.0
    solar: int = 0                 # 0 = lo existente (o nada), 1 = kit de balcón, 2 = FV de tejado
    aisl: bool = False
    vmc: bool = False
    vent: bool = False

    def etiqueta(self):
        t = []
        if self.solar == 1:
            t.append("Kit balcón")
        if self.solar == 2:
            t.append("FV tejado")
        t += [n for f, n in ((self.aisl, "Aislamiento"), (self.vmc, "VMC"), (self.vent, "Ventanas")) if f]
        t += [f"N{i}" for i, f in enumerate((self.n1, self.n2, self.n3, self.n4), 1) if f]
        if self.bat > 0:
            t.append(f"N5 ({self.bat:g} kWh)")
        return " + ".join(t) if t else "Situación actual"


def prod_esc(casa: Casa, esc: Esc):
    if esc.solar == 1:
        return casa.kit_kwp * casa.rend * casa.kit_factor
    if esc.solar == 2:
        return casa.kwp_nuevo * casa.rend
    return casa.kwp * casa.rend if casa.pv_existente else 0.0


def hay_pv(casa: Casa, esc: Esc):
    return casa.pv_existente or esc.solar > 0


def fac_term(casa: Casa, esc: Esc):
    r = 1 - (1 - casa.red_aisl * esc.aisl) * (1 - casa.red_vmc * esc.vmc) * (1 - casa.red_vent * esc.vent)
    return 1 - min(TOPE_EFICIENCIA, r)


# ==============================================================================
# 2. MOTOR HORARIO: balance de 24 h por mes y tipo de día
# ==============================================================================
def dia(casa: Casa, esc: Esc, m: int, soleado: bool):
    cf = NUBLADOS[m]
    e_dia = prod_esc(casa, esc) * SOL_SH[m] / DIAS[m]
    k = e_dia / ((1 - cf) + FACTOR_NUBLADO * cf)
    S = k * (1.0 if soleado else FACTOR_NUBLADO) * FORMA[m]

    f = casa.fuera
    base_d = casa.consumo_base * CARGA[m] / CARGA_NORM
    app_d = casa.app_kwh / 365.0
    ev_d = casa.ev_kwh / 365.0 if casa.ev else 0.0
    acs_d = casa.personas * LITROS_PERSONA * CP * (T_ACS - T_RED[m]) if casa.sumidero_acs else 0.0
    acs_elec_d = acs_d if casa.termo else 0.0
    cl_d = casa.clima_kwh * CLIMA_SH[m] / DIAS[m] * fac_term(casa, esc) if casa.clima_elec else 0.0

    base = base_d * BASE_P[f]
    app0, ev0 = app_d * APP_P[f], ev_d * EV_P
    acs0, cl0 = acs_elec_d * ACS_P[f], cl_d * CLIMA_P[f]

    storage = hay_pv(casa, esc)
    if esc.n1 and storage:
        app = app0 * (1 - F_APP) + app_d * F_APP * VENTANA
        fev = F_EV_FUERA if f else F_EV_DENTRO
        ev = ev0 * (1 - fev) + ev_d * fev * VENTANA
    else:
        app, ev = app0, ev0
    fijo = base + app + ev

    draw_acs, draw_cl = np.zeros(24), np.zeros(24)
    cap_acs = min(acs_d, casa.litros * CP * (T_TANQUE - T_RED[m])) if casa.sumidero_acs else 0.0
    if storage and esc.n3 and casa.factible_hems:
        surplus = np.maximum(0.0, S - (fijo + acs0 + cl0))
        left = cap_acs
        for h in range(24):
            d = min(P_RES, surplus[h], left)
            draw_acs[h], left, surplus[h] = d, left - d, surplus[h] - d
        left = F3C * cl_d * PEN_CLIMA if casa.clima_elec else 0.0
        for h in range(24):
            d = min(P_CLIMA, surplus[h], left)
            draw_cl[h], left = d, left - d
    elif storage and esc.n2 and casa.factible_hems:
        left = cap_acs
        for h in HORAS_TIMER:
            d = min(P_TIMER, left)
            draw_acs[h], left = d, left - d
        if casa.clima_elec:
            draw_cl = F2C * cl_d * PEN_CLIMA * VENTANA

    x_acs, x_cl = draw_acs.sum(), draw_cl.sum()
    acs_new = acs0 * (1 - x_acs / acs_d) if (acs_elec_d > 0 and acs_d > 0) else acs0
    cl_new = cl0 * (1 - (x_cl / PEN_CLIMA) / cl_d) if cl_d > 0 else cl0
    carga = fijo + acs_new + cl_new + draw_acs + draw_cl

    cap = esc.bat if storage else 0.0
    red, exc, dis = np.zeros(24), np.zeros(24), np.zeros(24)
    soc = 0.0
    for p in range(2):
        for h in range(24):
            net = S[h] - carga[h]
            if net >= 0:
                ch = min(net, cap * C_RATE, (cap - soc) / ETA_BAT) if cap > 0 else 0.0
                soc += ch * ETA_BAT
                r, e, d = 0.0, net - ch, 0.0
            else:
                need = -net
                d = min(need, cap * C_RATE, soc * ETA_BAT) if cap > 0 else 0.0
                soc -= d / ETA_BAT
                r, e = need - d, 0.0
            if p == 1:
                red[h], exc[h], dis[h] = r, e, d
    return dict(S=S, carga=carga, red=red, exc=exc, dis=dis, base=base, app=app, ev=ev,
                acs=acs_new + draw_acs, clima=cl_new + draw_cl, x_acs=x_acs, acs_d=acs_d,
                gas_th=x_acs if casa.acs_modulo else 0.0)


@lru_cache(maxsize=None)
def anio(casa: Casa, esc: Esc):
    z = lambda: np.zeros(12)
    r = dict(red=z(), exc=z(), prod=z(), dis=z(), acs_sol=z(), acs_tot=z(), gas_th=z())
    for m in range(12):
        for soleado, w in ((True, (1 - NUBLADOS[m]) * DIAS[m]), (False, NUBLADOS[m] * DIAS[m])):
            d = dia(casa, esc, m, soleado)
            r["red"][m] += w * d["red"].sum()
            r["exc"][m] += w * d["exc"].sum()
            r["prod"][m] += w * d["S"].sum()
            r["dis"][m] += w * d["dis"].sum()
            r["acs_sol"][m] += w * d["x_acs"]
            r["acs_tot"][m] += w * d["acs_d"]
            r["gas_th"][m] += w * d["gas_th"]
    return r


# ==============================================================================
# 3. ECONOMÍA: compensación simplificada mensual + batería virtual (monedero en €)
# ==============================================================================
def coste_medida(casa: Casa, clave: str, esc: Esc = Esc()):
    """Coste bruto de una medida concreta (clave de MEDIDAS)."""
    if clave == "solar1":
        return casa.c_kit
    if clave == "solar2":
        return casa.coste_kwp * casa.kwp_nuevo
    if clave == "aisl":
        return max(casa.c_aisl_min, casa.c_aisl_m2 * casa.m2)
    if clave == "vmc":
        return casa.c_vmc
    if clave == "vent":
        return max(casa.c_vent_min, casa.c_vent_m2 * casa.m2)
    if clave == "n1" or clave == "n4":
        return 0.0
    if clave == "n2":
        return casa.c_n2 + (casa.c_modulo if casa.acs_modulo else 0.0)
    if clave == "n3":
        return casa.c_n3 + (casa.c_modulo if casa.acs_modulo else 0.0)
    if clave == "bat":
        return casa.bat_fijo + casa.bat_kwh_precio * esc.bat
    raise KeyError(clave)


def claves_de(esc: Esc, casa: Casa):
    """Claves de las medidas activas en un escenario (sin efecto si faltan sus requisitos)."""
    k = []
    if esc.solar == 1:
        k.append("solar1")
    if esc.solar == 2:
        k.append("solar2")
    k += [c for c, f in (("aisl", esc.aisl), ("vmc", esc.vmc), ("vent", esc.vent)) if f]
    if hay_pv(casa, esc):
        if esc.n1:
            k.append("n1")
        if casa.factible_hems and esc.n2:
            k.append("n2")
        if casa.factible_hems and esc.n3:
            k.append("n3")
        if esc.n4:
            k.append("n4")
        if esc.bat > 0:
            k.append("bat")
    return k


def inversion(casa: Casa, esc: Esc):
    return sum(coste_medida(casa, c, esc) for c in claves_de(esc, casa))


@lru_cache(maxsize=None)
def metricas(casa: Casa, esc: Esc):
    a = anio(casa, esc)
    pe = 0.0 if esc.solar == 1 else casa.p_exc          # el kit de balcón no compensa excedentes
    wallet = 0.0
    left_y = use_y = 0.0
    cost_m = np.zeros(12)
    n4 = esc.n4 and hay_pv(casa, esc)
    for _ in range(2):                                   # 2 años: el segundo es el régimen estacionario
        left_y = use_y = 0.0
        for m in range(12):
            ec = a["red"][m] * casa.p_kwh
            comp = min(a["exc"][m] * pe, ec)
            left = a["exc"][m] * pe - comp
            rem = ec - comp
            use = 0.0
            if n4:
                use = min(wallet, rem)
                wallet += left - use
            rem -= use
            cost_m[m] = rem
            left_y += left
            use_y += use
    elec = cost_m.sum() * IVA_ELEC + (casa.cuota_vb * 12 if n4 else 0.0)
    gas_ahorro = a["gas_th"].sum() / ETA_CALDERA * casa.p_gas * IVA_GAS
    if casa.gas:
        gas_ahorro += casa.gas_kwh * GAS_TERMICO * (1 - fac_term(casa, esc)) * casa.p_gas * IVA_GAS
    prod = a["prod"].sum()
    if pe > 0:
        regalada = max(0.0, left_y - use_y) / pe
        rescatada = use_y / pe
    else:
        regalada, rescatada = a["exc"].sum(), 0.0
    auto = prod - a["exc"].sum()
    inv = inversion(casa, esc)
    return dict(
        inversion=inv, inversion_neta=inv * (1 - casa.ded_pct / 100.0), coste=elec - gas_ahorro,
        prod=prod, auto=auto, auto_pct=(auto / prod * 100 if prod > 0 else 0.0), vertido=a["exc"].sum(),
        regalada=regalada, rescatada_vb=rescatada, red=a["red"].sum(), bateria=a["dis"].sum(),
        acs_sol=a["acs_sol"].sum(),
    )


def evaluar(casa: Casa, esc: Esc):
    base = metricas(casa, Esc())
    m = dict(metricas(casa, esc))
    m["ahorro"] = base["coste"] - m["coste"]
    inv = m["inversion_neta"]
    m["payback"] = inv / m["ahorro"] if m["ahorro"] > 0 else (0.0 if inv == 0 else float("inf"))
    return m


# ==============================================================================
# 4. OPTIMIZADOR: combinaciones de las 5 niveles + medidas NEXA
# ==============================================================================
@lru_cache(maxsize=16)
def candidatos(casa: Casa, baterias=(0.0, 5.0, 10.0)):
    solares = [0] if casa.pv_existente else [0, 1] + ([2] if casa.techo_ok else [])
    if casa.envolvente_ok and casa.demanda_termica > 0:
        efs = list(itertools.product((False, True), repeat=3))
    else:
        efs = [(False, False, False)]
    out = []
    for s in solares:
        pv = casa.pv_existente or s > 0
        if not pv:
            stor = [(False, False, False, False, 0.0)]
        elif s == 1:
            stor = [(False, False, False, False, 0.0), (True, False, False, False, 0.0)]
        else:
            stor = []
            for n1, n2, n3, n4 in itertools.product((False, True), repeat=4):
                if (n2 or n3) and not casa.factible_hems:
                    continue
                if n2 and n3:
                    continue
                for b in baterias:
                    stor.append((n1, n2, n3, n4, b))
        for st_ in stor:
            for ef in efs:
                esc = Esc(*st_, solar=s, aisl=ef[0], vmc=ef[1], vent=ef[2])
                if esc == Esc():
                    continue
                m = evaluar(casa, esc)
                m["esc"] = esc
                out.append(m)
    return out


def convencional(casa: Casa):
    """Alternativa convencional: FV (existente o de tejado) + batería de 10 kWh sin HEMS ni gestión."""
    if casa.pv_existente:
        esc = Esc(bat=10.0)
    elif casa.techo_ok:
        esc = Esc(bat=10.0, solar=2)
    else:
        return None, None
    return esc, evaluar(casa, esc)


def estrategias(casa: Casa, presupuesto: float, max_pb: float, obj_auto: float):
    cands = candidatos(casa)
    dentro = [c for c in cands if c["inversion"] <= presupuesto and c["ahorro"] > 0]
    res = dict(mayor=None, rapido=None, equilibrada=None, objetivo=None, n_candidatos=len(cands), n_dentro=len(dentro))
    if dentro:
        mayor = max(dentro, key=lambda c: (c["ahorro"], -c["inversion"]))
        res["mayor"] = mayor
        umbral = max(60.0, 0.25 * mayor["ahorro"])
        rapidos = [c for c in dentro if c["inversion"] > 0 and c["ahorro"] >= umbral] or dentro
        res["rapido"] = min(rapidos, key=lambda c: (c["payback"], -c["ahorro"]))
        eq = [c for c in dentro if c["payback"] <= max_pb]
        res["equilibrada"] = max(eq, key=lambda c: (c["ahorro"], -c["inversion"])) if eq else None
    ok = [c for c in cands if c["prod"] > 0 and c["auto_pct"] >= obj_auto and c["regalada"] <= 0.02 * c["prod"]
          and c["ahorro"] > 0]
    res["objetivo"] = min(ok, key=lambda c: (c["inversion"], -c["ahorro"])) if ok else None
    return res


def perfil(casa: Casa):
    if not casa.pv_existente:
        return 0
    if not casa.fuera:
        return 3
    return 1 if (casa.termo or casa.clima_elec) else 2


# ==============================================================================
# 5. CATÁLOGO UNIFICADO DE MEDIDAS (PDF + NEXA) Y DESGLOSE DEL PLAN
# ==============================================================================
def catalogo(casa: Casa, esc: Esc):
    m2 = int(casa.m2)
    return {
        "solar1": ("Kit solar de balcón Plug&Play (láminas flexibles Full Black, 800 W)", "NEXA",
                   "Colgar en la barandilla con bridas de alta resistencia y enchufar al enchufe de casa.",
                   "Certificado CE + microinversor anti-isla certificado REBT. No requiere registro en la junta si es "
                   "privativo. El excedente no se compensa."),
        "solar2": (f"Instalación fotovoltaica de autoconsumo en tejado ({casa.kwp_nuevo:g} kWp)", "NEXA",
                   "Estructura coplanar en el tejado con cableado directo al cuadro de protección general.",
                   "CIE (Boletín Eléctrico) + licencia municipal / comunicación previa + registro en Industria de la CCAA."),
        "aisl": (f"Aislamiento de paredes por insuflado ({m2} m²)", "NEXA",
                 "Perforaciones pequeñas en tabiquería o cámara de aire e inyección de celulosa o lana mineral.",
                 "Declaración responsable de obra menor / certificado de eficiencia energética antes y después."),
        "vmc": ("Ventilación mecánica descentralizada con recuperador de calor cerámico", "NEXA",
                "Taladro de 160 mm a fachada exterior en salón/dormitorio y conexión a 230 V.",
                "Marcado CE. Cumplimiento del CTE DB-HS3."),
        "vent": ("Sustitución de ventanas (vidrio bajo emisivo)", "NEXA",
                 "Sustitución de carpinterías con cinta de estanqueidad perimetral.",
                 "Marcado CE de carpinterías y vidrios + certificado de eficiencia energética."),
        "n1": ("Nivel 1 · Cambio de hábitos (desplazamiento de cargas)", "SmartStorage 360",
               "Programación diferida y escalonada de lavadora, lavavajillas y secadora entre las 11:00 y las 16:00 h "
               "y recarga del VE con sol.", "Ninguno."),
        "n2": ("Nivel 2 · Inercia térmica (temporizador del termo y curva de climatización)", "SmartStorage 360",
               "Temporizador digital Wi-Fi de carril DIN (16 A) para el termo eléctrico y sobrecalentamiento/"
               "sobreenfriamiento de +1,5 °C a mediodía." + (" Incluye termo eléctrico precalentador en serie con la caldera."
                                                             if casa.acs_modulo else ""),
               "Instalación por electricista autorizado en el cuadro; sin trámites."),
        "n3": ("Nivel 3 · Domótica solar HEMS (desviador dinámico + SG Ready)", "SmartStorage 360",
               "Desviador de excedentes con pinza amperimétrica (modulación 0–3.000 W al termo) e interfaz SG Ready con "
               "la aerotermia." + (" Incluye termo eléctrico precalentador en serie con la caldera." if casa.acs_modulo else ""),
               "Instalación por empresa autorizada; comprobar compatibilidad con el inversor."),
        "n4": ("Nivel 4 · Batería virtual (monedero solar)", "SmartStorage 360",
               f"Alta en una tarifa solar con batería virtual (cuota ≈ {casa.cuota_vb:g} €/mes): el excedente que supera "
               "el tope mensual se acumula en euros para descontarlo después.",
               "Cambio de contrato con la comercializadora; sin obra."),
        "bat": (f"Nivel 5 · Batería física LFP de {esc.bat:g} kWh (con BMS y mano de obra)", "SmartStorage 360",
                "Almacenamiento modular dimensionado a medida para el consumo nocturno que no se puede desplazar.",
                "Instalación por empresa autorizada; actualizar el certificado de instalación y la comunicación a Industria."),
    }


ORDEN = ["solar1", "solar2", "aisl", "vmc", "vent", "n1", "n2", "n3", "n4", "bat"]


def _esc_de(claves, esc_final: Esc):
    return Esc(n1="n1" in claves, n2="n2" in claves, n3="n3" in claves, n4="n4" in claves,
               bat=esc_final.bat if "bat" in claves else 0.0,
               solar=esc_final.solar if ("solar1" in claves or "solar2" in claves) else 0,
               aisl="aisl" in claves, vmc="vmc" in claves, vent="vent" in claves)


def desglose(casa: Casa, esc: Esc):
    """Lista de medidas del plan, de menor a mayor inversión, con ahorro marginal acumulativo."""
    claves = claves_de(esc, casa)
    cat = catalogo(casa, esc)
    grupo0 = sorted([c for c in claves if c in ("solar1", "solar2", "aisl", "vmc", "vent")],
                    key=lambda c: (coste_medida(casa, c, esc), ORDEN.index(c)))
    grupo1 = sorted([c for c in claves if c in ("n1", "n2", "n3", "n4", "bat")],
                    key=lambda c: (coste_medida(casa, c, esc), ORDEN.index(c)))
    acumuladas, prev, filas = [], evaluar(casa, Esc()), []
    for c in grupo0 + grupo1:
        acumuladas.append(c)
        m = evaluar(casa, _esc_de(acumuladas, esc))
        coste = coste_medida(casa, c, esc)
        d_ah = m["ahorro"] - prev["ahorro"]
        nombre, fuente, inst, cert = cat[c]
        filas.append(dict(clave=c, nombre=nombre, fuente=fuente, coste=coste, ahorro=d_ah,
                          payback=(coste / d_ah if d_ah > 0 and coste > 0 else (0.0 if coste == 0 and d_ah > 0 else float("inf"))),
                          instalacion=inst, certificados=cert))
        prev = m
    return filas


# ==============================================================================
# 6. ENTREGABLE: PRESUPUESTO PERSONALIZADO (HTML autocontenido, imprimible a PDF)
# ==============================================================================
def _e(x):
    return _html.escape(str(x))


def eur(x):
    return f"{x:,.0f} €".replace(",", ".")


def kwh(x):
    return f"{x:,.0f} kWh".replace(",", ".")


def _n(v):
    return f"{v:,.0f}".replace(",", ".")


def fmt_pb(m):
    if m["ahorro"] <= 0:
        return "No se amortiza"
    if m["inversion_neta"] == 0:
        return "Inmediato"
    return ">30 años" if m["payback"] > 30 else f"{m['payback']:.1f} años"


def fmt_pb_fila(f):
    if f["ahorro"] <= 0:
        return "Sin coste" if f["coste"] == 0 else "No aporta ahorro"
    if f["coste"] == 0:
        return "Inmediato"
    return ">30 años" if f["payback"] > 30 else f"{f['payback']:.1f} años"


def flujo(m, años=15):
    return [-m["inversion_neta"] + m["ahorro"] * t for t in range(años + 1)]


def svg_lineas(xs, series, w=680, h=330):
    ml, mr, mt, mb = 74, 20, 20, 66
    ys_all = [y for _, _, ys in series for y in ys]
    y0, y1 = min(0.0, min(ys_all)), max(0.0, max(ys_all))
    if y1 - y0 < 1:
        y1 = y0 + 1
    pad = (y1 - y0) * 0.05
    y1 += pad
    y0 -= pad if y0 < 0 else 0
    X = lambda x: ml + (x - xs[0]) / (xs[-1] - xs[0]) * (w - ml - mr)
    Y = lambda y: mt + (1 - (y - y0) / (y1 - y0)) * (h - mt - mb)
    o = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" '
         f'style="max-width:100%;height:auto;font-family:Helvetica,Arial,sans-serif">']
    for i in range(6):
        v = y0 + (y1 - y0) * i / 5
        o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e2e8f0"/>')
        o.append(f'<text x="{ml - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" font-size="11" fill="#475569">{_n(v)}</text>')
    for x in xs:
        if x % 3 == 0:
            o.append(f'<text x="{X(x):.1f}" y="{h - mb + 18}" text-anchor="middle" font-size="11" fill="#475569">{x}</text>')
    o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="#64748b" stroke-dasharray="4 3"/>')
    o.append(f'<text x="{(ml + w - mr) / 2:.0f}" y="{h - mb + 38}" text-anchor="middle" font-size="12" fill="#1e293b">Años</text>')
    o.append(f'<text x="14" y="{(mt + h - mb) / 2:.0f}" transform="rotate(-90 14 {(mt + h - mb) / 2:.0f})" '
             f'text-anchor="middle" font-size="12" fill="#1e293b">€ acumulados (ahorro − inversión)</text>')
    for n, c, ys in series:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="3"/>')
    lx = ml
    for n, c, _ in series:
        o.append(f'<rect x="{lx}" y="{h - 14}" width="14" height="4" fill="{c}"/>'
                 f'<text x="{lx + 20}" y="{h - 8}" font-size="12" fill="#1e293b">{_e(n)}</text>')
        lx += 30 + 7 * len(n)
    o.append("</svg>")
    return "".join(o)


def svg_barras(grupos, w=680, h=260):
    ml, mr, mt, mb = 20, 20, 30, 56
    vmax = max(v for _, bs in grupos for _, _, v in bs) or 1.0
    gw = (w - ml - mr) / len(grupos)
    o = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" '
         f'style="max-width:100%;height:auto;font-family:Helvetica,Arial,sans-serif">']
    for gi, (tit, bs) in enumerate(grupos):
        x0 = ml + gi * gw
        bw = gw / (len(bs) * 1.6 + 0.6)
        o.append(f'<text x="{x0 + gw / 2:.0f}" y="{h - 34}" text-anchor="middle" font-size="13" fill="#1e293b">{_e(tit)}</text>')
        for bi, (nom, col, v) in enumerate(bs):
            bx = x0 + bw * 0.5 + bi * bw * 1.6
            bh = (h - mt - mb) * v / vmax
            o.append(f'<rect x="{bx:.1f}" y="{h - mb - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{col}" rx="3"/>')
            o.append(f'<text x="{bx + bw / 2:.1f}" y="{h - mb - bh - 6:.1f}" text-anchor="middle" font-size="12" fill="#1e293b">{_n(v)} €</text>')
            o.append(f'<text x="{bx + bw / 2:.1f}" y="{h - 18}" text-anchor="middle" font-size="10" fill="#475569">{_e(nom)}</text>')
    o.append("</svg>")
    return "".join(o)


CSS_INFORME = """
body{font-family:Helvetica,Arial,sans-serif;color:#1e293b;background:#f4f8fb;margin:0;padding:24px}
.hoja{max-width:900px;margin:0 auto;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.08)}
.cab{background:#0a528a;color:#fff;padding:22px 28px}.cab h1{margin:0;font-size:26px}
.cab p{margin:4px 0 0;color:#ffd95c;font-style:italic}.cont{padding:24px 28px}
h2{color:#0a528a;border-bottom:3px solid #24b7da;padding-bottom:4px;margin-top:28px;font-size:18px}
table{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0}
th{background:#0a528a;color:#fff;text-align:left;padding:7px 8px}td{padding:7px 8px;border-bottom:1px solid #e2e8f0;vertical-align:top}
td.n,th.n{text-align:right;white-space:nowrap}.sub td{font-size:12px;color:#475569;background:#f4f8fb}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}.kpi{flex:1 1 130px;background:#f4f8fb;border-left:5px solid #24b7da;padding:10px 12px;border-radius:6px}
.kpi b{display:block;font-size:20px;color:#0a528a}.kpi span{font-size:12px;color:#475569}
.ok{border-left-color:#10b981}.aviso{font-size:12px;color:#475569;background:#fff7e0;border-left:5px solid #ffd95c;padding:10px 12px;border-radius:6px}
@media print{body{background:#fff;padding:0}.hoja{box-shadow:none}}
"""


def informe_html(d):
    casa, plan, base, conv, filas = d["casa"], d["plan"], d["base"], d["conv"], d["filas"]
    bruta, neta = plan["inversion"], plan["inversion_neta"]
    ded = bruta - neta
    kpis = [("Inversión bruta", eur(bruta), ""), ("Deducción / ayuda", eur(ded), ""),
            ("Inversión neta", eur(neta), ""), ("Ahorro anual estimado", eur(plan["ahorro"]), "ok"),
            ("Retorno (payback)", fmt_pb(plan), "ok")]
    if plan["prod"] > 0:
        kpis += [("Autoconsumo", f"{plan['auto_pct']:.0f} %", ""), ("Excedente regalado", kwh(plan["regalada"]), "")]
    h = [f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Presupuesto personalizado · {_e(d['cliente'])}</title>"
         f"<style>{CSS_INFORME}</style></head><body><div class='hoja'>",
         "<div class='cab'><h1>SmartStorage 360 × NEXA · Presupuesto personalizado</h1>"
         "<p>Conectamos ideas, impulsamos tu ahorro energético</p></div><div class='cont'>",
         f"<p><b>Cliente / vivienda:</b> {_e(d['cliente'])} &nbsp;·&nbsp; <b>Fecha:</b> {d['fecha']}<br>"
         f"<b>Tipo de usuario:</b> {_e(d['usuario'])}<br><b>Perfil:</b> {_e(d['perfil'])}<br>"
         f"<b>Estrategia elegida:</b> {_e(d['estrategia'])} → <b>{_e(plan['esc'].etiqueta())}</b></p>",
         "<h2>1. Resumen económico</h2><div class='kpis'>"]
    h += [f"<div class='kpi {c}'><b>{_e(v)}</b><span>{_e(t)}</span></div>" for t, v, c in kpis]
    h.append("</div>")
    h.append("<h2>2. Datos de partida</h2><table><tr><th>Dato</th><th>Valor</th></tr>")
    for k, v in d["datos"]:
        h.append(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>")
    h.append("</table><h2>3. Desglose de costes y ahorro (de menor a mayor inversión)</h2>"
             "<table><tr><th>Medida</th><th>Origen</th><th class='n'>Coste</th><th class='n'>Ahorro anual</th>"
             "<th class='n'>Retorno</th></tr>")
    for f in filas:
        h.append(f"<tr><td>{_e(f['nombre'])}</td><td>{_e(f['fuente'])}</td><td class='n'>{eur(f['coste'])}</td>"
                 f"<td class='n'>{eur(f['ahorro'])}</td><td class='n'>{_e(fmt_pb_fila(f))}</td></tr>"
                 f"<tr class='sub'><td colspan='5'><b>Instalación:</b> {_e(f['instalacion'])}<br>"
                 f"<b>Certificados y trámites:</b> {_e(f['certificados'])}</td></tr>")
    h.append(f"<tr><td><b>TOTAL</b></td><td></td><td class='n'><b>{eur(bruta)}</b></td>"
             f"<td class='n'><b>{eur(plan['ahorro'])}</b></td><td class='n'><b>{_e(fmt_pb(plan))}</b></td></tr></table>")
    h.append("<p class='aviso'>El ahorro de cada medida se calcula sobre las anteriores (no se suman ahorros sueltos), así que "
             "los importes de la columna suman el ahorro total del plan sin contar dos veces el mismo kWh.</p>")
    h.append("<h2>4. Comparativa con la solución convencional</h2>")
    if conv:
        ce, cm = conv
        red = (1 - bruta / cm["inversion"]) * 100 if cm["inversion"] > 0 else 0.0
        h.append("<table><tr><th>Concepto</th><th class='n'>Tu plan</th><th class='n'>Convencional</th></tr>"
                 f"<tr><td>Solución</td><td class='n'>{_e(plan['esc'].etiqueta())}</td><td class='n'>{_e(ce.etiqueta())}</td></tr>"
                 f"<tr><td>Inversión inicial (bruta)</td><td class='n'>{eur(bruta)}</td><td class='n'>{eur(cm['inversion'])}</td></tr>"
                 f"<tr><td>Ahorro anual</td><td class='n'>{eur(plan['ahorro'])}</td><td class='n'>{eur(cm['ahorro'])}</td></tr>"
                 f"<tr><td>Retorno</td><td class='n'>{_e(fmt_pb(plan))}</td><td class='n'>{_e(fmt_pb(cm))}</td></tr>"
                 f"<tr><td>Autoconsumo</td><td class='n'>{plan['auto_pct']:.0f} %</td><td class='n'>{cm['auto_pct']:.0f} %</td></tr>"
                 f"<tr><td>Excedente regalado</td><td class='n'>{kwh(plan['regalada'])}</td><td class='n'>{kwh(cm['regalada'])}</td></tr>"
                 "</table>")
        h.append(f"<p><b>Reducción del desembolso inicial frente a la solución convencional: {red:.0f} %.</b> "
                 "La solución convencional es fotovoltaica (existente o de tejado) con una batería de 10 kWh sin "
                 "gestión inteligente.</p>")
        xs = list(range(16))
        h.append(svg_barras([("Inversión inicial", [("Tu plan", "#24b7da", bruta), ("Convencional", "#fd9c7d", cm["inversion"])]),
                             ("Ahorro anual", [("Tu plan", "#24b7da", max(0, plan["ahorro"])),
                                               ("Convencional", "#fd9c7d", max(0, cm["ahorro"]))])]))
        h.append(svg_lineas(xs, [("Tu plan", "#0a528a", flujo(plan)), ("Convencional", "#fd9c7d", flujo(cm))]))
    else:
        h.append("<p>No se ha podido plantear una solución convencional equivalente con las restricciones de obra de la "
                 "vivienda (se necesita acceso a tejado para una instalación fotovoltaica con batería).</p>")
        h.append(svg_lineas(list(range(16)), [("Tu plan", "#0a528a", flujo(plan))]))
    h.append("<h2>5. Ayudas e incentivos</h2><p>La inversión neta aplica una deducción/ayuda del "
             f"<b>{casa.ded_pct:g} %</b> introducida por el usuario. Consulta la convocatoria vigente de tu comunidad "
             "autónoma, las deducciones del IRPF y las bonificaciones municipales (IBI/ICIO) antes de contratar: "
             "cambian con frecuencia.</p>")
    h.append("<h2>6. Hipótesis del cálculo</h2><ul>")
    for t in d["hipotesis"]:
        h.append(f"<li>{_e(t)}</li>")
    h.append("</ul><p class='aviso'>Estimación orientativa generada por el simulador SmartStorage 360 × NEXA. No sustituye a un "
             "estudio técnico ni a un presupuesto firmado por una empresa instaladora.</p></div></div></body></html>")
    return "".join(h)


# ==============================================================================
# 7. INTERFAZ STREAMLIT
# ==============================================================================
st.set_page_config(page_title="SmartStorage 360 × NEXA", page_icon="🔋", layout="wide")

st.markdown(
    """<style>
.banner{background:#0a528a;color:#fff;padding:18px 24px;border-radius:12px;margin-bottom:10px}
.banner h1{margin:0;font-size:1.9rem;color:#fff}.banner p{margin:4px 0 0;color:#ffd95c;font-style:italic}
</style>
<div class="banner"><h1>🔋 SmartStorage 360 × NEXA</h1>
<p>Conectamos ideas, impulsamos tu ahorro energético</p></div>""",
    unsafe_allow_html=True)


def ancho(fn, *args, **kwargs):
    """`width="stretch"` en Streamlit reciente; `use_container_width` en versiones antiguas."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


U_SS = "🔋 Autoconsumidor con placas (SmartStorage 360)"
U_NX = "🌍 Vivienda en eficiencia (NEXA): sin placas / obra o presupuesto limitado"
EQ_GAS, EQ_TERMO = "Caldera de gas", "Termo eléctrico de agua"
EQ_CLIMA = "Aerotermia / Suelo radiante / Bomba de calor (incl. split)"
EQ_EV = "Vehículo eléctrico"
EQUIPOS = [EQ_GAS, EQ_TERMO, EQ_CLIMA, EQ_EV]
OCUP = ["Fuera en horas de sol (en casa por tardes/noches)", "Dentro de casa / teletrabajo (todo el día)",
        "Principalmente mañanas en casa"]
OBRAS = ["Solo Balcón/Sin Obra", "Acceso Total a Tejado", "Sin Restricciones"]
OBRA_MAP = {OBRAS[0]: OBRA_BALCON, OBRAS[1]: OBRA_TEJADO, OBRAS[2]: OBRA_TEJADO}
MODOS = ["Conozco mis kWh", "Solo sé lo que pago (€/mes)"]

COMUN = dict(cliente="Estudio personalizado", ccaa="Otras CCAA (Sur)", pers=3, litros=150, modulo=False,
             p_kwh=0.142, p_exc=0.07, p_gas=0.088, cuota_vb=4.0, c_n2=85, c_n3=690, c_modulo=450, bat_fijo=100,
             bat_kwh_precio=510, c_kit=550, coste_kwp=1267, kwp_nuevo=3.0, ded=0, max_pb=5.0, obj_auto=75,
             pot_kw=4.6, base=2400, app=600, clima_kwh=900, ev_kwh=1600, gas_kwh=4800, gasto_elec=70, gasto_gas=50,
             kwp=4.0, rend=1500, tipo=TIPOS[0], m2=85, obra=OBRAS[0])
PRESET_SS = dict(COMUN, usuario=U_SS, eq=[EQ_TERMO, EQ_CLIMA], ocup=OCUP[0], presupuesto=3500, modo=MODOS[0])
PRESET_NX = dict(COMUN, usuario=U_NX, eq=[EQ_GAS], ocup=OCUP[1], presupuesto=4000, modo=MODOS[1],
                 ccaa="Andalucía", rend=RENDS["Andalucía"])
for k, v in PRESET_SS.items():
    st.session_state.setdefault(k, v)


def cargar(preset):
    def _f():
        for k, v in preset.items():
            st.session_state[k] = v
    return _f


def cambiar_usuario():
    cargar(PRESET_SS if st.session_state["usuario"] == U_SS else PRESET_NX)()


def cambiar_ccaa():
    st.session_state["rend"] = RENDS[st.session_state["ccaa"]]


PERFILES = {
    0: dict(titulo="Perfil NEXA · Vivienda sin generación solar",
            problema="Pagas toda la energía a precio de red. Primero conviene generar y reducir la demanda; el "
                     "almacenamiento solo tiene sentido cuando ya hay placas.",
            ruta="Eficiencia (aislamiento, VMC, ventanas) según la obra permitida + generación (kit de balcón o "
                 "tejado) + niveles 1–4 de SmartStorage; batería de litio solo si el consumo nocturno sigue siendo alto.",
            esc=None),
    1: dict(titulo="Perfil 1 · Termo eléctrico y/o aerotermia + día FUERA de casa",
            problema="La vivienda está vacía en el pico solar y dispara el consumo eléctrico al volver (~19:30 h) para "
                     "calentar agua y climatizar.",
            ruta="Niveles 1 + 2 + 3 (desviador HEMS + SG Ready) + Nivel 4 (batería virtual): convierte el termo y la casa "
                 "en baterías térmicas a mediodía sin gastar en litio.",
            esc=Esc(True, True, True, True, 0.0)),
    2: dict(titulo="Perfil 2 · Calefacción y ACS de gas + día FUERA de casa",
            problema="No hay cargas térmicas eléctricas donde desviar el sol a mediodía, pero el consumo eléctrico "
                     "nocturno se mantiene.",
            ruta="Nivel 1 (programación de electrodomésticos) + Nivel 4 (batería virtual) + Nivel 5 (litio de 5 kWh). "
                 "Con el módulo híbrido (termo precalentador) también puedes usar los niveles 2 y 3.",
            esc=Esc(True, False, False, True, 5.0)),
    3: dict(titulo="Perfil 3 · Teletrabajo (DENTRO de casa) + vehículo eléctrico / climatización",
            problema="Presencia en las horas centrales con alta capacidad de consumo directo si se escalonan los equipos.",
            ruta="Nivel 1 (hábitos escalonados + carga solar del VE) + Nivel 2 (programador térmico) + Nivel 4 (batería virtual).",
            esc=Esc(True, True, False, True, 0.0)),
}

# --- Sidebar --------------------------------------------------------------------------
with st.sidebar:
    st.header("👤 ¿Quién eres?")
    usuario = st.radio("Tipo de usuario", [U_SS, U_NX], key="usuario", on_change=cambiar_usuario)
    es_ss = usuario == U_SS
    c_b1, c_b2 = st.columns(2)
    c_b1.button("Vivienda tipo PDF", on_click=cargar(PRESET_SS), help="4 kWp · 6.000 kWh/año · termo + aerotermia")
    c_b2.button("Vivienda tipo NEXA", on_click=cargar(PRESET_NX), help="Piso de 85 m² · gas · sin placas · 4.000 €")
    cliente = st.text_input("Nombre / vivienda", key="cliente")

    st.header("Paso 1 · Cuestionario")
    ccaa = st.selectbox("Comunidad Autónoma", list(RENDS.keys()), key="ccaa", on_change=cambiar_ccaa)
    eq = st.multiselect("A · ¿Qué sistemas tienes en casa?", EQUIPOS, key="eq")
    ocup = st.radio("B · ¿Dónde pasas el día?", OCUP, key="ocup")
    presupuesto = st.number_input("Presupuesto máximo (€)", 0, 60000, step=250, key="presupuesto")
    if not es_ss:
        tipo = st.selectbox("Tipo de vivienda", TIPOS, key="tipo")
        m2 = st.number_input("Superficie (m²)", 20, 600, step=5, key="m2")
        obra = st.selectbox("¿Tejado u obra exterior?", OBRAS, key="obra")
    else:
        tipo, m2, obra = TIPOS[0], 85.0, OBRAS[0]      # el PDF deja fuera reformas estructurales

    st.header("Paso 2 · Instalación y consumo")
    if es_ss:
        kwp = st.number_input("Potencia fotovoltaica instalada (kWp)", 0.5, 20.0, step=0.5, key="kwp")
        rend = st.number_input("Rendimiento (kWh/kWp·año)", 900, 2000, step=50, key="rend")
    else:
        kwp, rend = 0.0, st.number_input("Rendimiento solar de tu zona (kWh/kWp·año)", 900, 2000, step=50, key="rend")
    modo = st.radio("¿Cómo introduces tu consumo?", MODOS, key="modo", horizontal=False)
    pers, litros, modulo = 3, 150.0, False
    if EQ_TERMO in eq or EQ_GAS in eq:
        pers = st.number_input("Personas en la vivienda", 1, 8, key="pers")
    if EQ_TERMO in eq:
        litros = st.number_input("Capacidad del termo (L)", 50, 500, step=10, key="litros")
    if EQ_GAS in eq and EQ_TERMO not in eq:
        modulo = st.checkbox("Módulo híbrido gas: termo eléctrico precalentador en serie con la caldera", key="modulo")

    s = st.session_state
    if modo == MODOS[0]:
        base = st.number_input("Consumo base (kWh/año, sin ACS/clima/VE)", 500, 8000, step=100, key="base")
        app = st.number_input("Electrodomésticos programables (kWh/año)", 0, 2000, step=50, key="app")
        clima_kwh = st.number_input("Climatización eléctrica (kWh/año)", 100, 6000, step=100, key="clima_kwh") if EQ_CLIMA in eq else 0.0
        ev_kwh = st.number_input("Recarga del VE (kWh/año)", 200, 4000, step=100, key="ev_kwh") if EQ_EV in eq else 0.0
        gas_kwh = st.number_input("Consumo de gas (kWh/año)", 500, 20000, step=200, key="gas_kwh") if EQ_GAS in eq else 0.0
    else:
        gasto_elec = st.number_input("Gasto eléctrico (€/mes, factura completa)", 10, 600, step=5, key="gasto_elec")
        gasto_gas = st.number_input("Gasto de gas (€/mes)", 0, 400, step=5, key="gasto_gas") if EQ_GAS in eq else 0
        total = max(1200.0, (gasto_elec * 12 - FIJO_ELEC_ANUAL) / (s["p_kwh"] * IVA_ELEC))
        gas_kwh = max(0.0, (gasto_gas * 12 - FIJO_GAS_ANUAL) / (s["p_gas"] * IVA_GAS)) if EQ_GAS in eq else 0.0
        acs = pers * LITROS_PERSONA * CP * (T_ACS - float(np.mean(T_RED))) * 365 if EQ_TERMO in eq else 0.0
        clima_kwh = 0.18 * total if EQ_CLIMA in eq else 0.0
        ev_kwh = 1600.0 if EQ_EV in eq else 0.0
        app = 0.12 * total
        base = max(600.0, total - acs - clima_kwh - ev_kwh - app)
        st.caption(f"Estimación: {kwh(total)}/año de electricidad" + (f" y {kwh(gas_kwh)}/año de gas." if gas_kwh else "."))

    with st.expander("Tarifa, costes y ayudas (editables)"):
        st.number_input("Energía (€/kWh, sin impuestos)", 0.05, 0.40, step=0.005, format="%.3f", key="p_kwh")
        st.number_input("Excedentes (€/kWh)", 0.01, 0.20, step=0.005, format="%.3f", key="p_exc")
        st.number_input("Gas (€/kWh, sin IVA)", 0.03, 0.25, step=0.005, format="%.3f", key="p_gas")
        st.number_input("Cuota batería virtual (€/mes)", 0.0, 15.0, step=0.5, key="cuota_vb")
        st.number_input("Nivel 2 · temporizador (€)", 0, 500, step=5, key="c_n2")
        st.number_input("Nivel 3 · desviador HEMS + SG Ready (€)", 0, 2000, step=10, key="c_n3")
        st.number_input("Módulo híbrido gas · termo (€)", 0, 1500, step=10, key="c_modulo")
        st.number_input("Batería · coste fijo BMS + mano de obra (€)", 0, 1000, step=10, key="bat_fijo")
        st.number_input("Batería · €/kWh", 200, 900, step=10, key="bat_kwh_precio")
        st.number_input("Kit de balcón (€)", 100, 2000, step=10, key="c_kit")
        st.number_input("FV de tejado (€/kWp)", 600, 2500, step=25, key="coste_kwp")
        st.number_input("FV de tejado: potencia (kWp)", 1.0, 10.0, step=0.5, key="kwp_nuevo")
        st.slider("Deducción fiscal / subvención sobre la inversión (%)", 0, 80, key="ded")

    st.header("Objetivos")
    max_pb = st.slider("Retorno máximo aceptable (años)", 1.0, 15.0, step=0.5, key="max_pb")
    obj_auto = st.slider("Autoconsumo objetivo del PDF (%)", 40, 95, key="obj_auto") if es_ss else 75

casa = Casa(
    pv_existente=es_ss, kwp=float(kwp), rend=float(rend), consumo_base=float(base), app_kwh=float(app),
    personas=int(pers), litros=float(litros), termo=EQ_TERMO in eq, clima_elec=EQ_CLIMA in eq,
    clima_kwh=float(clima_kwh), gas=EQ_GAS in eq, gas_kwh=float(gas_kwh), modulo_gas=modulo, ev=EQ_EV in eq,
    ev_kwh=float(ev_kwh), fuera=(ocup == OCUP[0]), tipo=tipo, m2=float(m2), obra=OBRA_MAP[obra],
    presupuesto=float(presupuesto), ded_pct=float(s["ded"]), p_kwh=s["p_kwh"], p_exc=s["p_exc"], p_gas=s["p_gas"],
    cuota_vb=s["cuota_vb"], c_n2=float(s["c_n2"]), c_n3=float(s["c_n3"]), c_modulo=float(s["c_modulo"]),
    bat_fijo=float(s["bat_fijo"]), bat_kwh_precio=float(s["bat_kwh_precio"]), c_kit=float(s["c_kit"]),
    kwp_nuevo=float(s["kwp_nuevo"]), coste_kwp=float(s["coste_kwp"]),
)

# --- Cálculos en tiempo real -----------------------------------------------------------
perf = perfil(casa)
P = PERFILES[perf]
b0 = metricas(casa, Esc())
est = estrategias(casa, presupuesto, max_pb, obj_auto)
conv_esc, conv_m = convencional(casa)
demanda_elec = b0["red"] + b0["auto"]

OPC = {}
if est["equilibrada"]:
    OPC["⚖️ Equilibrada (mayor ahorro con retorno ≤ plazo)"] = est["equilibrada"]
if est["rapido"]:
    OPC["⚡ Retorno más rápido"] = est["rapido"]
if est["mayor"]:
    OPC["💰 Mayor ahorro posible"] = est["mayor"]
if es_ss and est["objetivo"]:
    OPC["🎯 Objetivo del PDF (autoconsumo y vertido ≈ 0)"] = est["objetivo"]
if OPC and st.session_state.get("estrategia") not in OPC:
    st.session_state["estrategia"] = list(OPC)[0]
nombre_est = st.radio("Plan que quieres presupuestar", list(OPC), key="estrategia", horizontal=True) if OPC else None
plan = OPC[nombre_est] if nombre_est else None

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🩺 Diagnóstico", "💶 Presupuesto en tiempo real", "⏱️ Balance 24 h",
                                        "✅ Validación con el PDF", "🧩 Cómo se fusionan"])

# ===================================== TAB 1 =========================================
with tab1:
    st.subheader(P["titulo"])
    st.write(P["problema"])
    if casa.pv_existente:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Producción solar", kwh(b0["prod"]))
        c2.metric("Autoconsumo actual", f"{b0['auto_pct']:.0f} %", kwh(b0["auto"]), delta_color="off")
        c3.metric("Excedente vertido a red", kwh(b0["vertido"]))
        c4.metric("Regalado a 0 € (tope mensual)", kwh(b0["regalada"]), delta_color="off")
        st.caption(f"Demanda eléctrica anual modelada: {kwh(demanda_elec)}. El excedente se compensa cada mes solo hasta el "
                   "importe del término de energía de esa factura; el resto se regala.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Demanda eléctrica anual", kwh(demanda_elec))
        c2.metric("Demanda de gas anual", kwh(casa.gas_kwh))
        pot = casa.kwp_nuevo * casa.rend if casa.techo_ok else casa.kit_kwp * casa.rend * casa.kit_factor
        c3.metric("Producción solar posible", kwh(pot), "tejado" if casa.techo_ok else "kit de balcón", delta_color="off")
        if not casa.envolvente_ok:
            st.info("Con alquiler o sin obra exterior no se consideran aislamiento, VMC ni ventanas: solo kit de balcón y hábitos.")
    st.markdown(f"**Hoja de ruta orientativa:** {P['ruta']}")
    if est["n_dentro"] == 0:
        pos = [c for c in candidatos(casa) if c["ahorro"] > 0]
        if pos:
            minimo = min(pos, key=lambda c: c["inversion"])
            st.warning(f"Con {eur(presupuesto)} no cabe ninguna combinación rentable. La más barata ({minimo['esc'].etiqueta()}) "
                       f"cuesta {eur(minimo['inversion'])}.")
        else:
            st.warning("Ninguna combinación genera ahorro con los datos introducidos. Revisa consumo, tarifa y obra permitida.")
    else:
        st.caption(f"Evaluadas {est['n_candidatos']} combinaciones; {est['n_dentro']} caben en tu presupuesto y ahorran dinero.")

    st.markdown("### 🎯 Las mejores combinaciones, calculadas al momento")
    cols = st.columns(4 if es_ss else 3)
    cartas = [("⚖️ Equilibrada", est["equilibrada"], f"No hay plan con retorno ≤ {max_pb:g} años."),
              ("⚡ Retorno más rápido", est["rapido"], "Sin opciones dentro del presupuesto."),
              ("💰 Mayor ahorro", est["mayor"], "Sin opciones dentro del presupuesto.")]
    if es_ss:
        cartas.append(("🎯 Objetivo PDF", est["objetivo"], f"Autoconsumo ≥ {obj_auto} % no alcanzable con tu consumo."))
    for col, (tit, c, vacio) in zip(cols, cartas):
        with col:
            st.markdown(f"**{tit}**")
            if c:
                st.success(c["esc"].etiqueta())
                st.metric("Inversión", eur(c["inversion"]))
                st.metric("Ahorro anual", eur(c["ahorro"]))
                st.metric("Retorno", fmt_pb(c))
                if c["prod"] > 0:
                    st.caption(f"Autoconsumo {c['auto_pct']:.0f} % · regalado {kwh(c['regalada'])}")
                if c["inversion"] > presupuesto:
                    st.caption(f"⚠️ Supera tu presupuesto en {eur(c['inversion'] - presupuesto)}.")
            else:
                st.info(vacio)

    if plan is not None and casa.sumidero_acs and hay_pv(casa, plan["esc"]) and (plan["esc"].n2 or plan["esc"].n3):
        a = anio(casa, plan["esc"])
        cobertura = np.divide(a["acs_sol"], a["acs_tot"], out=np.zeros(12), where=a["acs_tot"] > 0) * 100
        st.markdown("### 🚿 Ducha garantizada")
        fig = go.Figure(go.Bar(x=MESES, y=cobertura, marker_color="#fd9c7d", name="ACS cubierta con sol"))
        fig.update_layout(template="plotly_dark", height=300, yaxis_title="% del agua caliente con sol",
                          title=f"Cobertura solar del ACS ({plan['esc'].etiqueta()})")
        ancho(st.plotly_chart, fig)
        st.caption("El HEMS activa un respaldo de red a las 17:30 h solo los días en que el agua no ha alcanzado 50 °C con sol.")

# ===================================== TAB 2 =========================================
with tab2:
    st.subheader("Presupuesto personalizado")
    if plan is None:
        st.info("Ajusta presupuesto, equipamiento u obra permitida: ahora mismo no hay un plan rentable que presupuestar.")
    else:
        esc_p = plan["esc"]
        filas = desglose(casa, esc_p)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Inversión bruta", eur(plan["inversion"]))
        k2.metric("Deducción / ayuda", eur(plan["inversion"] - plan["inversion_neta"]))
        k3.metric("Inversión neta", eur(plan["inversion_neta"]))
        k4.metric("Ahorro anual", eur(plan["ahorro"]))
        k5.metric("Retorno", fmt_pb(plan))
        st.markdown(f"**Plan:** {esc_p.etiqueta()}")
        tabla = [{"Medida": f["nombre"], "Origen": f["fuente"], "Coste (€)": round(f["coste"]),
                  "Ahorro anual (€)": round(f["ahorro"]), "Retorno": fmt_pb_fila(f)} for f in filas]
        tabla.append({"Medida": "TOTAL", "Origen": "", "Coste (€)": round(plan["inversion"]),
                      "Ahorro anual (€)": round(plan["ahorro"]), "Retorno": fmt_pb(plan)})
        ancho(st.dataframe, pd.DataFrame(tabla), hide_index=True)
        with st.expander("Instalación y certificados de cada medida"):
            for f in filas:
                st.markdown(f"**{f['nombre']}** ({f['fuente']})  \n*Instalación:* {f['instalacion']}  \n*Certificados:* {f['certificados']}")

        st.markdown("### Comparativa con la solución convencional")
        if conv_m is None:
            st.info("Sin acceso a tejado no hay una solución convencional equivalente (FV + batería de 10 kWh) con la que comparar.")
        else:
            red = (1 - plan["inversion"] / conv_m["inversion"]) * 100
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Inversión: tu plan vs convencional", eur(plan["inversion"]), f"{-red:+.0f} % vs {eur(conv_m['inversion'])}",
                      delta_color="inverse")
            d2.metric("Ahorro anual: plan", eur(plan["ahorro"]), f"{plan['ahorro'] - conv_m['ahorro']:+,.0f} € vs convencional".replace(",", "."))
            d3.metric("Retorno: plan", fmt_pb(plan), f"convencional: {fmt_pb(conv_m)}", delta_color="off")
            d4.metric("Autoconsumo: plan", f"{plan['auto_pct']:.0f} %", f"convencional: {conv_m['auto_pct']:.0f} %", delta_color="off")
            g1, g2 = st.columns(2)
            figb = go.Figure()
            figb.add_bar(x=["Inversión inicial", "Ahorro anual"], y=[plan["inversion"], max(0, plan["ahorro"])], name="Tu plan", marker_color="#24b7da")
            figb.add_bar(x=["Inversión inicial", "Ahorro anual"], y=[conv_m["inversion"], max(0, conv_m["ahorro"])],
                         name=f"Convencional ({conv_esc.etiqueta()})", marker_color="#fd9c7d")
            figb.update_layout(barmode="group", template="plotly_dark", height=360, yaxis_title="€", title="Inversión y ahorro")
            with g1:
                ancho(st.plotly_chart, figb)
            figf = go.Figure()
            años = list(range(16))
            figf.add_trace(go.Scatter(x=años, y=flujo(plan), name="Tu plan", mode="lines", line=dict(color="#24b7da", width=4)))
            figf.add_trace(go.Scatter(x=años, y=flujo(conv_m), name="Convencional", mode="lines", line=dict(color="#fd9c7d", width=4)))
            figf.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
            figf.update_layout(template="plotly_dark", height=360, xaxis_title="Años", yaxis_title="€ acumulados (ahorro − inversión)",
                               title="Flujo de caja a 15 años")
            with g2:
                ancho(st.plotly_chart, figf)

        if hay_pv(casa, esc_p):
            st.markdown("### Escalera de los 5 niveles (PDF)")
            st.caption("Cada escalón se calcula sobre el anterior. Parte de la instalación solar del plan, sin almacenamiento.")
            s_base = esc_p.solar
            escalones = [("Nivel 1 · Cambio de hábitos", Esc(True, solar=s_base)),
                         ("Nivel 2 · Inercia térmica", Esc(True, True, solar=s_base)),
                         ("Nivel 3 · Domótica HEMS", Esc(True, True, True, solar=s_base)),
                         ("Nivel 4 · Batería virtual", Esc(True, True, True, True, solar=s_base)),
                         ("Nivel 5 · Batería LFP 5 kWh", Esc(True, True, True, True, 5.0, solar=s_base))]
            prev, rows = evaluar(casa, Esc(solar=s_base)), []
            if s_base == 1:
                escalones = escalones[:1]
            for nombre, esc in escalones:
                if nombre.startswith(("Nivel 2", "Nivel 3")) and not casa.factible_hems:
                    rows.append({"Escalón": nombre, "Coste (€)": "—", "Ahorro (€/año)": "—",
                                 "Retorno": "No aplicable (sin carga térmica eléctrica)"})
                    continue
                m = evaluar(casa, esc)
                d_inv, d_ah = m["inversion"] - prev["inversion"], m["ahorro"] - prev["ahorro"]
                pb = "No aporta ahorro" if d_ah <= 0 else ("Inmediato" if d_inv <= 0 else f"{d_inv / d_ah:.1f} años")
                rows.append({"Escalón": nombre, "Coste (€)": round(d_inv), "Ahorro (€/año)": round(d_ah), "Retorno": pb})
                prev = m
            ancho(st.dataframe, pd.DataFrame(rows), hide_index=True)

        st.markdown("### 📄 Entregable")
        datos = [("Tipo de vivienda / superficie", f"{casa.tipo}, {casa.m2:g} m²" if not es_ss else "Vivienda con placas"),
                 ("Comunidad Autónoma", ccaa), ("Sistemas", ", ".join(eq) or "Ninguno indicado"),
                 ("Presencia", ocup), ("Obra permitida", obra if not es_ss else "Sin obra de envolvente (alcance SmartStorage 360)"),
                 ("Instalación fotovoltaica existente", f"{casa.kwp:g} kWp · {kwh(casa.kwp * casa.rend)}/año" if es_ss else "No"),
                 ("Demanda eléctrica anual", kwh(demanda_elec)), ("Demanda de gas anual", kwh(casa.gas_kwh)),
                 ("Presupuesto máximo", eur(presupuesto))]
        hip = [f"Precio de la energía {casa.p_kwh * IVA_ELEC:.3f} €/kWh con impuestos; excedentes a {casa.p_exc:.3f} €/kWh, "
               "compensados cada mes solo hasta el importe del término de energía.",
               f"Producción con {casa.rend:.0f} kWh/kWp·año; día soleado y nublado ponderados por la nubosidad mensual.",
               f"Termo: Etérmica = m·cp·ΔT; temporizador {P_TIMER} kW, desviador hasta {P_RES} kW; batería con rendimiento del {ETA_BAT:.0%}.",
               f"Medidas de envolvente: reducen la demanda térmica un {casa.red_aisl:.0%} (aislamiento), {casa.red_vmc:.0%} (VMC) y "
               f"{casa.red_vent:.0%} (ventanas), con un tope conjunto del {TOPE_EFICIENCIA:.0%}.",
               "El kit de balcón no compensa excedentes. No se modelan tarifas por tramos horarios (2.0TD) ni la inflación."]
        if st.button("📄 Generar Presupuesto Personalizado", type="primary"):
            html_doc = informe_html(dict(
                casa=casa, plan=plan, base=b0, conv=(conv_esc, conv_m) if conv_m else None, filas=filas, cliente=cliente,
                usuario=usuario, perfil=P["titulo"], estrategia=nombre_est, fecha=datetime.date.today().strftime("%d/%m/%Y"),
                datos=datos, hipotesis=hip))
            st.session_state["informe"] = (casa, esc_p, html_doc)
        if "informe" in st.session_state:
            casa_i, esc_i, html_i = st.session_state["informe"]
            if (casa_i, esc_i) != (casa, esc_p):
                st.warning("Has cambiado datos o plan desde que generaste el presupuesto. Vuelve a generarlo para actualizarlo.")
            st.success("Presupuesto generado. Descárgalo y ábrelo en el navegador (Imprimir → Guardar como PDF).")
            st.download_button("⬇️ Descargar presupuesto personalizado (HTML)", data=html_i,
                               file_name="presupuesto_smartstorage360_nexa.html", mime="text/html")
            with st.expander("Vista previa"):
                st.components.v1.html(html_i, height=900, scrolling=True)

# ===================================== TAB 3 =========================================
with tab3:
    st.subheader("Balance energético de 24 h")
    if not (casa.pv_existente or (plan and plan["esc"].solar > 0)):
        st.info("Sin generación solar no hay balance horario que mostrar. Elige un plan que incluya placas.")
    opciones = {"Situación actual": Esc()}
    if plan:
        opciones["Plan elegido"] = plan["esc"]
    if conv_esc:
        opciones["Solución convencional"] = conv_esc
    k1, k2, k3 = st.columns(3)
    nom = k1.selectbox("Escenario a comparar con la situación actual", list(opciones.keys()), index=min(1, len(opciones) - 1))
    mes = k2.selectbox("Mes", MESES, index=5)
    tipo_dia = k3.radio("Tipo de día", ["Soleado", "Nublado"], horizontal=True)
    mi, soleado = MESES.index(mes), tipo_dia == "Soleado"

    def fig_dia(d, titulo):
        fig = go.Figure()
        for n, y, c in (("Consumo base", d["base"], "#64748b"), ("Electrodomésticos", d["app"], "#a78bfa"),
                        ("Vehículo eléctrico", d["ev"], "#24b7da"), ("ACS / termo", d["acs"], "#fd9c7d"),
                        ("Climatización", d["clima"], "#ef4444")):
            fig.add_bar(x=HORAS, y=y, name=n, marker_color=c)
        fig.add_trace(go.Scatter(x=HORAS, y=d["S"], name="Producción solar", mode="lines", line=dict(color="#ffd95c", width=4)))
        fig.add_trace(go.Scatter(x=HORAS, y=-d["exc"], name="Excedente a red (negativo)", fill="tozeroy", mode="lines",
                                 line=dict(color="#10b981", width=1)))
        fig.update_layout(barmode="stack", template="plotly_dark", height=420, title=titulo, xaxis_title="Hora del día",
                          yaxis_title="kWh por hora", legend=dict(orientation="h", y=-0.25))
        return fig

    d0, d1 = dia(casa, Esc(), mi, soleado), dia(casa, opciones[nom], mi, soleado)
    g1, g2 = st.columns(2)
    for col, d, tit in ((g1, d0, "Situación actual"), (g2, d1, nom)):
        with col:
            ancho(st.plotly_chart, fig_dia(d, tit))
            sp = d["S"].sum()
            st.caption(f"Producción {sp:.1f} kWh · autoconsumo {(sp - d['exc'].sum()) / sp * 100 if sp else 0:.0f} % · "
                       f"excedente {d['exc'].sum():.1f} kWh · compra a red {d['red'].sum():.1f} kWh · "
                       f"descarga de batería {d['dis'].sum():.1f} kWh")
    with st.expander("Cómo calcula el simulador"):
        st.markdown(
            f"""
- **Día típico por mes** (soleado y nublado, ponderados por nubosidad) con perfil solar y de consumo **hora a hora**.
- **Etérmica = m · cp · ΔT**: el termo absorbe como máximo `litros × 1,163 Wh/(L·K) × (65 °C − T agua de red)`, limitado por el ACS que se gasta ese día.
- **Nivel 2:** temporizador de {P_TIMER} kW a horas fijas (12–15 h); en un día nublado tira de red. **Nivel 3:** desviador que absorbe watt a watt solo excedente real (hasta {P_RES} kW en el termo y {P_CLIMA} kW en climatización, +1,5 °C con penalización del {int((PEN_CLIMA - 1) * 100)} %).
- **Nivel 5:** batería con rendimiento del {ETA_BAT:.0%} en cada sentido y potencia C/{1 / C_RATE:.0f}.
- **Kit de balcón, FV de tejado, aislamiento, VMC y ventanas (NEXA):** el kit y el tejado entran como producción solar en el mismo balance; las medidas de envolvente reducen la demanda térmica (gas o climatización eléctrica).
- **Regalo a la red:** cada mes el excedente solo se compensa hasta el importe del término de energía; el resto se regala o se guarda en euros con la batería virtual.
            """)

# ===================================== TAB 4 =========================================
with tab4:
    st.subheader("Contraste con los números del PDF (vivienda tipo)")
    tipo_casa = Casa(clima_elec=True)
    bt = metricas(tipo_casa, Esc())
    st.caption("Vivienda tipo del PDF: 4,0 kWp, 6.000 kWh/año, perfil «fuera de casa» con termo y aerotermia.")
    cA, cB, cC = st.columns(3)
    cA.metric("Autoconsumo directo base", f"{bt['auto_pct']:.0f} %", "PDF: 35 %", delta_color="off")
    cB.metric("Vertido a red", kwh(bt["vertido"]), "PDF: 3.900 kWh", delta_color="off")
    cC.metric("Regalado a 0 €", kwh(bt["regalada"]), "PDF: 1.650 kWh", delta_color="off")
    pdf_rows = [("Nivel 1", 0, 650, 117, Esc(True)), ("Nivel 2", 85, 950, 171, Esc(True, True)),
                ("Nivel 3", 690, 1600, 288, Esc(True, True, True)), ("Nivel 4", 0, "100 % sobrante", 215, Esc(True, True, True, True)),
                ("Nivel 5 (5 kWh)", 2650, 1825, 328, Esc(True, True, True, True, 5.0))]
    prev, filas_v = bt | dict(ahorro=0.0), []
    for nombre, c, rec_kwh, ah, esc in pdf_rows:
        m = evaluar(tipo_casa, esc)
        d_inv, d_ah = m["inversion"] - prev["inversion"], m["ahorro"] - prev["ahorro"]
        d_rec = (m["auto"] - prev["auto"]) + (m["rescatada_vb"] - prev.get("rescatada_vb", 0.0))
        filas_v.append({"Escalón": nombre, "Coste PDF (€)": c, "Coste simulador (€)": round(d_inv), "Recuperado PDF (kWh)": rec_kwh,
                        "Recuperado simulador (kWh)": round(d_rec), "Ahorro PDF (€/año)": ah, "Ahorro simulador (€/año)": round(d_ah)})
        prev = m
    for nombre, c, ah, esc in (("Paquete óptimo sin litio", 775, 576, Esc(True, True, True, True, 0.0)),
                               ("Paquete independencia total", 3340, 793, Esc(True, False, True, True, 5.0))):
        m = evaluar(tipo_casa, esc)
        filas_v.append({"Escalón": nombre, "Coste PDF (€)": c, "Coste simulador (€)": round(m["inversion"]), "Recuperado PDF (kWh)": "—",
                        "Recuperado simulador (kWh)": round(m["auto"] - bt["auto"] + m["rescatada_vb"]),
                        "Ahorro PDF (€/año)": ah, "Ahorro simulador (€/año)": round(m["ahorro"])})
    ancho(st.dataframe, pd.DataFrame(filas_v), hide_index=True)
    st.caption("El simulador valora cada kWh recuperado al precio de lo que deja de comprarse menos lo que dejaba de compensarse "
               "por excedentes; el PDF lo valora siempre a 0,18 €/kWh.")
    st.subheader("Auditoría aritmética del PDF")
    ok, mal = "✅", "❌"
    chk = [
        ("Ahorro N1+N2+N3 = suma de niveles", f"117+171+288 = {117 + 171 + 288} €", "576 €", ok),
        ("Ahorro N1+N3+N5 = suma de niveles", f"117+288+328 = {117 + 288 + 328} €", "793 €", mal),
        ("kWh recuperados N1+N2+N3 caben en el vertido", f"650+950+1.600 = {650 + 950 + 1600} kWh", "vertido 3.900 kWh", ok),
        ("kWh recuperados N1+N3+N5 caben en el vertido", f"650+1.600+1.825 = {650 + 1600 + 1825} kWh", "vertido 3.900 kWh", mal),
        ("1.825 kWh de la batería de 5 kWh", f"5 × 365 = {5 * 365} kWh (ciclo completo diario, sin pérdidas)", "con 90–95 % ≈ 1.640–1.730 kWh", mal),
        ("Nivel 4 aditivo con N1–N3", f"215 € ÷ 1.650 kWh = {215 / 1650:.2f} €/kWh; tras N1–N3 quedan {3900 - 3200} kWh",
         "215 €/año como ahorro propio; el paquete de 576 € ni lo incluye ni resta la cuota de 48 €/año", mal),
        ("kWh recuperados valorados a 0,18 €/kWh", f"1.550 kWh se compensaban a ≈0,07 → 576 − 1.550·0,07 = {576 - 1550 * 0.07:.0f} €", "576 €", mal),
        ("Retornos N2, N3, N5", f"{85 / 171:.2f} · {690 / 288:.2f} · {2650 / 328:.2f} años", "0,5 · 2,4 · 8,1", ok),
        ("Retornos de los paquetes", f"{775 / 576:.2f} · {3340 / 793:.2f} años", "1,35 · 4,2", ok),
        ("Reducción del desembolso inicial", f"{(1 - 3340 / 5200) * 100:.0f} % (independencia) · {(1 - 775 / 5200) * 100:.0f} % (sin litio)", "«hasta el 50 %»", ok),
        ("Autoconsumo de partida", "30 % (métricas, §2) frente a 35 % (§6.2)", "mismo dato", mal),
    ]
    ancho(st.dataframe, pd.DataFrame(chk, columns=["Comprobación", "Cálculo", "El PDF dice", "Resultado"]), hide_index=True)

# ===================================== TAB 5 =========================================
with tab5:
    st.subheader("Cómo se fusionan SmartStorage 360 y NEXA")
    fus = [
        ("Usuario", "Propietario con placas que regala excedentes", "Vivienda que quiere ahorrar (sin placas, con obra o presupuesto limitado)",
         "Eliges tu tipo de usuario al empezar; cada uno carga su cuestionario y sus valores de ejemplo."),
        ("Preguntas de entrada", "A: equipamiento · B: ocupación 11:00–16:00", "Tipo de vivienda, m², CCAA, presupuesto, gasto mensual, presencia, obra",
         "Comunes: CCAA, equipamiento, ocupación y presupuesto. Solo NEXA: tipo, m² y obra. Solo SmartStorage: kWp y kWh. "
         "El consumo se da en kWh o en €/mes."),
        ("Motor", "Balance horario de 24 h con inercia térmica y tope de compensación", "Constantes fijas de % de ahorro",
         "Un único motor horario. Los % de NEXA pasan a reducir la demanda térmica; el kit y el tejado entran como producción solar."),
        ("Soluciones", "5 niveles de almacenamiento", "Kit solar, FV de tejado, aislamiento, VMC, ventanas",
         "Catálogo unificado de 10 medidas; el almacenamiento solo se activa si hay placas (existentes o del plan)."),
        ("Presupuesto", "Escalera modular con payback", "Selección de medidas dentro del presupuesto máximo",
         "El optimizador evalúa todas las combinaciones, filtra por presupuesto y propone: retorno más rápido, equilibrada y mayor ahorro."),
        ("Salida", "Recomendación por perfil + comparativa con batería de 10 kWh", "Informe de texto con certificados e instalación",
         "Botón «Generar Presupuesto Personalizado»: desglose, ahorro, certificados, gráfico frente a la solución convencional y descarga en HTML."),
    ]
    ancho(st.dataframe, pd.DataFrame(fus, columns=["", "SmartStorage 360 (PDF)", "NEXA (aplicación de escritorio)", "Aplicación fusionada"]),
          hide_index=True)
    st.caption("Mejoras sobre NEXA: ahorros calculados con física (no porcentajes fijos), restricciones coherentes para alquiler y "
               "sin obra, y deducción fiscal editable (por defecto 0 %) en lugar de un porcentaje fijo.")
