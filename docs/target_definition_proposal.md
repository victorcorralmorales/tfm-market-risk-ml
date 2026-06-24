# Propuesta de definición del target de riesgo de mercado

**TFM:** Máster en Inteligencia Artificial — estimación de riesgo de mercado con Machine Learning  
**Fase:** definición de la variable objetivo (exploratoria; sin implementación en el dataset final)  
**Dataset de referencia:** `data/processed/initial_market_dataset.csv`  
**Fecha del análisis:** mayo 2026  

---

## 1. Contexto del problema

El objetivo del TFM **no es predecir precios ni retornos** del S&P 500, sino **anticipar escenarios futuros de riesgo** en el mercado estadounidense, usando el índice como referencia principal.

La modalidad prevista es **comparativa de soluciones** (por ejemplo, distintos modelos o enfoques de ML frente a un mismo target bien definido). Para ello hace falta una **variable objetivo (target)** clara, medible y alineada con la idea de “riesgo”, no con “subida o bajada puntual del precio”.

Ya disponemos de un panel diario con variables de mercado (S&P 500), volatilidad implícita (VIX), tipos de interés, curva, volatilidades rolling, drawdown actual, medias móviles y cambios recientes. El siguiente paso es decidir **qué evento futuro** queremos que el modelo aprenda a anticipar.

---

## 2. Qué es un target en Machine Learning supervisado

En aprendizaje supervisado, cada fila del dataset tiene:

- **Features (X):** información conocida **hasta el día t** (precios pasados, VIX, tipos, etc.).
- **Target (y):** lo que queremos **predecir** para ese día.

El modelo aprende una función aproximada \( f(X_t) \approx y_t \). Por tanto, el target debe:

1. **Tener sentido económico** (riesgo, no solo ruido diario).
2. **Ser observable en el histórico** (para entrenar con ejemplos reales).
3. **Mirar al futuro** respecto a t, porque en producción queremos anticipar lo que **aún no ha ocurrido**.

En este TFM, \( y_t \) representará algo como: *“¿habrá un episodio de riesgo elevado en los próximos N días hábiles?”*.

---

## 3. Por qué el target debe mirar al futuro

Las features del día t solo pueden usar información **disponible al cierre de t** (o antes). Si el target usara solo el pasado, estaríamos describiendo el presente o el pasado, no **anticipando** riesgo.

El target **sí puede y debe** usar datos de \( t+1, t+2, \ldots \) porque define el **evento que queremos predecir**. Eso no es “trampa” en el entrenamiento: es la definición del problema. La trampa (fuga de información o *leakage*) aparece si:

- calculamos features con datos futuros;
- fijamos umbrales (p. ej. percentiles de volatilidad) usando **todo el histórico incluido el periodo de test**, sin separar train/validación.

Esto se detalla en la sección 8.

---

## 4. Definiciones operativas usadas en el análisis exploratorio

Horizonte: **20 días hábiles** (mismo orden de magnitud que un mes de trading).

Para cada día \( t \) (con al menos 20 días posteriores en la muestra):

| Magnitud | Definición |
|----------|------------|
| `future_drawdown_20d` | \( \min(P_{t+1},\ldots,P_{t+20}) / P_t - 1 \) — peor caída desde el cierre de hoy mirando solo los **próximos** 20 cierres. |
| `future_vol_20d_ann` | Desviación típica de los **20 retornos diarios** \( P_{t+k}/P_{t+k-1}-1 \), \( k=1,\ldots,20 \), multiplicada por \( \sqrt{252} \) (anualizada). |

Los targets binarios explorados se construyen a partir de estas magnitudes. **No se han guardado** en `initial_market_dataset.csv`.

---

## 5. Opciones analizadas

### 5.A Target basado solo en drawdown futuro

**Idea:** `target = 1` si en los próximos 20 días el S&P cae al menos un X% desde el cierre de hoy (equivalente a `future_drawdown_20d <= -X`).

| Umbral | Ventajas | Inconvenientes |
|--------|----------|----------------|
| **3%** | Muchos eventos; sensible a correcciones cortas | Muchos falsos positivos de “riesgo”; mezcla volatilidad normal y estrés |
| **5%** | Interpretación clara (“corrección” en ~1 mes); equilibrio razonable en la muestra | No captura solo “nerviosismo” sin caída fuerte |
| **7,5%** | Eventos más ligados a estrés real | Menos positivos; clases más desbalanceadas |
| **10%** | Episodios muy claros (crisis) | Muy pocos eventos; difícil entrenar y evaluar |

**Resultados exploratorios** (9 139 días con futuro calculable; 1990–2026):

| Definición | Eventos positivos | % positivos | Comentario |
|------------|-------------------|-------------|------------|
| Drawdown ≤ −3% (20d) | 2 904 | 31,8% | Posiblemente **laxa** |
| Drawdown ≤ −5% (20d) | 1 472 | 16,1% | **Razonable** |
| Drawdown ≤ −7,5% (20d) | 702 | 7,7% | **Razonable** (más estricta) |
| Drawdown ≤ −10% (20d) | 341 | 3,7% | Posiblemente **estricta** |

---

### 5.B Target basado solo en volatilidad futura realizada

**Idea:** `target = 1` si `future_vol_20d_ann` supera un percentil alto de su distribución histórica (en el análisis exploratorio, percentil calculado sobre **toda la muestra**; ver sección 8).

| Umbral | Ventajas | Inconvenientes |
|--------|----------|----------------|
| **Percentil 80** | ~20% de positivos por construcción; captura “meses turbulentos” aunque no haya crash | No distingue caída fuerte vs. alta vol lateral; umbral depende de la muestra |
| **Percentil 90** | Episodios más extremos; menos positivos | Más desbalance; puede perder alertas tempranas |

**Umbrales en la muestra completa:** P80 ≈ 0,200 (20% anualizado), P90 ≈ 0,252.

| Definición | Eventos positivos | % positivos | Comentario |
|------------|-------------------|-------------|------------|
| Vol futura > P80 | 1 828 | 20,0% | Por construcción ~20% (exploración in-sample) |
| Vol futura > P90 | 914 | 10,0% | Por construcción ~10% |

---

### 5.C Target combinado (drawdown OR volatilidad)

**Idea:** `target = 1` si se cumple **al menos una**:

1. `future_drawdown_20d <= -5%`
2. `future_vol_20d_ann` > percentil 80 (histórico)

| Ventajas | Inconvenientes |
|----------|----------------|
| El riesgo no es solo “precio abajo”: puede haber **alta volatilidad** sin caída del 5% (mercado inquieto, gaps, recuperaciones en V) | Definición más compleja de explicar y de calibrar |
| Alineado con el TFM (anticipar **escenarios** de riesgo) | Mayor tasa de positivos que solo drawdown 5% si se usa OR |
| Complementa drawdown con régimen de volatilidad | Hay que fijar bien el percentil sin fuga (train only) |

**Resultado exploratorio (misma muestra, P80 global):**

| Definición | Eventos positivos | % positivos | Comentario |
|------------|-------------------|-------------|------------|
| **Combinado: DD ≤ −5% OR vol > P80** | 2 306 | **25,2%** | **Razonable** para una primera versión |

Aproximadamente **834** días positivos por volatilidad sin cumplir drawdown 5% (diferencia entre combinado y solo drawdown), lo que justifica el componente de volatilidad.

---

### 5.D Target multiclase (posibilidad)

**Ejemplo:** 0 = riesgo bajo, 1 = moderado, 2 = alto (según reglas sobre drawdown y volatilidad).

| Ventajas | Inconvenientes |
|----------|----------------|
| Matices (“alerta” vs “crisis”) | Más difícil de definir, validar y explicar en la memoria |
| Útil en fases avanzadas | Métricas y comparativa de modelos más complejas |
| | Requiere más datos por clase |

**Recomendación para la primera versión del TFM:** **no** usar multiclase todavía. Empezar con **binario** clarifica el problema, facilita la comparativa de soluciones (AUC, F1, precisión/recall) y reduce decisiones arbitrarias de fronteras entre clases.

---

## 6. Validación visual (periodos de referencia)

En el notebook `notebooks/02_target_definition_exploration.ipynb` se superponen:

- nivel del S&P 500 con sombreado cuando el **target combinado exploratorio** vale 1;
- series de `future_drawdown_20d` y `future_vol_20d_ann`.

**Lectura cualitativa esperada:**

| Periodo | ¿Debería activarse con frecuencia? | Comentario |
|---------|-----------------------------------|------------|
| **2000–2002** (burbuja dotcom) | Sí | Drawdowns y volatilidad elevada encajan bien |
| **2008–2009** (crisis financiera) | Sí | Muchos días positivos consecutivos |
| **2020** (COVID) | Sí | Caída brusca y volatilidad extrema |
| **2022** (subida de tipos) | Parcial | Drawdown moderado pero vol elevada; el componente de **vol** ayuda frente a solo drawdown |

El target combinado suele **cubrir mejor** episodios de “estrés” que no siempre alcanzan −5% en exactamente 20 días, pero puede marcar también tramos cortos muy volátiles en mercados laterales (trade-off aceptable si se documenta).

---

## 7. Fuga de información (*leakage*)

| Elemento | ¿Puede usar el futuro? | Notas |
|----------|------------------------|-------|
| **Features en día t** | **No** | Solo datos ≤ t (como en el CSV actual: retornos pasados, VIX, tipos, etc.) |
| **Target en día t** | **Sí** | Se define con \( t+1 \ldots t+20 \); es lo que queremos predecir |
| **Percentil de volatilidad** | Cuidado | En exploración se usó el percentil sobre **toda la serie** (válido para decidir definición, **no** para evaluación final) |
| **Implementación recomendada** | Train only | Calcular P80 (u otro umbral) solo con **train**; aplicar el mismo umbral a validación y test. Alternativa: percentil **expanding** usando solo pasado (más realista, más complejo) |

En la memoria del TFM conviene **justificar explícitamente** cómo se fijó el umbral de volatilidad y mostrar que no se usó información del conjunto de test para definirlo.

---

## 8. Recomendación final

### Propuesta adoptada para la siguiente fase (pendiente de implementar en CSV)

**Target binario combinado a 20 días — `target_risk_20d`:**

```text
target_risk_20d = 1  si en los próximos 20 días hábiles ocurre AL MENOS UNA de:
  (a) future_drawdown_20d <= -5%
  (b) future_vol_20d_ann > percentil 80 (calculado solo en TRAIN al implementar)

target_risk_20d = 0  en caso contrario
```

### Justificación

1. **Alineación con el TFM:** mide **riesgo futuro** (caída relevante **o** régimen de alta volatilidad realizada), no dirección del mercado.
2. **Interpretabilidad:** −5% en ~1 mes es un umbral estándar en narrativa de “corrección”; la volatilidad realizada complementa episodios de estrés sin crash del 5%.
3. **Balance de clases:** ~25% de positivos en exploración global es **entrenable** (mejor que 3,7% del −10% o 32% del −3%).
4. **Comparativa de soluciones:** binario facilita comparar modelos con métricas homogéneas.
5. **Evaluación crítica de la preferencia inicial:** la definición propuesta por el autor es **razonable**; no proponemos cambiarla por otra radicalmente distinta, sino **acotar la implementación** (percentil en train, horizonte 20d fijo, documentar definición de vol).

### Alternativas si en implementación aparecen problemas

| Situación | Ajuste posible |
|-----------|----------------|
| Demasiados positivos en periodos recientes | Subir umbral de drawdown a −7,5% o usar P90 en vol |
| Demasiados pocos positivos | Mantener −5% pero bajar vol a P80 solo en train (revisar balance por año) |
| Memoria pide simplicidad máxima | Solo drawdown −5% (~16% positivos), a costa de perder episodios solo-vol |

---

## 9. Decisiones abiertas antes de implementar el target definitivo

1. **Percentil de volatilidad:** ¿fijo en train, expanding histórico o ventana móvil de 5–10 años?
2. **Horizonte:** ¿mantener 20 días o probar 10 / 21 como robustez?
3. **Filas finales:** ¿eliminar las últimas 20 filas (sin target) al exportar el dataset con target?
4. **Métricas:** ¿priorizar recall (detectar crisis) o precisión (menos falsas alarmas)?
5. **Etiquetas en crisis largas:** muchos días consecutivos con `y=1` — ¿modelar como clasificación i.i.d. o considerar agrupación temporal en validación?

---

## 10. Próximos pasos (fuera de este documento)

1. Notebook exploratorio: `notebooks/02_target_definition_exploration.ipynb` (cálculos temporales, tablas y gráficos).
2. Tras validar con el director del TFM: nuevo notebook o script que genere `data/processed/...` **con target**, sin sobrescribir `initial_market_dataset.csv`.
3. Diseño de partición temporal train/validation/test y protocolo para el percentil P80.

---

*Documento generado en fase exploratoria. Los porcentajes provienen de `initial_market_dataset.csv` (9 159 filas; 9 139 con horizonte futuro completo).*
