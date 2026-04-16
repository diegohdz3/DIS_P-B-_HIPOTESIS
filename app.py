import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(page_title="App Estadística", layout="wide")
st.title("📊 Análisis Estadístico con Prueba Z")
st.markdown("---")

st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", [
    "🏠 Inicio",
    "📂 Carga de Datos",
    "📈 Visualización",
    "🧪 Prueba Z",
    "🤖 Asistente IA"
])

if pagina == "🏠 Inicio":
    st.header("Bienvenido")
    st.write("Esta app permite visualizar distribuciones, realizar pruebas de hipótesis y consultar un asistente de IA.")

elif pagina == "📂 Carga de Datos":
    st.header("📂 Carga de Datos")

    opcion = st.radio("¿Cómo quieres cargar los datos?", ["Subir CSV", "Generar datos sintéticos"])

    if opcion == "Subir CSV":
        archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
        if archivo:
            df = pd.read_csv(archivo)
            st.success("Archivo cargado correctamente")
            st.dataframe(df.head())
            columna = st.selectbox("Selecciona la variable a analizar", df.columns)
            st.session_state["datos"] = df[columna].dropna().values
            st.session_state["nombre_variable"] = columna
            st.info(f"Variable seleccionada: **{columna}** — {len(st.session_state['datos'])} datos")

    elif opcion == "Generar datos sintéticos":
        tipo = st.selectbox("Tipo de distribución", ["Normal", "Sesgada a la derecha", "Sesgada a la izquierda"])
        n = st.slider("Número de datos (n)", 30, 500, 100)

        if st.button("Generar datos"):
            if tipo == "Normal":
                datos = np.random.normal(loc=50, scale=10, size=n)
            elif tipo == "Sesgada a la derecha":
                datos = np.random.exponential(scale=10, size=n) + 40
            elif tipo == "Sesgada a la izquierda":
                datos = 100 - np.random.exponential(scale=10, size=n)

            st.session_state["datos"] = datos
            st.session_state["nombre_variable"] = tipo
            st.success(f"Datos generados: {n} observaciones")
            st.dataframe(pd.DataFrame(datos, columns=[tipo]).head(10))

elif pagina == "📈 Visualización":
    st.header("📈 Visualización de Distribución")

    if "datos" not in st.session_state:
        st.warning("Primero carga o genera datos en la sección 'Carga de Datos'")
    else:
        datos = st.session_state["datos"]
        nombre = st.session_state["nombre_variable"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Media", f"{np.mean(datos):.2f}")
        col2.metric("Desv. Estándar", f"{np.std(datos):.2f}")
        col3.metric("N", len(datos))

        col4, col5, col6 = st.columns(3)
        col4.metric("Mínimo", f"{np.min(datos):.2f}")
        col5.metric("Máximo", f"{np.max(datos):.2f}")
        col6.metric("Sesgo", f"{stats.skew(datos):.2f}")

        st.markdown("---")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].hist(datos, bins=20, color="steelblue", edgecolor="white", density=True, alpha=0.7)
        xmin, xmax = axes[0].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        kde = stats.gaussian_kde(datos)
        axes[0].plot(x, kde(x), color="red", linewidth=2)
        axes[0].set_title("Histograma + KDE")
        axes[0].set_xlabel(nombre)

        axes[1].boxplot(datos, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="steelblue", color="navy"))
        axes[1].set_title("Boxplot")
        axes[1].set_ylabel(nombre)

        stats.probplot(datos, dist="norm", plot=axes[2])
        axes[2].set_title("QQ Plot (Normalidad)")

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("🔍 Análisis Automático")

        sesgo = stats.skew(datos)
        stat_norm, p_norm = stats.shapiro(datos[:50] if len(datos) > 50 else datos)

        if p_norm > 0.05:
            st.success("✅ La distribución **parece normal** (Shapiro-Wilk p > 0.05)")
        else:
            st.error("❌ La distribución **no parece normal** (Shapiro-Wilk p ≤ 0.05)")

        if abs(sesgo) < 0.5:
            st.info("📊 **Sin sesgo significativo**")
        elif sesgo > 0.5:
            st.warning("📊 **Sesgo positivo** (cola a la derecha)")
        else:
            st.warning("📊 **Sesgo negativo** (cola a la izquierda)")

        q1 = np.percentile(datos, 25)
        q3 = np.percentile(datos, 75)
        iqr = q3 - q1
        outliers = np.sum((datos < q1 - 1.5*iqr) | (datos > q3 + 1.5*iqr))
        if outliers > 0:
            st.warning(f"⚠️ Se detectaron **{outliers} outliers**")
        else:
            st.success("✅ **No se detectaron outliers**")

elif pagina == "🧪 Prueba Z":
    st.header("🧪 Prueba de Hipótesis Z")
    st.write("Realiza una prueba Z para la media de una población cuando se conoce la varianza poblacional.")

    # 1. Inputs del usuario
    st.subheader("1. Configuración de la Prueba")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Parámetros de la Hipótesis**")
        mu0 = st.number_input("Valor bajo H0 (μ0)", value=0.0, step=0.1)
        sigma = st.number_input("Desviación estándar poblacional (σ)", value=1.0, min_value=0.0001)
        alpha = st.selectbox("Nivel de significancia (α)", [0.01, 0.05, 0.10], index=1)
        tipo_cola = st.selectbox("Tipo de prueba (H1)", [
            "Bilateral (≠)",
            "Unilateral Derecha (>)",
            "Unilateral Izquierda (<)"
        ])

    with col2:
        st.markdown("**Datos de la Muestra**")
        fuente_datos = st.radio("¿Qué datos deseas utilizar?",
                                ["Usar datos cargados en sesión", "Ingresar datos manualmente"])

        if fuente_datos == "Usar datos cargados en sesión":
            if "datos" in st.session_state:
                datos = st.session_state["datos"]
                n = len(datos)
                x_bar = np.mean(datos)
                st.success(f"Datos detectados: **n = {n}**, **Media muestral = {x_bar:.4f}**")
            else:
                st.warning("⚠️ No hay datos en memoria. Ve a 'Carga de Datos' o selecciona ingreso manual.")
                st.stop()
        else:
            n = st.number_input("Tamaño de la muestra (n)", min_value=1, value=30, step=1)
            x_bar = st.number_input("Media muestral calculada", value=0.0, step=0.1)

    # Botón para ejecutar
    if st.button("🚀 Calcular Prueba Z", type="primary"):
        st.markdown("---")
        st.subheader("2. Resultados y Decisión")

        # 2. Cálculo del estadístico Z y p-value
        se = sigma / np.sqrt(n)  # Error estándar
        z_stat = (x_bar - mu0) / se

        # P-value y valores críticos según el tipo de cola
        if tipo_cola == "Bilateral (≠)":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            z_crit_inf = stats.norm.ppf(alpha / 2)
            z_crit_sup = stats.norm.ppf(1 - alpha / 2)
        elif tipo_cola == "Unilateral Derecha (>)":
            p_value = 1 - stats.norm.cdf(z_stat)
            z_crit_inf = None
            z_crit_sup = stats.norm.ppf(1 - alpha)
        else:  # Unilateral Izquierda (<)
            p_value = stats.norm.cdf(z_stat)
            z_crit_inf = stats.norm.ppf(alpha)
            z_crit_sup = None

        # Mostrar métricas
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Estadístico Z", f"{z_stat:.4f}")
        col_res2.metric("P-Value", f"{p_value:.4e}" if p_value < 0.001 else f"{p_value:.4f}")
        col_res3.metric("Nivel de Significancia (α)", f"{alpha}")

        # 3. Decisión automática
        if p_value < alpha:
            st.error(f"🚨 **DECISIÓN:** Se **RECHAZA** la Hipótesis Nula (H0).")
            st.write("Existe evidencia estadística suficiente para respaldar la Hipótesis Alternativa (H1).")
        else:
            st.success(f"✅ **DECISIÓN:** **NO SE RECHAZA** la Hipótesis Nula (H0).")
            st.write("No hay evidencia estadística suficiente para rechazar H0.")

        st.markdown("---")
        st.subheader("3. Gráfica de la Zona de Rechazo")

        # 4. Gráfica de la curva normal con zona de rechazo
        fig, ax = plt.subplots(figsize=(10, 5))

        # Generar distribución normal estándar
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x, 0, 1)
        ax.plot(x, y, color='black', linewidth=1.5)

        # Colorear zonas de rechazo según H1
        if tipo_cola == "Bilateral (≠)":
            x_fill_left = np.linspace(-4, z_crit_inf, 100)
            x_fill_right = np.linspace(z_crit_sup, 4, 100)
            ax.fill_between(x_fill_left, stats.norm.pdf(x_fill_left, 0, 1), color='red', alpha=0.5,
                            label='Zona de Rechazo')
            ax.fill_between(x_fill_right, stats.norm.pdf(x_fill_right, 0, 1), color='red', alpha=0.5)
            ax.axvline(z_crit_inf, color='darkred', linestyle='--', label=f'Z Crítico ({z_crit_inf:.2f})')
            ax.axvline(z_crit_sup, color='darkred', linestyle='--')

        elif tipo_cola == "Unilateral Derecha (>)":
            x_fill_right = np.linspace(z_crit_sup, 4, 100)
            ax.fill_between(x_fill_right, stats.norm.pdf(x_fill_right, 0, 1), color='red', alpha=0.5,
                            label='Zona de Rechazo')
            ax.axvline(z_crit_sup, color='darkred', linestyle='--', label=f'Z Crítico ({z_crit_sup:.2f})')

        else:  # Unilateral Izquierda
            x_fill_left = np.linspace(-4, z_crit_inf, 100)
            ax.fill_between(x_fill_left, stats.norm.pdf(x_fill_left, 0, 1), color='red', alpha=0.5,
                            label='Zona de Rechazo')
            ax.axvline(z_crit_inf, color='darkred', linestyle='--', label=f'Z Crítico ({z_crit_inf:.2f})')

        # Dibujar el Z calculado
        ax.axvline(z_stat, color='dodgerblue', linewidth=2.5, label=f'Z Calculado ({z_stat:.2f})')

        # Formateo de la gráfica
        ax.set_title("Distribución Normal Estándar - Prueba Z", fontsize=14)
        ax.set_xlabel("Z", fontsize=12)
        ax.set_ylabel("Densidad", fontsize=12)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        st.pyplot(fig)