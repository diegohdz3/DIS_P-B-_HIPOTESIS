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