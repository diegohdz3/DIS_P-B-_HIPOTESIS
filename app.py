import streamlit as st
import pandas as pd
import numpy as np

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