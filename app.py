import streamlit as st

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