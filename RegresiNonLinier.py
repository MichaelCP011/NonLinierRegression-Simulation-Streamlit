import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# ==============================================================================
# Fungsi-fungsi Perhitungan Model
# ==============================================================================

def calculate_exponential_model(x_data, y_data):
    """Menghitung model regresi eksponensial y = C * e^(b*x)"""
    try:
        if np.any(y_data <= 0):
            st.warning("Model Eksponensial: Melewatkan perhitungan karena ada nilai y <= 0.", icon="⚠️")
            return None, float('inf'), None, None, None
            
        Y = np.log(y_data)
        X = x_data
        b, a_lnC = np.polyfit(X, Y, 1)
        C = np.exp(a_lnC)
        
        y_pred = C * np.exp(b * x_data)
        error_sq = (y_data - y_pred)**2
        sse = np.sum(error_sq)
        
        df_calc = pd.DataFrame({
            'x': x_data,
            'y_asli': y_data,
            'Y = ln(y)': Y,
            'y_pred_exp': y_pred,
            'Error^2': error_sq
        })
        
        equation = f"y = {C:.4f} \\cdot e^({{{b:.4f}x}})"
        coefficients = {'C': C, 'b': b}
        return equation, sse, y_pred, df_calc, coefficients
    except Exception as e:
        return None, float('inf'), None, f"Error: {e}", None

def calculate_power_model(x_data, y_data):
    """Menghitung model regresi pangkat y = C * x^b"""
    try:
        fit_indices = np.where((x_data > 0) & (y_data > 0))
        if len(fit_indices[0]) < 2:
            st.warning("Model Pangkat: Data tidak cukup (perlu > 1 titik dengan x > 0 dan y > 0).", icon="⚠️")
            return None, float('inf'), None, None, None

        x_fit = x_data[fit_indices]
        y_fit = y_data[fit_indices]
        
        Y = np.log(y_fit)
        X = np.log(x_fit)
        
        b, a_lnC = np.polyfit(X, Y, 1)
        C = np.exp(a_lnC)
        
        y_pred = C * np.power(x_data.astype(float), b)
        error_sq = (y_data - y_pred)**2
        sse = np.sum(error_sq)

        df_calc = pd.DataFrame({
            'x': x_data,
            'y_asli': y_data,
            'y_pred_pangkat': y_pred,
            'Error^2': error_sq
        })
        
        equation = f"y = {C:.4f} \\cdot x^({{{b:.4f}}})"
        coefficients = {'C': C, 'b': b}
        return equation, sse, y_pred, df_calc, coefficients
    except Exception as e:
        return None, float('inf'), None, f"Error: {e}", None

def calculate_saturation_model(x_data, y_data):
    """Menghitung model laju pertumbuhan jenuh y = C*x / (d+x)"""
    try:
        fit_indices = np.where((x_data > 0) & (y_data > 0))
        if len(fit_indices[0]) < 2:
            st.warning("Model Saturasi: Data tidak cukup (perlu > 1 titik dengan x > 0 dan y > 0).", icon="⚠️")
            return None, float('inf'), None, None, None

        x_fit = x_data[fit_indices]
        y_fit = y_data[fit_indices]

        Y = 1 / y_fit
        X = 1 / x_fit
        
        b_d_div_C, a_1_div_C = np.polyfit(X, Y, 1)
        
        if a_1_div_C == 0:
            return None, float('inf'), None, "Error: Perhitungan gagal (division by zero).", None
            
        C = 1 / a_1_div_C
        d = b_d_div_C * C

        y_pred = np.zeros_like(x_data, dtype=float)
        non_zero_indices = x_data > 0
        y_pred[non_zero_indices] = (C * x_data[non_zero_indices]) / (d + x_data[non_zero_indices])
        
        error_sq = (y_data - y_pred)**2
        sse = np.sum(error_sq)
        
        df_calc = pd.DataFrame({
            'x': x_data,
            'y_asli': y_data,
            'y_pred_saturasi': y_pred,
            'Error^2': error_sq
        })
        
        equation = f"y = \\frac{{{C:.4f} \\cdot x}}{{{d:.4f} + x}}"
        coefficients = {'C': C, 'd': d}
        return equation, sse, y_pred, df_calc, coefficients
    except Exception as e:
        return None, float('inf'), None, f"Error: {e}", None

# ==============================================================================
# Antarmuka Streamlit (UI)
# ==============================================================================

st.set_page_config(layout="wide", page_title="Simulasi Regresi Non-Linier")

st.title("📊 Simulasi Regresi Non-Linier")
st.write("Copyright: Kelompok 4 Metode Numerik (A) [RRM]")
st.sidebar.header("Input Data")

input_method = st.sidebar.radio("Pilih metode input data:", ("Input Manual", "Upload File CSV"))

x_data, y_data = None, None

if input_method == "Input Manual":
    st.sidebar.subheader("Masukkan Data Secara Manual")
    x_input = st.sidebar.text_area("Masukkan nilai x (pisahkan dengan koma)", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
    y_input = st.sidebar.text_area("Masukkan nilai y (pisahkan dengan koma)", "1, 5, 10, 16, 22, 29, 37, 45, 54, 63")
    
    if st.sidebar.button("Proses Data"):
        try:
            x_data = np.array([float(i.strip()) for i in x_input.split(',')])
            y_data = np.array([float(i.strip()) for i in y_input.split(',')])
            if len(x_data) != len(y_data):
                st.sidebar.error("Jumlah data x dan y harus sama!")
                x_data, y_data = None, None
        except ValueError:
            st.sidebar.error("Pastikan semua input adalah angka dan dipisahkan dengan koma.")
            x_data, y_data = None, None

elif input_method == "Upload File CSV":
    st.sidebar.subheader("Unggah File CSV")
    uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type="csv")
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            if 'x' in df_input.columns and 'y' in df_input.columns:
                x_data = df_input['x'].values
                y_data = df_input['y'].values
            else:
                st.sidebar.error("File CSV harus memiliki kolom bernama 'x' dan 'y'.")
        except Exception as e:
            st.sidebar.error(f"Gagal memproses file: {e}")

# --- Area Utama untuk Menampilkan Hasil ---
if x_data is not None and y_data is not None:
    st.header("1. Visualisasi Data Asli")
    df_asli = pd.DataFrame({'x': x_data, 'y': y_data})
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tabel Data")
        st.dataframe(df_asli)
    with col2:
        st.subheader("Plot Sebar Data")
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
        ax_scatter.scatter(x_data, y_data, color='red', label='Data Asli')
        ax_scatter.set_title('Plot Data Asli')
        ax_scatter.set_xlabel('Nilai x')
        ax_scatter.set_ylabel('Nilai y')
        ax_scatter.grid(True)
        ax_scatter.legend()
        st.pyplot(fig_scatter)
        
    st.markdown("---")
    st.header("2. Analisis Model Regresi")
    
    eq_exp, sse_exp, y_pred_exp, df_exp, coeffs_exp = calculate_exponential_model(x_data, y_data)
    eq_pow, sse_pow, y_pred_pow, df_pow, coeffs_pow = calculate_power_model(x_data, y_data)
    eq_sat, sse_sat, y_pred_sat, df_sat, coeffs_sat = calculate_saturation_model(x_data, y_data)
    
    sse_results = {
        'Model Eksponensial': sse_exp,
        'Model Pangkat Sederhana': sse_pow,
        'Model Laju Pertumbuhan Jenuh': sse_sat
    }
    
    with st.expander("Model 1: Regresi Eksponensial", expanded=True):
        if eq_exp:
            st.subheader("Hasil Model Eksponensial")
            st.latex(eq_exp)
            st.metric(label="Total Sum of Squared Errors (SSE)", value=f"{sse_exp:.4f}")
            st.write("Tabel Perhitungan:")
            st.dataframe(df_exp.style.format("{:.4f}"))
        else:
            st.error(df_exp or "Perhitungan tidak dapat dilakukan.")

    with st.expander("Model 2: Regresi Pangkat Sederhana", expanded=True):
        if eq_pow:
            st.subheader("Hasil Model Pangkat Sederhana")
            st.latex(eq_pow)
            st.metric(label="Total Sum of Squared Errors (SSE)", value=f"{sse_pow:.4f}")
            st.write("Tabel Perhitungan:")
            st.dataframe(df_pow.style.format("{:.4f}"))
        else:
            st.error(df_pow or "Perhitungan tidak dapat dilakukan.")

    with st.expander("Model 3: Regresi Laju Pertumbuhan Jenuh", expanded=True):
        if eq_sat:
            st.subheader("Hasil Model Laju Pertumbuhan Jenuh")
            st.latex(eq_sat)
            st.metric(label="Total Sum of Squared Errors (SSE)", value=f"{sse_sat:.4f}")
            st.write("Tabel Perhitungan:")
            st.dataframe(df_sat.style.format("{:.4f}"))
        else:
            st.error(df_sat or "Perhitungan tidak dapat dilakukan.")
            
    st.markdown("---")
    st.header("3. Perbandingan dan Kesimpulan")
    
    col3, col4 = st.columns([1, 2])
    
    with col3:
        st.subheader("Perbandingan SSE")
        valid_sse_results = {k: v for k, v in sse_results.items() if v != float('inf')}
        if valid_sse_results:
            df_sse = pd.DataFrame(list(valid_sse_results.items()), columns=['Model', 'SSE']).set_index('Model')
            st.dataframe(df_sse.style.format("{:.4f}"))

            best_model_name = min(valid_sse_results, key=valid_sse_results.get)
            st.success(f"🏆 Model Terbaik: **{best_model_name}**")
            st.write(f"Model ini memiliki Sum of Squared Errors (SSE) terkecil, yang menunjukkan kecocokan terbaik dengan data.")
        else:
            st.error("Semua model gagal dihitung. Tidak ada perbandingan.")

    with col4:
        st.subheader("Plot Perbandingan Model")
        fig_final, ax_final = plt.subplots(figsize=(10, 6))
        
        ax_final.scatter(x_data, y_data, color='red', label='Data Asli', s=50, zorder=5)

        x_smooth = np.linspace(min(x_data), max(x_data), 200)
        
        if coeffs_exp:
            y_smooth_exp = coeffs_exp['C'] * np.exp(coeffs_exp['b'] * x_smooth)
            ax_final.plot(x_smooth, y_smooth_exp, label=f'Eksponensial (SSE={sse_exp:.2f})', color='blue')
        
        if coeffs_pow:
            y_smooth_pow = coeffs_pow['C'] * np.power(x_smooth.astype(float), coeffs_pow['b'])
            ax_final.plot(x_smooth, y_smooth_pow, label=f'Pangkat (SSE={sse_pow:.2f})', color='green')
        
        if coeffs_sat:
            with np.errstate(divide='ignore', invalid='ignore'):
                 y_smooth_sat = (coeffs_sat['C'] * x_smooth) / (coeffs_sat['d'] + x_smooth)
            ax_final.plot(x_smooth, y_smooth_sat, label=f'Saturasi (SSE={sse_sat:.2f})', color='purple')

        ax_final.set_title('Perbandingan Model Regresi Non-Linier')
        ax_final.set_xlabel('Nilai x')
        ax_final.set_ylabel('Nilai y')
        ax_final.legend()
        ax_final.grid(True)
        ax_final.set_ylim(bottom=0)
        st.pyplot(fig_final)

else:
    st.info("Silakan masukkan data di panel samping untuk memulai analisis.")