import pickle
import streamlit as st
import pandas as pd
from PIL import Image
import tensorflow as tf

# Memuat model
model_file = 'model.h5'
model = tf.keras.models.load_model(model_file)

# Memuat transformer
dv_file = 'transformer.pkl'
with open(dv_file, 'rb') as f:
    transformer = pickle.load(f)

def preprocess_input(data):
    yes_no_mapping = {"yes": 1, "no": 0}
    no_service_mapping = {"no internet service": 0, "no phone service": 0}
    
    # Daftar semua kolom yang diharapkan
    expected_columns = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
                        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                        "MonthlyCharges", "TotalCharges", "tenure"]
    
    # Memeriksa apakah input adalah DataFrame (batch) atau kamus (input tunggal)
    if isinstance(data, dict):
        for key in data:
            if data[key] in yes_no_mapping:
                data[key] = yes_no_mapping[data[key]]
            elif data[key] in no_service_mapping:
                data[key] = no_service_mapping[data[key]]
        # Menambahkan kolom yang hilang dengan nilai default
        for col in expected_columns:
            if col not in data:
                data[col] = 0  # Nilai default, bisa disesuaikan
        return pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        for col in data.columns:
            if data[col].iloc[0] in yes_no_mapping:
                data[col] = data[col].map(yes_no_mapping)
            elif data[col].iloc[0] in no_service_mapping:
                data[col] = data[col].map(no_service_mapping)
            # Penanganan khusus untuk kolom InternetService
            elif col == "InternetService" and data[col].iloc[0] == "no":
                data[col] = 0
        # Menambahkan kolom yang hilang dengan nilai default
        for col in expected_columns:
            if col not in data.columns:
                data[col] = 0  # Nilai default, bisa disesuaikan
        return data



def main():
    st.set_page_config(page_title="Prediksi Customer Churn", layout="wide")

    # Menyesuaikan gambar untuk ukuran yang lebih kecil
    image = Image.open('images/rilo_bg.jpg').resize((220, 110))
    image2 = Image.open('images/gunadarma_logo.png')
    st.sidebar.image(image, use_column_width=True)
    
    pilihan = st.sidebar.radio("Pilih Menu:", ("Prediksi", "Tambahan"))
    prediksi_selectbox = st.sidebar.selectbox("Pilih Mode Prediksi", ("Online", "Per_Batch"))
    tambahan_selectbox = st.sidebar.selectbox("Pilih Informasi Tambahan", ("Keterangan Input", "Penjelasan Dataset"))
        
    st.sidebar.info('Aplikasi ini dibuat untuk memprediksi customer churn')
    st.sidebar.image(image2, use_column_width=True)
    st.info("Baca bagian tambahan untuk keterangan input dan penjelasan dataset", icon="ℹ️")
    if pilihan == 'Prediksi':
        
        st.title("Prediksi Customer Churn")

        if prediksi_selectbox == 'Online':
            st.write("### Masukkan Detail Pelanggan")
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                gender = st.selectbox('Gender: ', ['male', 'female'])
                seniorcitizen = st.selectbox('Senior Citizen: ', [0, 1])
                partner = st.selectbox('Partner: ', ['yes', 'no'])
                dependents = st.selectbox('Dependents: ', ['yes', 'no'])

            with col2:
                phoneservice = st.selectbox('Phone Service: ', ['yes', 'no'])
                multiplelines = st.selectbox("Multiple Lines: ", ['yes', 'no', 'no phone service'])
                internetservice = st.selectbox('Internet Service: ', ['dsl', '0', 'fiber optic'])
                onlinesecurity = st.selectbox("Online Security: ", ['yes', 'no', 'no internet service'])

            with col3:
                onlinebackup = st.selectbox("Online Backup: ", ['yes', 'no', 'no internet service'])
                deviceprotection = st.selectbox("Device Protection: ", ['yes', 'no', 'no internet service'])
                techsupport = st.selectbox("Tech Support: ", ['yes', 'no', 'no internet service'])
                streamingtv = st.selectbox("Streaming TV: ", ['yes', 'no', 'no internet service'])

            with col4:
                streamingmovies = st.selectbox("Streaming Movies: ", ['yes', 'no', 'no internet service'])
                contract = st.selectbox("Contract: ", ['month-to-month', 'one year', 'two year'])
                paperlessbilling = st.selectbox("Paperless Billing: ", ['yes', 'no'])
                paymentmethod = st.selectbox("Payment Method: ", ['Bank transfer (automatic)', 'Credit card (automatic)', 'electronic check', 'mailed check'])
            
            with col5:
                tenure = st.number_input("Tenure (Months): ", min_value=0, max_value=240, value=0)
                monthlycharges = st.number_input("Monthly Charges: ", min_value=0, max_value=240, value=0)
                totalcharges = tenure * monthlycharges

            input_dict = {
                "gender": gender,
                "SeniorCitizen": seniorcitizen,
                "Partner": partner,
                "Dependents": dependents,
                "PhoneService": phoneservice,
                "MultipleLines": multiplelines,
                "InternetService": internetservice,
                "OnlineSecurity": onlinesecurity,
                "OnlineBackup": onlinebackup,
                "DeviceProtection": deviceprotection,
                "TechSupport": techsupport,
                "StreamingTV": streamingtv,
                "StreamingMovies": streamingmovies,
                "Contract": contract,
                "PaperlessBilling": paperlessbilling,
                "PaymentMethod": paymentmethod,
                "MonthlyCharges": monthlycharges,
                "TotalCharges": totalcharges,
                "tenure": tenure
            }

            if st.button("Prediksi"):
                input_df = preprocess_input(input_dict)
                X = transformer.transform(input_df)
                y_pred_prob = model.predict(X)[0, 0]
                churn = y_pred_prob >= 0.5
                output_prob = float(y_pred_prob)
                output = bool(churn)
                st.success(f'Churn: {output}, Skor Risiko: {output_prob}')

        elif prediksi_selectbox == 'Per_Batch':
            st.write("### Unggah File CSV")
            st.markdown("""
            **Contoh:**
            ```plaintext
            gender,SeniorCitizen,Partner,Dependents,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,Tenure,MonthlyCharges,TotalCharges
            male,0,yes,no,yes,no,dsl,yes,no,yes,no,yes,no,month-to-month,yes,bank_transfer,12,30,360
            female,1,no,yes,no,no_phone_service,no,no_internet_service,no_internet_service,no_internet_service,no_internet_service,no_internet_service,no_internet_service,one_year,no,credit_card,24,50,1200
            ```
            """)

            file_upload = st.file_uploader("Unggah file CSV", type=['csv'])
            if file_upload is not None:
                data = pd.read_csv(file_upload)
                data = data.applymap(lambda s: s.lower() if type(s) == str else s)
                data = preprocess_input(data)
                X = transformer.transform(data)
                y_pred_prob = model.predict(X)[:, 0]
                churn = (y_pred_prob >= 0.5).astype(bool)
                data['Churn'] = churn
                st.write(data)
                
            st.markdown("""
            Format CSV harus sesuai dengan kolom berikut:
            - **gender**
            - **SeniorCitizen**
            - **Partner**
            - **Dependents**
            - **PhoneService**
            - **MultipleLines**
            - **InternetService**
            - **OnlineSecurity**
            - **OnlineBackup**
            - **DeviceProtection**
            - **TechSupport**
            - **StreamingTV**
            - **StreamingMovies**
            - **Contract**
            - **PaperlessBilling**
            - **PaymentMethod**
            - **Tenure**
            - **MonthlyCharges**
            - **TotalCharges**
            """)


    elif pilihan == 'Tambahan':
        
        if tambahan_selectbox == 'Keterangan Input':
            st.header("Keterangan Input")
            st.markdown("""
            - **0** = tidak
            - **1** = iya
            - **no_phone_service** = tidak ada layanan telepon
            - **no_internet_service** = tidak ada layanan internet
            - **month-to-month** = per bulan
            - **one_year** = satu tahun
            - **two_year** = dua tahun
            """)

        elif tambahan_selectbox == 'Penjelasan Dataset':
            st.header("Penjelasan Dataset")
            st.markdown("""
        - **gender**: Jenis kelamin pelanggan (male/female)
        - **SeniorCitizen**: Apakah pelanggan adalah warga senior (0 = tidak, 1 = iya)
        - **Partner**: Apakah pelanggan memiliki pasangan (yes/no)
        - **Dependents**: Apakah pelanggan memiliki tanggungan (yes/no)
        - **PhoneService**: Apakah pelanggan memiliki layanan telepon (yes/no)
        - **MultipleLines**: Apakah pelanggan memiliki beberapa jalur telepon (yes/no/no_phone_service)
        - **InternetService**: Jenis layanan internet pelanggan (dsl/no/fiber_optic)
        - **OnlineSecurity**: Apakah pelanggan memiliki keamanan online (yes/no/no_internet_service)
        - **OnlineBackup**: Apakah pelanggan memiliki cadangan online (yes/no/no_internet_service)
        - **DeviceProtection**: Apakah pelanggan memiliki perlindungan perangkat (yes/no/no_internet_service)
        - **TechSupport**: Apakah pelanggan memiliki dukungan teknis (yes/no/no_internet_service)
        - **StreamingTV**: Apakah pelanggan memiliki layanan streaming TV (yes/no/no_internet_service)
        - **StreamingMovies**: Apakah pelanggan memiliki layanan streaming film (yes/no/no_internet_service)
        - **Contract**: Jenis kontrak pelanggan (month-to-month/one_year/two_year)
        - **PaperlessBilling**: Apakah pelanggan menggunakan penagihan tanpa kertas (yes/no)
        - **PaymentMethod**: Metode pembayaran pelanggan (bank_transfer/credit_card/electronic_check/mailed_check)
        - **Tenure**: Durasi pelanggan telah berlangganan (dalam bulan)
        - **MonthlyCharges**: Biaya bulanan yang dikenakan kepada pelanggan
        - **TotalCharges**: Total biaya yang dikenakan kepada pelanggan
        """)
    st.caption("Rilo Prianoko | Informatika | Universitas Gunadarma")

if __name__ == '__main__':
    main()
