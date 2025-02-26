import streamlit as st
from buoi4.Classification import Classification

option = st.sidebar.selectbox(
    'Chọn ứng dụng:',
    ('Classification với MNIST') 
)

if option == 'Classification với MNIST':
    Classification() 
else:
    st.write("Vui lòng chọn một ứng dụng từ thanh điều hướng.")
