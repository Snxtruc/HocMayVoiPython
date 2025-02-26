import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import openml
from sklearn import datasets
from sklearn.model_selection import train_test_split
from streamlit_drawable_canvas import st_canvas
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import joblib
import pandas as pd
import os
import mlflow
from mlflow.tracking import MlflowClient

# Load dữ liệu MNIST

def thong_tin_ung_dung():
    st.header("📂 Tải Dữ Liệu")
    st.write(
        "🔹 Người dùng có thể lấy bộ dữ liệu **MNIST** trực tiếp từ **OpenML** "
        "hoặc tải lên tệp CSV tùy chỉnh (yêu cầu có cột `label` chứa nhãn của ảnh)."
    )
    st.write("🔹 Sau khi tải dữ liệu, ứng dụng hiển thị một số mẫu chữ số để trực quan hóa dữ liệu.")
    st.header("📊 Huấn Luyện & Đánh Giá Mô Hình")
    st.write("Ứng dụng hỗ trợ hai mô hình phân loại chính:")

    st.subheader("1️⃣ Decision Tree")
    st.write("Mô hình **Cây quyết định** với các tham số:")
    st.markdown(
        """
        - **criterion**: Tiêu chí phân nhánh (Gini hoặc Entropy).
        - **max_depth**: Độ sâu tối đa của cây.
        """
    )
    st.write("👉 Sau khi huấn luyện, ứng dụng hiển thị độ chính xác trên tập validation và test.")

    st.subheader("2️⃣ Support Vector Machine (SVM)")
    st.write("Mô hình **SVM** với các tham số:")
    st.markdown(
        """
        - **C**: Hệ số điều chỉnh độ phạt lỗi.
        - **kernel**: Loại kernel (linear, polynomial, RBF, v.v.).
        - **gamma**: Tham số cho kernel RBF.
        - **degree**: Bậc của kernel Polynomial (nếu sử dụng).
         """
    )
    st.write("👉 Hiển thị độ chính xác của mô hình sau khi huấn luyện.")
    st.header("✅ So Sánh Kết Quả")
    st.write(
        "Ứng dụng hiển thị **biểu đồ confusion matrix** để giúp người dùng "
        "đánh giá hiệu suất mô hình và phát hiện lỗi phân loại."
    )
    st.header("🖼️ Demo Dự Đoán")
    st.write(
        "🔹 Người dùng có thể chọn một mẫu từ tập Test hoặc tải lên hình ảnh chữ số mới để kiểm tra dự đoán của mô hình."
    )
    st.write("🔹 Ứng dụng sẽ hiển thị hình ảnh đầu vào cùng với nhãn dự đoán.")
    st.header("📈 Thông Tin Huấn Luyện")
    st.write("Ứng dụng ghi lại quá trình huấn luyện và kết quả thông qua **MLflow**, giúp theo dõi hiệu suất mô hình.")
    st.write("🔹 Người dùng có thể xem chi tiết **Run ID**, các tham số sử dụng, và độ chính xác trên các tập dữ liệu.")
    st.write("🔹 Giao diện **MLflow UI** có thể được mở để phân tích sâu hơn về các mô hình đã huấn luyện.")

def ly_thuyet_Decision_tree():
    st.header("📖 Lý thuyết về Decision Tree") 
    st.header("🌳 Giới thiệu về Decision Tree")
    st.markdown(" ### 1️⃣ Decision Tree là gì?")
    st.write("""
    Decision Tree (Cây quyết định) là một thuật toán học có giám sát được sử dụng trong **phân loại (classification)** và **hồi quy (regression)**.
    Nó hoạt động bằng cách chia dữ liệu thành các nhóm nhỏ hơn dựa trên các điều kiện được thiết lập tại các **nút (nodes)** của cây.
    """) 
    
    st.markdown(" ### 📌 Cấu trúc của Decision Tree") 
    image_url = "https://trituenhantao.io/wp-content/uploads/2019/06/dt.png"
    st.image(image_url, caption="Ví dụ về cách Cây quyết định phân chia dữ liệu", use_column_width=True)

    st.write("""
    - **Nút gốc (Root Node)**: Là điểm bắt đầu của cây, chứa toàn bộ dữ liệu.
    - **Nút quyết định (Decision Nodes)**: Các nút trung gian nơi dữ liệu được chia nhỏ dựa trên một điều kiện.
    - **Nhánh (Branches)**: Các đường nối giữa các nút, thể hiện lựa chọn có thể xảy ra.
    - **Nút lá (Leaf Nodes)**: Điểm cuối của cây, đại diện cho quyết định cuối cùng hoặc nhãn dự đoán.
    """)

    st.markdown(" ### 🔍 Cách hoạt động của Decision Tree")
    st.write("""
    1. **Chọn đặc trưng tốt nhất để chia dữ liệu** bằng các tiêu chí như:
    - Gini Impurity: Đánh giá độ lẫn lộn của tập dữ liệu.
    - Entropy (dùng trong ID3): Xác định mức độ không chắc chắn.
    - Reduction in Variance (dùng cho hồi quy).
    2. **Tạo các nhánh con** từ đặc trưng được chọn.
    3. **Lặp lại quy trình** trên từng nhánh con cho đến khi đạt điều kiện dừng.
    4. **Dự đoán dữ liệu mới** bằng cách đi theo cây từ gốc đến lá.
    """)

    st.markdown("### 🌳 Công Thức Chính của Cây Quyết Định và Cách Áp Dụng")

    st.subheader("📌 1. Entropy – Độ hỗn loạn của dữ liệu")
    st.write("Entropy đo lường mức độ hỗn loạn trong dữ liệu. Nếu một tập dữ liệu càng đồng nhất, entropy càng thấp.")
    st.latex(r"H(S) = - \sum_{i=1}^{c} p_i \log_2 p_i")

    st.write("""
    - Nếu tất cả dữ liệu thuộc cùng một lớp → Entropy = 0 (thuần khiết).
    - Nếu dữ liệu được phân bố đều giữa các lớp → Entropy đạt giá trị cao nhất.
    """)

    st.subheader("📌 2. Information Gain – Mức độ giảm độ hỗn loạn sau khi chia dữ liệu")
    st.latex(r"IG = H(S) - \sum_{j=1}^{k} \frac{|S_j|}{|S|} H(S_j)")

    st.write("""
    - IG càng cao → thuộc tính đó giúp phân loại dữ liệu tốt hơn.
    - IG thấp → thuộc tính đó không có nhiều giá trị trong việc phân tách dữ liệu.
    """)

    st.subheader("📌 3. Gini Impurity – Đo lường mức độ hỗn loạn thay thế Entropy")
    st.latex(r"Gini(S) = 1 - \sum_{i=1}^{c} p_i^2")

    st.write("""
    - Gini = 0 → tập dữ liệu hoàn toàn thuần khiết.
    - Gini càng cao → dữ liệu càng hỗn loạn.
    """)

    st.subheader("💡 Cách Áp Dụng để Xây Dựng Decision Tree")
    st.write("""
    1. Tính Entropy hoặc Gini của tập dữ liệu ban đầu.
    2. Tính Entropy hoặc Gini của từng tập con sau khi chia theo từng thuộc tính.
    3. Tính Information Gain cho từng thuộc tính.
    4. Chọn thuộc tính có Information Gain cao nhất để chia nhánh.
    5. Lặp lại quy trình trên cho đến khi tất cả dữ liệu trong các nhánh đều thuần khiết hoặc đạt điều kiện dừng.
    """)

    st.markdown("""
    **📌 Lưu ý:**  
    - Nếu cây quá sâu → có thể gây overfitting, cần sử dụng cắt tỉa (pruning).  
    - Decision Tree có thể sử dụng với cả phân loại (Classification) và hồi quy (Regression).  
    """)

    st.write("🚀 Cây quyết định là một thuật toán mạnh mẽ và dễ hiểu, nhưng cần điều chỉnh để tránh overfitting và tối ưu hiệu suất!") 
    
    
    
def ly_thuyet_SVM():
    # Tiêu đề chính
    st.title("📖 Lý Thuyết Về Support Vector Machine (SVM)")
    st.image("https://neralnetwork.wordpress.com/wp-content/uploads/2018/01/svm1.png", caption="Hình minh họa SVM")

    st.markdown("""
    Support Vector Machine (SVM) là một thuật toán học máy mạnh mẽ thường được sử dụng cho bài toán **phân loại (classification)** và **hồi quy (regression)**. 
    Nó hoạt động dựa trên nguyên lý tìm **siêu phẳng (hyperplane)** tối ưu để phân tách dữ liệu.
    """)

    # 1. Nguyên lý hoạt động
    st.header("1. Nguyên Lý Hoạt Động của SVM")

    st.subheader("📌 1.1. Tìm Siêu Phẳng Tối Ưu")
    st.markdown("""
    - Một **siêu phẳng (hyperplane)** là một đường (trong không gian 2D) hoặc một mặt phẳng (trong không gian 3D) dùng để phân tách dữ liệu thành các nhóm.
    - SVM tìm **siêu phẳng tối ưu** sao cho khoảng cách từ siêu phẳng đến các điểm dữ liệu gần nhất (**support vectors**) là lớn nhất.
    """)

    st.write("🚀 **Công thức siêu phẳng:**")
    st.latex(r"w \cdot x + b = 0")

    st.markdown("""
    Trong đó:
    - \( w \) là **vector trọng số**,
    - \( x \) là **vector dữ liệu đầu vào**,
    - \( b \) là **bias**.
    """)

    st.subheader("📌 1.2. Khoảng Cách Lề (Margin)")
    st.markdown("""
    - **Soft Margin SVM**: Chấp nhận một số điểm bị phân loại sai nhưng tăng khả năng tổng quát hóa (**giảm overfitting**).
    - **Hard Margin SVM**: Yêu cầu phân tách hoàn hảo, không cho phép lỗi nhưng dễ bị **overfitting**.
    """)

    # 2. Hàm mục tiêu
    st.header("2. Hàm Mục Tiêu trong SVM")
    st.markdown("""
    Mục tiêu của SVM là tìm \( w \) và \( b \) để tối đa hóa khoảng cách lề \( \frac{2}{||w||} \), tương đương với bài toán tối ưu:
    """)

    st.latex(r"\min_{w, b} \frac{1}{2} ||w||^2")

    st.markdown("Sao cho:")

    st.latex(r"y_i (w \cdot x_i + b) \geq 1, \forall i")

    st.markdown("""
    Trong đó:
    - \( y_i \) là **nhãn của dữ liệu** (1 hoặc -1),
    - \( x_i \) là **điểm dữ liệu**.
    """)

    # 3. Kernel Trick
    st.header("3. Kernel Trick – Mở Rộng SVM Cho Dữ Liệu Phi Tuyến")
    st.markdown("""
    Khi dữ liệu không thể phân tách tuyến tính, SVM sử dụng **hàm kernel** để ánh xạ dữ liệu vào không gian chiều cao hơn, nơi có thể phân tách tuyến tính.

    📌 **Một số loại Kernel phổ biến**:
    """)

    st.subheader("1️⃣ Linear Kernel")
    st.latex(r"K(x_i, x_j) = x_i \cdot x_j")
    st.markdown("👉 Sử dụng khi dữ liệu có thể phân tách tuyến tính.")

    st.subheader("2️⃣ Polynomial Kernel")
    st.latex(r"K(x_i, x_j) = (x_i \cdot x_j + c)^d")
    st.markdown("👉 Phù hợp với dữ liệu có ranh giới phi tuyến.")

    st.subheader("3️⃣ Radial Basis Function (RBF) Kernel")
    st.latex(r"K(x_i, x_j) = \exp(- \gamma ||x_i - x_j||^2)")
    st.markdown("👉 Phổ biến nhất vì có thể xử lý **mọi loại dữ liệu**.")

    st.write("🚀 SVM là một thuật toán mạnh mẽ, nhưng cần điều chỉnh đúng tham số để đạt hiệu suất tối ưu!")


def data():
    st.header("MNIST Dataset")
    st.title("Tổng quan về tập dữ liệu MNIST")

    st.header("1. Giới thiệu")
    st.write("Tập dữ liệu MNIST (Modified National Institute of Standards and Technology) là một trong những tập dữ liệu phổ biến nhất trong lĩnh vực Machine Learning và Computer Vision, thường được dùng để huấn luyện và kiểm thử các mô hình phân loại chữ số viết tay.") 
    
    st.image("https://datasets.activeloop.ai/wp-content/uploads/2019/12/MNIST-handwritten-digits-dataset-visualized-by-Activeloop.webp", use_container_width=True)

    st.subheader("Nội dung")
    st.write("- 70.000 ảnh grayscale (đen trắng) của các chữ số viết tay từ 0 đến 9.")
    st.write("- Kích thước ảnh: 28x28 pixel.")
    st.write("- Định dạng: Mỗi ảnh được biểu diễn bằng một ma trận 28x28 có giá trị pixel từ 0 (đen) đến 255 (trắng).")
    st.write("- Nhãn: Một số nguyên từ 0 đến 9 tương ứng với chữ số trong ảnh.")

    st.header("2. Nguồn gốc và ý nghĩa")
    st.write("- Được tạo ra từ bộ dữ liệu chữ số viết tay gốc của NIST, do LeCun, Cortes và Burges chuẩn bị.")
    st.write("- Dùng làm benchmark cho các thuật toán nhận diện hình ảnh, đặc biệt là mạng nơ-ron nhân tạo (ANN) và mạng nơ-ron tích chập (CNN).")
    st.write("- Rất hữu ích cho việc kiểm thử mô hình trên dữ liệu hình ảnh thực tế nhưng đơn giản.")

    st.header("3. Phân chia tập dữ liệu")
    st.write("- Tập huấn luyện: 60.000 ảnh.")
    st.write("- Tập kiểm thử: 10.000 ảnh.")
    st.write("- Mỗi tập có phân bố đồng đều về số lượng chữ số từ 0 đến 9.")

    st.header("4. Ứng dụng")
    st.write("- Huấn luyện và đánh giá các thuật toán nhận diện chữ số viết tay.")
    st.write("- Kiểm thử và so sánh hiệu suất của các mô hình học sâu (Deep Learning).")
    st.write("- Làm bài tập thực hành về xử lý ảnh, trích xuất đặc trưng, mô hình phân loại.")
    st.write("- Cung cấp một baseline đơn giản cho các bài toán liên quan đến Computer Vision.")

    st.header("5. Phương pháp tiếp cận phổ biến")
    st.write("- Trích xuất đặc trưng truyền thống: PCA, HOG, SIFT...")
    st.write("- Machine Learning: KNN, SVM, Random Forest, Logistic Regression...")
    st.write("- Deep Learning: MLP, CNN (LeNet-5, AlexNet, ResNet...), RNN")

    st.caption("Ứng dụng hiển thị thông tin về tập dữ liệu MNIST bằng Streamlit 🚀")
    


def up_load_db():
    # Tiêu đề
    st.header("📥 Tải Dữ Liệu")

    # Chọn nguồn dữ liệu
    option = st.radio("Chọn nguồn dữ liệu:", ["Tải từ OpenML", "Upload dữ liệu"])

    # Nếu chọn tải từ OpenML
    if option == "Tải từ OpenML":
        st.markdown("#### 📂 Tải dữ liệu MNIST từ OpenML")
        if st.button("Tải dữ liệu MNIST"):
            st.write("🔄 Đang tải dữ liệu MNIST từ OpenML...")
            
            # Load dữ liệu MNIST từ OpenML
            mnist = openml.datasets.get_dataset(554)  # MNIST dataset ID trên OpenML
            X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
            
            # Hiển thị 5 dòng dữ liệu đầu tiên
            st.write("📊 **Dữ liệu mẫu:**")
            st.write(pd.DataFrame(X.head()))

            st.success("✅ Dữ liệu MNIST đã được tải thành công!")

    # Nếu chọn upload dữ liệu từ máy
    else:
        st.markdown("#### 📤 Upload dữ liệu của bạn")

        uploaded_file = st.file_uploader("Chọn một file ảnh", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            # Mở file hình ảnh
            image = Image.open(uploaded_file)

            # Hiển thị ảnh
            st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

            # Kiểm tra kích thước ảnh
            if image.size != (28, 28):
                st.error("❌ Ảnh không đúng kích thước 28x28 pixel. Vui lòng tải lại ảnh đúng định dạng.")
            else:
                st.success("✅ Ảnh hợp lệ!")

    # Hiển thị lưu ý
    st.markdown("""
    🔹 **Lưu ý:**
    - Ứng dụng chỉ sử dụng dữ liệu ảnh dạng **28x28 pixel (grayscale)**.
    - Dữ liệu phải có cột **'label'** chứa nhãn (số từ 0 đến 9).
    - Nếu dữ liệu của bạn không đúng định dạng, vui lòng sử dụng dữ liệu MNIST từ OpenML.
    """)

import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split

def chia_du_lieu():
    st.title("📌 Chia dữ liệu Train/Validation/Test")

    # Đọc dữ liệu
    X = np.load("buoi4/X.npy")
    y = np.load("buoi4/y.npy")
    total_samples = X.shape[0]

    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("Chọn số lượng ảnh để sử dụng:", 1000, total_samples, 10000)

    # Thanh kéo chọn tỷ lệ Test
    test_size = st.slider("Chọn tỷ lệ Test:", 0.1, 0.5, 0.2)

    # Thanh kéo chọn tỷ lệ Validation (trên tổng dữ liệu)
    val_size = st.slider("Chọn tỷ lệ Validation:", 0.1, 0.5, 0.2)

    # Tính toán tỷ lệ Train
    train_size = 1.0 - test_size - val_size
    st.write(f"📌 Tỷ lệ Train được tính toán: {train_size:.2f}")

    if train_size <= 0:
        st.error("⚠️ Tổng tỷ lệ Test + Validation vượt quá 1. Vui lòng điều chỉnh lại!")
        return

    if st.button("✅ Xác nhận & Lưu"):
        # Lấy số lượng ảnh mong muốn
        X_selected, y_selected = X[:num_samples], y[:num_samples]

        # Chia dữ liệu thành Train và Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(X_selected, y_selected, test_size=test_size, random_state=42)

        # Chia tiếp tập Train thành Train và Validation
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size / (train_size + val_size), random_state=42)

        # Lưu vào session_state để sử dụng sau
        st.session_state["X_train"] = X_train
        st.session_state["y_train"] = y_train
        st.session_state["X_val"] = X_val
        st.session_state["y_val"] = y_val
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

        st.success(f"🔹 Dữ liệu đã được chia: Train ({len(X_train)}), Validation ({len(X_val)}), Test ({len(X_test)})")

    # Kiểm tra nếu đã lưu dữ liệu vào session_state
    if "X_train" in st.session_state:
        st.write("📌 Dữ liệu Train/Validation/Test đã sẵn sàng để sử dụng!")


def train():
    """Huấn luyện mô hình Decision Tree hoặc SVM và lưu trên MLflow."""
    
    mlflow_input()

    # 📥 Kiểm tra và lấy dữ liệu từ session_state
    if not all(key in st.session_state for key in ["X_train", "y_train", "X_test", "y_test"]):
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    X_train, y_train = st.session_state["X_train"], st.session_state["y_train"]
    X_test, y_test = st.session_state["X_test"], st.session_state["y_test"]

    # 🌟 Chuẩn hóa dữ liệu
    X_train, X_test = X_train.reshape(-1, 28 * 28) / 255.0, X_test.reshape(-1, 28 * 28) / 255.0

    st.header("⚙️ Chọn mô hình & Huấn luyện")

    # 📌 Lựa chọn mô hình
    model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"])
    
    if model_choice == "Decision Tree":
        criterion = st.selectbox("Criterion", ["gini", "entropy"])
        max_depth = st.slider("max_depth", 1, 20, 5)
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    else:
        C = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        model = SVC(C=C, kernel=kernel)

    # 🚀 Bắt đầu huấn luyện khi nhấn nút
    if st.button("Huấn luyện mô hình"):
        if "mlflow_url" not in st.session_state:
            st.session_state["mlflow_url"] = f"https://dagshub.com/Snxtruc/HocMayVoiPython.mlflow"

        with mlflow.start_run():
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            st.success(f"✅ Độ chính xác: {acc:.4f}")

            # Ghi log thông số và kết quả lên MLflow
            mlflow.log_param("model", model_choice)
            if model_choice == "Decision Tree":
                mlflow.log_param("criterion", criterion)
                mlflow.log_param("max_depth", max_depth)
            elif model_choice == "SVM":
                mlflow.log_param("C", C)
                mlflow.log_param("kernel", kernel)

            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, model_choice.lower())

        # 📌 Quản lý mô hình trong session_state
        if "models" not in st.session_state:
            st.session_state["models"] = []

        model_name = model_choice.lower().replace(" ", "_")
        if model_choice == "Decision Tree":
            model_name += f"_{criterion}_depth{max_depth}"
        elif model_choice == "SVM":
            model_name += f"_{kernel}"

        # Xử lý trùng lặp tên mô hình
        existing_names = {m["name"] for m in st.session_state["models"]}
        count = 1
        while model_name in existing_names:
            model_name = f"{model_name}_{count}"
            count += 1

        # Lưu mô hình vào session_state
        st.session_state["models"].append({"name": model_name, "model": model})
        st.write(f"🔹 Mô hình đã được lưu với tên: **{model_name}**")
        st.write(f"📋 Tổng số mô hình hiện tại: {len(st.session_state['models'])}")

        # Hiển thị danh sách mô hình đã lưu
        model_names = [m["name"] for m in st.session_state["models"]]
        st.write("📋 Danh sách mô hình đã lưu:", ", ".join(model_names))

        st.success("📌 Mô hình đã được lưu trên MLflow!")
        st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")

def mlflow_input():
    #st.title("🚀 MLflow DAGsHub Tracking với Streamlit")


    DAGSHUB_USERNAME = "Snxtruc"  # Thay bằng username của bạn
    DAGSHUB_REPO_NAME = "HocMayVoiPython"
    DAGSHUB_TOKEN = "ca4b78ae4dd9d511c1e0c333e3b709b2cd789a19"  # Thay bằng Access Token của bạn

    # Đặt URI MLflow để trỏ đến DagsHub
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")

    # Thiết lập authentication bằng Access Token
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    # Đặt thí nghiệm MLflow
    mlflow.set_experiment("Classification")  
    
def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"⚠️ Không tìm thấy mô hình tại `{path}`")
        st.stop()

# ✅ Xử lý ảnh từ canvas (chuẩn 28x28 cho MNIST)
def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
        img = img.resize((28, 28)).convert("L")  # Resize và chuyển thành grayscale
        img = np.array(img, dtype=np.float32) / 255.0  # Chuẩn hóa về [0, 1]
        return img.reshape(1, -1)  # Chuyển thành vector 1D
    return None


def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is None:
        return None
    img = Image.fromarray((canvas_result.image_data[:, :, 0] * 255).astype(np.uint8))
    img = img.convert("L").resize((28, 28))  # Chuyển sang ảnh xám 28x28
    img = np.array(img) / 255.0  # Chuẩn hóa
    return img.reshape(1, -1)

def du_doan():
    st.header("✍️ Dự đoán số")
    
    # 🔹 Chọn phương thức dự đoán
    mode = st.radio("Chọn phương thức dự đoán:", ["Vẽ số", "Upload file test"])
    
    if mode == "Vẽ số":
        # ✍️ Vẽ số
        st.subheader("🖌️ Vẽ số vào khung dưới đây:")
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            height=150,
            width=150,
            drawing_mode="freedraw",
            key="canvas"
        )
    
    elif mode == "Upload file test":
        # 🔹 Upload file test
        st.header("📂 Dự đoán trên tập test")
        uploaded_file = st.file_uploader("Tải tập test (CSV hoặc NPY):", type=["csv", "npy"])
        
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                test_data = pd.read_csv(uploaded_file).values
            else:
                test_data = np.load(uploaded_file)
            
            st.write(f"📊 Dữ liệu test có {test_data.shape[0]} mẫu.")
    
    # 🔹 Danh sách mô hình có sẵn
    available_models = {
        "SVM Linear": "buoi4/svm_mnist_linear.joblib",
        "SVM Poly": "buoi4/svm_mnist_poly.joblib",
        "SVM Sigmoid": "buoi4/svm_mnist_sigmoid.joblib",
        "SVM RBF": "buoi4/svm_mnist_rbf.joblib",
    }
    
    # 📌 Chọn mô hình
    model_option = st.selectbox("🔍 Chọn mô hình:", list(available_models.keys()))
    
    # Tải mô hình
    model = joblib.load(available_models[model_option])
    st.success(f"✅ Mô hình {model_option} đã được tải thành công!")
    
    if mode == "Vẽ số":
        if st.button("Dự đoán số"):
            if canvas_result.image_data is not None:
                img = preprocess_canvas_image(canvas_result)
                st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)
                prediction = model.predict(img)
                probabilities = model.decision_function(img) if hasattr(model, 'decision_function') else model.predict_proba(img)
                confidence = np.max(probabilities) if probabilities is not None else "Không xác định"
                st.subheader(f"🔢 Kết quả dự đoán: {prediction[0]} (Độ tin cậy: {confidence:.2f})")
            else:
                st.error("⚠️ Vui lòng vẽ một số trước khi dự đoán!")
    
    elif mode == "Upload file test" and uploaded_file is not None:
        if st.button("Dự đoán trên tập test"):
            predictions = model.predict(test_data)
            probabilities = model.decision_function(test_data) if hasattr(model, 'decision_function') else model.predict_proba(test_data)
            confidences = np.max(probabilities, axis=1) if probabilities is not None else ["Không xác định"] * len(predictions)
            
            st.write("🔢 Kết quả dự đoán:")
            for i in range(min(10, len(predictions))):
                st.write(f"Mẫu {i + 1}: {predictions[i]} (Độ tin cậy: {confidences[i]:.2f})")
            
            fig, axes = plt.subplots(1, min(5, len(test_data)), figsize=(10, 2))
            for i, ax in enumerate(axes):
                ax.imshow(test_data[i].reshape(28, 28), cmap='gray')
                ax.set_title(f"{predictions[i]} ({confidences[i]:.2f})")
                ax.axis("off")
            st.pyplot(fig)




def Classification():
    st.markdown(
        """
        <style>
        .stTabs [role="tablist"] {
            overflow-x: auto;
            white-space: nowrap;
            display: flex;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🖥️ MNIST Classification App")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "ℹ️ Thông tin", "📖 Lý thuyết Decision Tree", "📖 Lý thuyết SVM", 
        "🚀 Review database", "📥 Tải dữ liệu", "⚙️ Huấn luyện & mlflow", "🔮 Dự đoán"
    ])

    with tab1: 
        thong_tin_ung_dung()

    with tab2:
        ly_thuyet_Decision_tree()

    with tab3:
        ly_thuyet_SVM()
    
    with tab4:
        data()

    with tab5:
        up_load_db()    
      
    with tab6:      
        chia_du_lieu()
        train()
        
    with tab7:
        du_doan()
  
        
if __name__ == "__main__":
    Classification()