import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory, image
import os

# Đường dẫn dataset
DATASET_PATH = "D:\\PROJECT DEEP LEARNING\\khungbo\\jpeg-192x192"  

# Thiết lập thông số
IMG_SIZE = (180, 180)
BATCH_SIZE = 32

# Tạo dataset từ thư mục
train_dataset = image_dataset_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    shuffle=True,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = image_dataset_from_directory(
    os.path.join(DATASET_PATH, 'val'),
    shuffle=True,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_dataset = image_dataset_from_directory(
    os.path.join(DATASET_PATH, 'test'),
    shuffle=True,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Lấy tên lớp từ thư mục
class_names = train_dataset.class_names
print("Classes:", class_names)

# Chuẩn hóa dữ liệu
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])

# Xây dựng mô hình CNN
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile mô hình
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=100)

# Lưu mô hình
model.save("flower_classification_model.h5")

# Hiển thị thông tin huấn luyện
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Load lại mô hình để sử dụng
model = tf.keras.models.load_model("flower_classification_model.h5")

# Tiền xử lý ảnh đầu vào để dự đoán
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(180, 180))  # Resize ảnh
    img_array = image.img_to_array(img) / 255.0  # Chuẩn hóa ảnh
    img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
    return img_array

# Dự đoán ảnh mới
def predict_flower(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Lấy lớp có xác suất cao nhất
    return class_names[predicted_class]

# Ví dụ sử dụng
img_path = "test_flower.jpg"  # Thay bằng đường dẫn ảnh cần dự đoán
print(f"Ảnh thuộc lớp: {predict_flower(img_path)}")
