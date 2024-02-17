import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# โฟลเดอร์ที่เก็บชุดข้อมูลรูปภาพ
train_dir = r'C:\Users\admin\OneDrive\เอกสาร\model\dataset\train'
validation_dir = r'C:\Users\admin\OneDrive\เอกสาร\model\dataset\validation'

# สร้างโมเดล CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # เปลี่ยนจำนวน unit ใน output layer เป็น 3 และใช้ softmax activation function
])

# คอมไพล์และสร้างโมเดล
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # เปลี่ยน loss function เป็น sparse categorical crossentropy
              metrics=['accuracy'])

# โหลดและเตรียมข้อมูล
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# สร้างชุดข้อมูลโดยไม่สลับข้อมูล
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='sparse'  # เปลี่ยน class_mode เป็น 'sparse'
)

# เตรียมชุดข้อมูลการทดสอบ
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='sparse'  # เปลี่ยน class_mode เป็น 'sparse'
)

# เทรนโมเดล
history = model.fit(
      train_generator,
      epochs=30,
      validation_data=validation_generator
)

model.save('C:\\Users\\admin\\Downloads\\my_model.h5')

# ประเมินผล
test_loss, test_acc = model.evaluate(validation_generator)
print('Test accuracy:', test_acc)
