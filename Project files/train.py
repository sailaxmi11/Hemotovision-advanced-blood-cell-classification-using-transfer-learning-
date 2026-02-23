import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# =========================
# Paths
# =========================
train_path = r"C:\Users\HP\Desktop\hematovision\dataset\TRAIN"
test_path = r"C:\Users\HP\Desktop\hematovision\dataset\TEST"

# =========================
# Image Preprocessing
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# =========================
# Load MobileNetV2
# =========================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)   # ✅ VERY IMPORTANT
)

# Freeze layers (Transfer Learning)
base_model.trainable = False

# =========================
# Custom Classification Head
# =========================
x = base_model.output
x = GlobalAveragePooling2D()(x)   # ✅ FIXED (NO Flatten)
x = Dropout(0.3)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# Compile Model
# =========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# Train Model
# =========================
model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=5
)

# =========================
# Save Model
# =========================
model.save(r"C:\Users\HP\Desktop\hematovision\blood_cell.h5")

print("✅ Model Saved Successfully!")
