# Function to preprocess a batch of images on the GPU
def preprocess_image(image_path, target_size=(224, 224)):
    try:
        # Load image from file
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)

        # Resize the image
        img = tf.image.resize(img, target_size)

        # Normalize pixel values to [0, 1]
        img = img / 255.0
        return img
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Function to preprocess images using GPU and save them
def preprocess_and_save(image_dir, output_dir, target_size=(224, 224), batch_size=100):
    os.makedirs(output_dir, exist_ok=True)

    # Get all image paths
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(('.jpg', '.jpeg', '.png'))]

    num_images = len(image_paths)
    print(f"Processing {num_images} images...")

    for i in tqdm(range(0, num_images, batch_size)):
        batch_paths = image_paths[i:i + batch_size]

        # Load and preprocess the batch of images
        images = [preprocess_image(path, target_size) for path in batch_paths]

        # Convert to NumPy arrays and save
        for img, path in zip(images, batch_paths):
            if img is not None:
                # Convert tensor to NumPy array
                img_np = tf.keras.preprocessing.image.array_to_img(img)
                output_path = os.path.join(output_dir, os.path.basename(path))
                img_np.save(output_path)

# Preprocess training and test images using GPU
preprocess_and_save('/content/drive/My Drive/aml_images/train', '/content/drive/My Drive/preprocessed/train', batch_size=1000)
preprocess_and_save('/content/drive/My Drive/aml_images/test', '/content/drive/My Drive/preprocessed/test', batch_size=1000)

print("Image preprocessing with GPU complete!")