# Define the download function
def download_image(image_url, image_filename):
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(image_filename, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
        print(f"Downloaded: {image_filename}")
    except Exception as e:
        print(f"Failed to download {image_filename}: {e}")

# Specify the directory to store images in Google Drive
drive_image_dir = '/content/drive/My Drive/aml_images'

# Load the CSV files
train_df = pd.read_csv('/content/drive/MyDrive/aml_resource/dataset/train.csv')
test_df = pd.read_csv('/content/drive/MyDrive/aml_resource/dataset/test.csv')

# Create directories for images if they don't exist
os.makedirs(f'{drive_image_dir}/train', exist_ok=True)
os.makedirs(f'{drive_image_dir}/test', exist_ok=True)

# Function to handle parallel download
def download_images_in_parallel(df, folder_name, max_workers=20):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for index, row in df.iterrows():
            image_url = row['image_link']
            image_filename = f"{folder_name}/{index}.jpg"
            futures.append(executor.submit(download_image, image_url, image_filename))

        # Wait for all downloads to complete
        for future in futures:
            future.result()

# Download training images
print("Downloading training images...")
download_images_in_parallel(train_df, f'{drive_image_dir}/train')

# Download test images
print("Downloading test images...")
download_images_in_parallel(test_df, f'{drive_image_dir}/test')

print("Data preparation complete!")