from google.colab import drive
drive.mount('/content/drive')

train_df = pd.read_csv('/content/drive/MyDrive/aml_resource/dataset/train.csv')
test_df = pd.read_csv('/content/drive/MyDrive/aml_resource/dataset/test.csv')

print("Train Dataset:")
train_df.head()

print("Test Dataset:")
test_df.head()

train_df.shape

test_df.shape

sample_test_input=pd.read_csv('/content/drive/MyDrive/aml_resource/dataset/sample_test.csv')
sample_test_output=pd.read_csv('/content/drive/MyDrive/aml_resource/dataset/sample_test_out.csv')

# basic info
train_df.info()

# missing values
print("Missing values in train dataset:")
print(train_df.isnull().sum())

print("\nMissing values in test dataset:")
print(test_df.isnull().sum())

# Descriptive statistics for numeric columns
train_df.describe()

# Examining the unique entity values (to understand how product attributes vary)
train_df['entity_value'].unique()

len(train_df['entity_value'].unique())

# Check the distribution of the entity names (to understand the types of labels)
print("\nDistribution of Entity Names:")
train_df['entity_name'].value_counts()

plt.figure(figsize=(10,5))
sns.countplot(y=train_df['entity_name'], order=train_df['entity_name'].value_counts().index)
plt.title("Distribution of Entity Names in Training Data")
plt.show()

# images with associated entity and value
def show_image_with_entity_info(image_url, entity_name, entity_value):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.axis('off')

    # Print the entity information
    plt.title(f"Entity: {entity_name} | Value: {entity_value}")
    plt.show()
sample_images = train_df[['image_link', 'entity_name', 'entity_value']].head(5)

for idx, row in sample_images.iterrows():
    print(f"Displaying image from: {row['image_link']}")
    show_image_with_entity_info(row['image_link'], row['entity_name'], row['entity_value'])

# Function to extract units from the entity_value column
def extract_unit(entity_value):
    if isinstance(entity_value, str):
        # Regular expression to extract non-numeric values (units)
        unit = re.findall(r'[a-zA-Z]+', entity_value)
        if unit:
            return unit[0].lower()  # Return the first unit found in lowercase
    return None

# Apply the function to extract units
train_df['unit'] = train_df['entity_value'].apply(extract_unit)

# Check how many different units exist for each entity_name
units_per_entity = train_df.groupby('entity_name')['unit'].unique()

# Display the variety of units for each entity
print("Variety of Units for Each Entity:")
print(units_per_entity)

# analyzing the frequency of units
unit_distribution = train_df.groupby(['entity_name', 'unit']).size().reset_index(name='count')

# Display the unit distribution for each entity
print("Unit Distribution for Each Entity:")
print(unit_distribution)

# Analyze distribution of 'group_id' column
print("\nGroup ID Distribution:")
group_id_counts = train_df['group_id'].value_counts()
print(group_id_counts)