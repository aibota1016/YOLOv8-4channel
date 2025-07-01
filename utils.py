import os

# Define the path to the folder containing label text files
labels_folder = "C:/Users/User04/Downloads/dataset_9k_weight_pred/test/labels"

# Ensure the folder exists
if not os.path.exists(labels_folder):
    print(f"The folder '{labels_folder}' does not exist. Please check the path.")
    exit()

# Loop through each file in the folder
for file_name in os.listdir(labels_folder):
    # Check if the file is a text file
    if file_name.endswith(".txt"):
        file_path = os.path.join(labels_folder, file_name)

        # Read the file content
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Modify each line to keep only the first 6 columns
            modified_lines = []
            for line in lines:
                columns = line.split()
                if len(columns) >= 6:  # Ensure there are at least 6 columns
                    modified_lines.append(" ".join(columns[:6]) + "\n")

            # Write the modified lines back to the file
            with open(file_path, 'w') as file:
                file.writelines(modified_lines)

            print(f"Processed file: {file_name}")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

print("All label files have been updated successfully!")
