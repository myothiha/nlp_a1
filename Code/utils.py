def open_file(path_to_file):
    # Open the file in read mode
    try:
        with open(path_to_file, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"The file {path_to_file} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return content