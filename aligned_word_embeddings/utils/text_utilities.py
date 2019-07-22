def concatenate_files(list_of_text_files, concatenate_path):
    """
    Create the text file for the compass
    :param list_of_text_files:
    :param concatenate_path:
    :return:
    """
    with open(concatenate_path) as file_obj:
        for path_file in list_of_text_files:
            with open(path_file, "r") as myfile:
                current_text = myfile.readlines()
                file_obj.write(current_text + "\n")
