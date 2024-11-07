import os
import shutil

# directory initialization
# - if dir_path does not exist, creates a directory
# - if dir_path does exist, empties its contents (including sub-directories)
# - returns the directory path for convenience
def initialize_directory(dir_path, empty=False):
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
	else:
		if empty:
			for file_name in os.listdir(dir_path):
				file_path = os.path.join(dir_path, file_name)
				try:
					if os.path.isfile(file_path) or os.path.islink(file_path):
						os.unlink(file_path)
					elif os.path.isdir(file_path):
						shutil.rmtree(file_path)
				except Exception as e:
					print("Failed to delete {}. Reason: {}".format(file_path, e))
	return dir_path
