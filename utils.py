import os

def get_path(path):
	"""
	Checks if path exists
	"""

	assert os.path.exists(path), \
		"Can't find '{0}'. Make sure this script is being run in the same folder as this script."\
			.format(path)

	return path