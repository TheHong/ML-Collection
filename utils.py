import os


def get_path(path):
    """
    Checks if path exists
    """
    advice = "Make sure this script is run in the same folder as this script."
    assert os.path.exists(path), f"Can't find '{path}'. {advice}"
    return path
