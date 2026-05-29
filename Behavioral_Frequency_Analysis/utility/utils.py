#Utility Functions for the project

import os

#file utilities
def sanitize_filename(name):
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def ensure_dir(path): 
    os.makedirs(path, exist_ok=True)


#Time utilities

def seconds_to_timestamp(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:05.2f}"

#participant loader

