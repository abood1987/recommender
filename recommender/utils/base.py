import os
from django.core.management.utils import get_random_secret_key


################################################################################
# SECRET KEY

def get_secret_key():
    secret_key_file = os.path.join(
        os.path.dirname(
            os.path.abspath(os.path.join(__file__)),
        ),
        "secret_key.txt",
    )

    if os.path.exists(secret_key_file):
        f = open(secret_key_file, "r")
        return f.read()

    secret_key = get_random_secret_key()

    f = open(secret_key_file, "w")
    f.write(secret_key)
    f.close()

    return secret_key
