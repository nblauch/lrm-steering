import random
import string

def get_random_name():
    """
    generate a random 10-digit string of uppercase and lowercase letters, and digits
    """
    return ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=10))
