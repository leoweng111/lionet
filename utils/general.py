"""
General convenient methods.
"""


def get_attribute_value(obj, attribute_name):
    try:
        value = getattr(obj, attribute_name)
        return value
    except AttributeError:
        return None
