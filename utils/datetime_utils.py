"""
Useful methods for datetime operations.
"""


def generate_date_strings(start_year, start_month, end_year, end_month):
    """
    Generates a list of year-month strings between the specified start and end dates.
    :param start_year: (int) Start year.
    :param start_month: (int) Start month (1 to 12).
    :param end_year: (int) End year.
    :param end_month: (int) End month (1 to 12).
    :return: A list of year-month strings (e.g., '202001').
    """

    date_strings = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if (year == start_year and month < start_month) or (year == end_year and month > end_month):
                continue
            date_strings.append(f"{year}{month:02d}")

    return date_strings


