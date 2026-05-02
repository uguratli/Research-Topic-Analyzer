from datetime import datetime, timedelta


def delta_time(months="", days="", start="", end=""):
    """date format: YYYY-MM-DD"""
    if start and end:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
        return start_date, end_date
    elif months:
        days = int(months) * 30 
        return datetime.today() - timedelta(days=days), datetime.today()
    elif days:
        return datetime.today() - timedelta(days=int(days)), datetime.today() 
    else:
        raise ValueError("Please provide either months, days, or start and end dates.")
    