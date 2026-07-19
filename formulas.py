import numpy as np

def haversine(lat1, lon1, lat2, lon2): # haversine distance (in km)
    R = 6371

    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)

    a = np.sin(dLat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d

def mag2e(mag): # magnitude to seismic energy
    return 10 ** (1.5 * mag)