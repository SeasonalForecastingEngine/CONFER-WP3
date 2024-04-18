
# Quadratic bounding boxes for the GHA and each country within it

def domain_boundaries(region):
    if region == 'GHA':
        return [22, 52], [-12, 18]
    elif region == 'Burundi':
        return [26, 34], [-7, 1]
    elif region == 'Djibouti':
        return [38, 46], [7, 15]
    elif region == 'Eritrea':
        return [36, 44], [11, 19]
    elif region == 'Ethiopia':
        return [33, 48], [1, 16]
    elif region == 'Kenya':
        return [33, 43], [-5, 5]
    elif region == 'Rwanda':
        return [26, 34], [-7, 1]
    elif region == 'Somalia':
        return [39, 53], [-2, 12]
    elif region == 'South Sudan':
        return [23, 36], [1, 14]
    elif region == 'Sudan':
        return [21, 39], [5, 23]
    elif region == 'Tanzania':
        return [29, 41], [-12, 0]
    elif region == 'Uganda':
        return [29, 37], [-2, 6]
    else:
        print('Warning! Region not found.')



# Start and end year of the climatological reference period

def climatological_reference_period():
    return 1993, 2022



# Various global parameters

def global_parameters():
    day_start = 15                       # day within this month when the onset date search starts
    nwks = 22                            # length (number of weeks) of the onset search window
    ndts = 7*nwks+23                     # number of lead time days needed to calculate an onset date within the chosen search window
    ntwd = 25                            # size (days) of the time window over which data for estimating a climatology are collected
    return day_start, nwks, ndts, ntwd




