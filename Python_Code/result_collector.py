import os

def directory_results(path):
    results = []
    sub_results = []
    count = 0
    dirs = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]
    sorted(dirs)
    successes, fails = 0, 0
    key_ratios = []
    example_directories = []
    
    for direc in dirs:
        is_success = find_result(os.path.join(path, direc))
        count+=1
        sub_results.append(is_success)
        if is_success:
            successes += 1
        else:
            fails += 1
        
        if (count % 20 == 0):
            results.append(sub_results)
            sub_results = []
            key_ratios.append((successes,fails))
            example_directories.append(direc)
            successes, fails = 0, 0
            
    return results, key_ratios, example_directories

def find_result(path):
    file = path + "\\result.txt"
    f = open(file, "r")
    return str_to_bool(f.read())

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError