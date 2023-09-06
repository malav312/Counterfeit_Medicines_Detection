import concurrent.futures
from helper_methods import * 

def run_concurrent(func, args_list, max_workers=5):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, args) for args in args_list]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results

def main():
    args_list = getMedicineNames("../data/input/main.csv")
    results = run_concurrent(getDataForMedicine, args_list*10)
    print(results)

if __name__ == '__main__':
    main()
