def binary_search_recursive(arr, target, left=0, right=None):
    """Return the index of `target` in `arr` using recursive binary search.
    Returns -1 if `target` is not present."""
    if right is None:
        right = len(arr) - 1
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)


def binary_search_iterative(arr, target):
    """Return the index of `target` in `arr` using an iterative binary search.
    Returns -1 if `target` is not present."""
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


if __name__ == "__main__":
    # Sample array
    arr = [2, 5, 8, 12, 16, 23, 26, 33, 38, 42]

    targets = [23, 7, 100]

    for target in targets:
        result_recursive = binary_search_recursive(arr, target)
        result_iterative = binary_search_iterative(arr, target)
        print(f"Array: {arr}")
        print(f"Target: {target}")
        print(f"  Recursive search -> index = {result_recursive}")
        print(f"  Iterative search -> index = {result_iterative}")
        print()
