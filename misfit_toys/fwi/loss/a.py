import numpy as np


def bool_slice(*args, permute=None, none_dims=(), ctrl=None):
    permute = list(permute) or list(range(len(args)))
    permute.reverse()

    # Total number of combinations
    total_combinations = np.prod(args)

    # Initialize indices
    idx = [slice(None) if i in none_dims else 0 for i in range(len(args))]

    for _ in range(total_combinations):
        if ctrl is None:
            yield tuple(idx)
        else:
            yield tuple([tuple(idx)]) + (ctrl(idx, args),)

        # Update indices
        for i in permute:
            if i in none_dims:
                continue
            idx[i] += 1
            if idx[i] < args[i]:
                break
            idx[i] = 0


if __name__ == '__main__':
    # Example usage
    shape = (2, 7, 3, 4)
    permute = (0, 1, 2, 3)
    none_dims = (0, 3)

    def ctrl(idx, shape):
        return any(
            idx != slice(None) and e == last - 1 for e, last in zip(idx, shape)
        )

    for index_combo in bool_slice(
        *shape, permute=permute, none_dims=none_dims, ctrl=ctrl
    ):
        print(index_combo)
