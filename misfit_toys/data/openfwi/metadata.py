from masthay_helpers.global_helpers import save_metadata


@save_metadata(cli=True)
def metadata():
    return {
        'ext': 'npy',
        'num_urls': 10,
        'mode': 'front',
        'derived': {'FlatVel_A': {}},
    }


if __name__ == "__main__":
    metadata()
