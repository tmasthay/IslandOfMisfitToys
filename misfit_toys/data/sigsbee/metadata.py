from mh.core_legacy import save_metadata


@save_metadata(cli=True)
def metadata():
    return {}


if __name__ == "__main__":
    metadata()
