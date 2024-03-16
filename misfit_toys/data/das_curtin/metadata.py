from mh.core_legacy import save_metadata


@save_metadata(cli=True)
def metadata():
    return {
        "url": "https://ddfe.curtin.edu.au/7h0e-d392/",
        "ext": "sgy",
        "das_curtin": {"filename": "2020_GeoLab_WVSP_DAS_wgm"},
        "geophone_curtin": {"filename": "2020_GeoLab_WVSP_geophone_wgm"},
    }


if __name__ == "__main__":
    metadata()
