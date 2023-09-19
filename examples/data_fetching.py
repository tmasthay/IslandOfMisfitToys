import misfit_toys.data.marmousi.factory as m1
import misfit_toys.data.marmousi2.factory as m2
import misfit_toys.data.das_curtin.factory as das

factory = m1.Factory(path='conda/data/marmousi')
factory.manufacture_data()

factory = m2.Factory(path='conda/data/marmousi2')
factory.manufacture_data()

pull_das = False
if( pull_das ):
    factory = das.Factory(path='conda/data/das_curtin')
    print('Curtin DAS data takes a while, almost 10 GB of data...starting now')
    d = factory.manufacture_data()
    print('Curtin DAS data extraction successful')
else:
    print(
        'Curtin DAS data not pulled.  To pull, set pull_das=True in ' 
        'examples/data_fetching.py. Dataset is large, ~10 GB.'
    )
