import pysubgroup as ps
from pysubgroup.datasets import get_titanic_data
from thesis.prediction_target import BinaryTarget, StandardQF


def test_prediction_target():

    # Load the example dataset
    data = get_titanic_data()

    # target / qf
    target = BinaryTarget ('Survived', True)
    qf = StandardQF(1.0)

    searchspace = ps.create_selectors(data, ignore=['Survived'])
    task = ps.SubgroupDiscoveryTask (
        data,
        target,
        searchspace,
        result_set_size=5,
        depth=2,
        qf=qf)
    result = ps.BeamSearch().execute(task)
    # result = ps.DFS().execute(task)

    result.to_dataframe()

    assert True


if __name__ == '__main__':
    test_prediction_target()
