import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def get_vqc_result(
    num_params: int, cost_function: callable, optimizer_function: callable
) -> Tuple[float, NDArray[np.float_]]:
    params = np.zeros(num_params)
    params = optimizer_function(cost_function, params)
    return cost_function(params), params


# Ce qu'il va manquer c'est juste créer le circuit et cost_function avec circuit complet et vqc duqeul on retourn le mean square error, fonction principale... à voir comment elle veut qu'on le fasse
