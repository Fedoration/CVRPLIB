import cvrplib
import matplotlib.pyplot as plt
import numpy as np
import random
import re
import sys

import itertools

ITERS_LIMIT = 10000
ITERS_STAGNATION_LIMIT = 1000
SEED = 123

SET_A = [
    "A-n32-k5",
    "A-n33-k5",
    "A-n33-k6",
    "A-n34-k5",
    "A-n36-k5",
    "A-n37-k5",
    "A-n37-k6",
    "A-n38-k5",
    "A-n39-k5",
    "A-n39-k6",
    "A-n44-k6",
    "A-n45-k6",
    "A-n45-k7",
    "A-n46-k7",
    "A-n48-k7",
    "A-n53-k7",
    "A-n54-k7",
    "A-n55-k9",
    "A-n60-k9",
    "A-n61-k9",
    "A-n62-k8",
    "A-n63-k9",
    "A-n63-k10",
    "A-n64-k9",
    "A-n65-k9",
    "A-n69-k9",
    "A-n80-k10",
]

SET_B = [
    "B-n31-k5",
    "B-n34-k5",
    "B-n35-k5",
    "B-n38-k6",
    "B-n39-k5",
    "B-n41-k6",
    "B-n43-k6",
    "B-n44-k7",
    "B-n45-k5",
    "B-n45-k6",
    "B-n50-k7",
    "B-n50-k8",
    "B-n51-k7",
    "B-n52-k7",
    "B-n56-k7",
    "B-n57-k7",
    "B-n57-k9",
    "B-n63-k10",
    "B-n64-k9",
    "B-n66-k9",
    "B-n67-k10",
    "B-n68-k9",
    "B-n78-k10",
]

SET_E = [
    "E-n13-k4",
    "E-n22-k4",
    "E-n23-k3",
    "E-n30-k3",
    "E-n31-k7",
    "E-n33-k4",
    "E-n51-k5",
    "E-n76-k7",
    "E-n76-k8",
    "E-n76-k10",
    "E-n76-k14",
    "E-n101-k8",
    "E-n101-k14",
]


def find_customer_groups(distance_matrix, n_customers, demands, instance):
    customer_groups = [0]
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            if distance_matrix[i][j] == 0:
                if j not in customer_groups:
                    customer_groups.append(j)
                demands[i] = instance.demands[i] + instance.demands[j]
    return customer_groups, demands


def init_tau(distance_matrix, n_customers):
    most_consuming_edge = 0
    for i in range(n_customers):
        for j in range(n_customers):
            if (
                distance_matrix[i][j] != 0
                and distance_matrix[i][j] > most_consuming_edge
            ):
                most_consuming_edge = distance_matrix[i][j]

    tau = ((n_customers) * most_consuming_edge) ** (-1)
    return tau


def aco_optimization(alpha, beta, q0, n_ants, instance):
    print(f"Start optimization {instance.name}")

    best_route = []
    minimal_route_cost = 50000
    iter = 1
    stagnation_iter = 0

    np.random.seed(SEED)
    random.seed(SEED)

    distance_matrix = tuple(instance.distances)
    capacity = instance.capacity
    customers = instance.customers
    n_customers = instance.n_customers
    demands = instance.demands.copy()

    customer_groups, demands = find_customer_groups(
        distance_matrix, n_customers, demands, instance
    )

    tau = init_tau(distance_matrix, n_customers)
    pheromon_matrix = tuple(
        [tau for _ in range(n_customers + 1)] for _ in range(n_customers + 1)
    )
    # print("Start optimization")

    # Зададим количество итерации и итерации стагнации
    while (iter <= ITERS_LIMIT) and (stagnation_iter <= ITERS_STAGNATION_LIMIT):
        for ant in range(n_ants):
            current_route = [0]
            non_visited_customers = list(range(1, n_customers + 1))

            for group_id in range(1, len(customer_groups)):
                non_visited_customers.remove(customer_groups[group_id])

            capacity_remains = capacity
            vehicles = 1

            # Посещаем клиентов
            # print("Visit customers")
            while len(non_visited_customers) > 0:
                current_customer = current_route[-1]

                # preferences (предпочтения) равны феромону умноженному на обратное расстояние
                preferences = np.array(
                    [
                        (pheromon_matrix[current_customer][customer] ** (1))
                        * ((1 / distance_matrix[current_customer][customer]) ** beta)
                        for customer in non_visited_customers
                        if (
                            customer not in current_route
                            and distance_matrix[current_customer][customer] != 0
                        )
                    ]
                )

                # print("Select next customer")
                if np.random.random_sample() < q0:
                    # Идём по пути максимального предпочтения
                    next_customer = non_visited_customers[np.argmax(preferences)]
                else:
                    # Идём по некоторому "случайному" пути
                    next_customer = np.random.choice(
                        non_visited_customers, p=preferences / sum(preferences)
                    )

                capacity_remains -= demands[next_customer]

                if capacity_remains < 0:
                    # Если машина заполнена полностью - отправляем на склад
                    current_route.append(0)
                    capacity_remains = capacity
                    vehicles += 1
                else:
                    non_visited_customers.remove(next_customer)
                    current_route.append(next_customer)
                # print(f"vehicles: {vehicles}")
                # print(f"len(non_visited_customers): {len(non_visited_customers)}")
                # print(f"non_visited_customers: {non_visited_customers}")

            current_route.append(0)
            iter += 1
            current_route_cost = 0

            # Считаем матрицу феромонов и общую длину пути
            for i in range(1, len(current_route)):
                # Локальное обновление феромона
                p1, p2 = current_route[i - 1], current_route[i]
                pheromon_matrix[p1][p2] = pheromon_matrix[p2][p1] = (
                    pheromon_matrix[p1][p2] * (1 - alpha) + alpha * tau
                )
                current_route_cost += distance_matrix[p1][p2]

            # Проверяем, является ли этот путь лучше предыдущего
            if current_route_cost < minimal_route_cost:
                best_route = current_route.copy()
                minimal_route_cost = current_route_cost
                stagnation_iter -= 1
            else:
                stagnation_iter += 1

        for i in range(1, len(best_route)):
            # Глобальное обновление феромона
            p_1, p_2 = best_route[i - 1], best_route[i]
            pheromon_matrix[p_1][p_2] = pheromon_matrix[p_2][p_1] = (
                pheromon_matrix[p_1][p_2] * (1 - alpha)
                + 10 * alpha / current_route_cost
            )

    # print(f"Количество итераций: {iter}")
    # print(stagnation_iter)
    return (best_route, minimal_route_cost, vehicles)


def solve_instance(instance_name, alpha, beta, q0, n_ants):
    instance, solution = cvrplib.download(instance_name, solution=True)

    route, cost, n_cars = aco_optimization(
        alpha=alpha, beta=beta, q0=q0, n_ants=n_ants, instance=instance
    )
    return route, cost, n_cars


def grid_search(param_grid):
    keys = param_grid.keys()
    values = param_grid.values()

    param_combinations = list(itertools.product(*values))

    for combination in param_combinations:
        param_dict = dict(zip(keys, combination))
        yield param_dict


if __name__ == "__main__":
    # alpha = 0.8
    # beta = 3
    # q0 = 0.4
    # p = 0.6
    # n_ants = 50

    param_grid = {
        "alpha": [0.2, 0.4, 0.6, 0.8],
        "beta": [1, 2, 3],
        "q0": [0.2, 0.4, 0.6, 0.8],
    }

    min_cost = 1e6
    min_cars = 1e6
    best_params = None
    for i, params in enumerate(grid_search(param_grid)):
        alpha = params["alpha"]
        beta = params["beta"]
        q0 = params["q0"]
        n_ants = 50

        costs = []
        for inst_name in SET_E:
            route, cost, n_cars = solve_instance(
                instance_name=inst_name, alpha=alpha, beta=beta, q0=q0, n_ants=n_ants
            )
            costs.append(cost)

        cost = np.mean(costs)

        if cost < min_cost:
            min_cost = cost
            min_cars = n_cars
            best_params = params

        print(f"\nИтерация [{i}]")
        print(f"Параметры: {params}")
        print(f"Стоимость маршрута: {cost}")
        print(f"Количество тс: {n_cars}")
        print(f"Лучшая стоимость: {min_cost}")
        print(f"Лучшее количество тс: {min_cars}")
        print(f"Лучшие параметры: {best_params}")
        print("--------------------------------")
