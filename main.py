import matplotlib.pyplot as plt
import time
import numpy as np


def create_matrix(mat_size):
    mat = np.random.randint(2, size=(mat_size, mat_size))  # Press Ctrl+F8 to toggle the breakpoint.
    return mat


def print_matrix(key, mat):
    print("\n=======================================================================================")
    print_list = list()
    print(f'Matrix {key}: [{mat.shape[0]}][{mat.shape[1]}]')
    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in mat]))
    for idx, val in np.ndenumerate(mat):
        if val == 1:
            print_list.append(f'{idx} -> {val}')
    print(print_list)


def multiply_matrix(mat_x, mat_y):
    mat_size = (mat_x.shape[0], mat_y.shape[1])
    result = np.zeros(mat_size)
    # iterating rows of mat_X
    for i in range(0, mat_x.shape[0]):
        # iterating columns of mat_Y
        for j in range(0, mat_y.shape[1]):
            # iterating rows of mat_Y
            for k in range(0, mat_y.shape[0]):
                result[i][j] = 1 if result[i][j] + (mat_x[i][k] * mat_y[k][j]) >= 1 else 0
    return result


def naive_algo(mat, print_res):
    temp = mat
    for i in range(mat.shape[0]):
        temp = multiply_matrix(mat, temp)
        if print_res:
            print(f"W{i+1} =")
            print('\n'.join([''.join(['{:8}'.format(item) for item in row]) for row in temp]))
    return temp


def warshall_algo(mat, print_res):
    result = ""
    for k in range(mat.shape[0]):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[0]):
                mat[i][j] = mat[i][j] or (mat[i][k] and mat[k][j])
        result += ("W" + str(k+1) + " is: \n" + str(mat).replace("],", "] \n") + "\n")
    if print_res:
        print(result)
    return mat


def interactive_console():
    try:
        print("1. View Random matrices of order - 2 to 10")
        print("2. Insert User Inputs")
        input_method = int(input("Select the input method for your relation matrix : 1 or 2\n"))
        if input_method == 1:
            for mat_order in range(2, 11):
                mr = create_matrix(mat_size=mat_order)
                print_matrix("of order", mr)
                # res_mat = np.empty([5, 5])
                print("\n=============== NAIVE ALGORITHM ===============")
                print(f"W0 =")
                print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in mr]))
                res_mat = naive_algo(mr, True)
                print(f'Transitive closure using NAIVE, MR* =')
                print('\n'.join([''.join(['{:6}'.format(item) for item in row]) for row in res_mat]))
                print("\n=============== WARSHALL ALGORITHM ===============")
                print(f'MR0 =')
                print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in mr]))
                res_mat = warshall_algo(mr, True)
                print(f'Transitive closure using WARSHALL, MR* =')
                print('\n'.join([''.join(['{:6}'.format(item) for item in row]) for row in res_mat]))
        elif input_method == 2:
            order = int(input("Enter order of the matrix: 2 to 10\n"))
            if order < 2 or order > 10:
                print("SORRY, not supported.")
            else:
                print("Enter the entries in a single line (separated by space): ")
                entries = list(map(int, input().split()))
                matrix = np.array(entries).reshape(order, order)
                print_matrix("Entered", matrix)
                print("\n=============== NAIVE ALGORITHM ===============")
                print(f"W0 =")
                print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in matrix]))
                res_1 = naive_algo(matrix, True)
                print(f'\nTransitive closure using NAIVE, MR* =')
                print('\n'.join([''.join(['{:8}'.format(item) for item in row]) for row in res_1]))
                print("\n=============== WARSHALL ALGORITHM ===============")
                print(f'MR0 =')
                print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in matrix]))
                res_2 = warshall_algo(matrix, True)
                print(f'\nTransitive closure using WARSHALL, MR* =')
                print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in res_2]))
        else:
            print("SORRY, Not Supported.")
    except Exception as exp:
        print(exp.args)


def log_log_graph(naive_time_list, warshall_time_list):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.loglog(naive_time_list, warshall_time_list, color='blue', marker='o', linestyle='--')
    ax.set_xlabel('NAIVE Algorithm Time')
    ax.set_ylabel('WARSHALL Algorithm Time')
    ax.set_title('Log Log Plot of the time taken')
    plt.show()
    fig.savefig("loglog_plot.png")
    plt.close(fig)


def assignment_problem():
    naive_time_list = []
    warshall_time_list = []
    print("Computing Naive and Warshall algorithm for Matrices of order 10-100")
    for mat_order in range(10, 101):
        mr = np.random.randint(2, size=(mat_order, mat_order))
        start_time = time.time()
        transitive_closure_using_naive = naive_algo(mr, False)
        print(f'Matrix {mat_order}: [{transitive_closure_using_naive.shape[0]}]'
              f'[{transitive_closure_using_naive.shape[1]}]')
        naive_time_list.append(time.time() - start_time)
        # print(transitive_closure_using_naive)
        start_time = time.time()
        transitive_closure_using_warshall = warshall_algo(mr, False)
        print(f'Matrix {mat_order}: [{transitive_closure_using_warshall.shape[0]}]'
              f'[{transitive_closure_using_warshall.shape[1]}]')
        warshall_time_list.append(time.time() - start_time)
        # print(transitive_closure_using_warshall)
    print(naive_time_list)
    print(warshall_time_list)
    log_log_graph(naive_time_list, warshall_time_list)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    more = "n"
    print("======================== MATHS ASSIGNMENT 2 ========================")
    while more == "n":
        print("OPTION 1 : Interactive Console for inserting your own inputs or printing random matrices of order 2-10")
        print("OPTION 2 : Assignment Problem wih Log Log Graph of execution time")
        option = int(input("Select the option for your choice of execution : 1 or 2 else 0 to quit\n"))
        if option == 1:
            interactive_console()
        elif option == 2:
            assignment_problem()
        else:
            print("SORRY, exiting...")
        more = input("Press 'q' when you are done OR press 'n' to try again..\n")
        if more == 'q':
            break