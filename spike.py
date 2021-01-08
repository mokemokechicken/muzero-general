import numpy

BOARD_SIZE = 4
RIGHT, UP, LEFT, DOWN = [0, 1, 2, 3]

def virtual_move(board, action):
    rot_board = numpy.rot90(board, -action).copy()  # for moving always right
    reward = 0
    moved = False
    for row in range(BOARD_SIZE):
        line = rot_board[row]
        merged_in_line = False
        for col in range(BOARD_SIZE - 1, -1, -1):  # 2, 1, 0 (when BOARD_SIZE == 3)
            if line[col] == 0:
                continue
            for r_col in range(col + 1, BOARD_SIZE):
                if line[col] == line[r_col] and not merged_in_line:
                    line[r_col] += 1
                    reward += line[r_col]
                    line[col] = 0
                    moved = True
                    merged_in_line = True
                    print(f"({row}, {col}): rule1")
                    break
                elif r_col == BOARD_SIZE - 1 and line[r_col] == 0:
                    line[r_col] = line[col]
                    line[col] = 0
                    moved = True
                    print(f"({row}, {col}): rule2")
                    break
                elif line[r_col] > 0 and line[r_col-1] == 0 and col < r_col - 1:
                    line[r_col - 1] = line[col]
                    line[col] = 0
                    moved = True
                    print(f"({row}, {col}): rule3")
                    break
                elif line[r_col] > 0:
                    break
    return numpy.rot90(rot_board, action), reward, moved


b = numpy.array([
    [1, 2, 5, 2],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
])
print(virtual_move(b, RIGHT))
