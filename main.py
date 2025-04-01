import matplotlib.pyplot as plt
import numpy as np
import collections
import time 

class PlotResults:
    """
    Class to plot the results. 
    """
    def plot_results(self, data1, data2, label1, label2, filename):
        """
        This method receives two lists of data point (data1 and data2) and plots
        a scatter plot with the information. The lists store statistics about individual search 
        problems such as the number of nodes a search algorithm needs to expand to solve the problem.

        The function assumes that data1 and data2 have the same size. 

        label1 and label2 are the labels of the axes of the scatter plot. 
        
        filename is the name of the file in which the plot will be saved.
        """
        _, ax = plt.subplots()
        ax.scatter(data1, data2, s=100, c="g", alpha=0.5, cmap=plt.cm.coolwarm, zorder=10)
    
        lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
    
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.grid()
        plt.savefig(filename)

class Grid:
    """
    Class to represent an assignment of values to the 81 variables defining a Sudoku puzzle. 

    Variable _cells stores a matrix with 81 entries, one for each variable in the puzzle. 
    Each entry of the matrix stores the domain of a variable. Initially, the domains of variables
    that need to have their values assigned are 123456789; the other domains are limited to the value
    initially assigned on the grid. Backtracking search and AC3 reduce the the domain of the variables 
    as they proceed with search and inference.
    """
    def __init__(self):
        self._cells = []
        self._complete_domain = "123456789"
        self._width = 9

    def copy(self):
        """
        Returns a copy of the grid. 
        """
        copy_grid = Grid()
        copy_grid._cells = [row.copy() for row in self._cells]
        return copy_grid

    def get_cells(self):
        """
        Returns the matrix with the domains of all variables in the puzzle.
        """
        return self._cells

    def get_width(self):
        """
        Returns the width of the grid.
        """
        return self._width

    def read_file(self, string_puzzle):
        """
        Reads a Sudoku puzzle from string and initializes the matrix _cells. 

        This is a valid input string:

        4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......

        This is translated into the following Sudoku grid:

        - - - - - - - - - - - - - 
        | 4 . . | . . . | 8 . 5 | 
        | . 3 . | . . . | . . . | 
        | . . . | 7 . . | . . . | 
        - - - - - - - - - - - - - 
        | . 2 . | . . . | . 6 . | 
        | . . . | . 8 . | 4 . . | 
        | . . . | . 1 . | . . . | 
        - - - - - - - - - - - - - 
        | . . . | 6 . 3 | . 7 . | 
        | 5 . . | 2 . . | . . . | 
        | 1 . 4 | . . . | . . . | 
        - - - - - - - - - - - - - 
        """
        i = 0
        row = []
        for p in string_puzzle:
            if p == '.':
                row.append(self._complete_domain)
            else:
                row.append(p)

            i += 1

            if i % self._width == 0:
                self._cells.append(row)
                row = []
            
    def print(self):
        """
        Prints the grid on the screen. Example:

        - - - - - - - - - - - - - 
        | 4 . . | . . . | 8 . 5 | 
        | . 3 . | . . . | . . . | 
        | . . . | 7 . . | . . . | 
        - - - - - - - - - - - - - 
        | . 2 . | . . . | . 6 . | 
        | . . . | . 8 . | 4 . . | 
        | . . . | . 1 . | . . . | 
        - - - - - - - - - - - - - 
        | . . . | 6 . 3 | . 7 . | 
        | 5 . . | 2 . . | . . . | 
        | 1 . 4 | . . . | . . . | 
        - - - - - - - - - - - - - 
        """
        for _ in range(self._width + 4):
            print('-', end=" ")
        print()

        for i in range(self._width):

            print('|', end=" ")

            for j in range(self._width):
                if len(self._cells[i][j]) == 1:
                    print(self._cells[i][j], end=" ")
                elif len(self._cells[i][j]) > 1:
                    print('.', end=" ")
                else:
                    print(';', end=" ")

                if (j + 1) % 3 == 0:
                    print('|', end=" ")
            print()

            if (i + 1) % 3 == 0:
                for _ in range(self._width + 4):
                    print('-', end=" ")
                print()
        print()

    def print_domains(self):
        """
        Print the domain of each variable for a given grid of the puzzle.
        """
        for row in self._cells:
            print(row)

    def is_solved(self):
        """
        Returns True if the puzzle is solved and False otherwise. 
        """
        for i in range(self._width):
            for j in range(self._width):
                if len(self._cells[i][j]) > 1 or not self.is_value_consistent(self._cells[i][j], i, j):
                    return False
        return True
    
    def is_value_consistent(self, value, row, column):
        for i in range(self.get_width()):
            if i == column: continue
            if self.get_cells()[row][i] == value:
                return False
        
        for i in range(self.get_width()):
            if i == row: continue
            if self.get_cells()[i][column] == value:
                return False

        row_init = (row // 3) * 3
        column_init = (column // 3) * 3

        for i in range(row_init, row_init + 3):
            for j in range(column_init, column_init + 3):
                if i == row and j == column:
                    continue
                if self.get_cells()[i][j] == value:
                    return False
        return True



class FirstAvailable():
    """
    Naïve method for selecting variables; simply returns the first variable encountered whose domain is larger than one.
    """
    def select_variable(self, grid):
        width = grid.get_width()
        domain = grid.get_cells()
        for i in range(width):
            for j in range(width):
                # so check if the domain has more than one value.
                if len(domain[i][j]) > 1:
                    return (i, j)
        return None

class MRV():
    """
    Implements the MRV heuristic, which returns one of the variables with smallest domain. 
    """
    def select_variable(self, grid):
        """
        Uses the MRV Heuristic to return the coordinate of the variable (Minimum Remaining Value)
        """
        width = grid.get_width()
        domain = grid.get_cells()
        low = 10
        res = None
        for i in range(width):
            for j in range(width):
                if len(domain[i][j]) > 1 and len(domain[i][j]) < low:
                    low = len(domain[i][j])
                    res = (i,j)
        return res


class AC3:
    """
    This class implements the methods needed to run AC3 on Sudoku. 
    """
    def remove_domain_row(self, grid, row, column):
        """
        Given a matrix (grid) and a cell on the grid (row and column) whose domain is of size 1 (i.e., the variable has its
        value assigned), this method removes the value of (row, column) from all variables in the same row. 
        """
        variables_assigned = []

        for j in range(grid.get_width()):
            if j != column:
                new_domain = grid.get_cells()[row][j].replace(grid.get_cells()[row][column], '')

                if len(new_domain) == 0:
                    return None, True

                if len(new_domain) == 1 and len(grid.get_cells()[row][j]) > 1:
                    variables_assigned.append((row, j))

                grid.get_cells()[row][j] = new_domain
        
        return variables_assigned, False

    def remove_domain_column(self, grid, row, column):
        """
        Given a matrix (grid) and a cell on the grid (row and column) whose domain is of size 1 (i.e., the variable has its
        value assigned), this method removes the value of (row, column) from all variables in the same column. 
        """
        variables_assigned = []

        for j in range(grid.get_width()):
            if j != row:
                new_domain = grid.get_cells()[j][column].replace(grid.get_cells()[row][column], '')
                
                if len(new_domain) == 0:
                    return None, True

                if len(new_domain) == 1 and len(grid.get_cells()[j][column]) > 1:
                    variables_assigned.append((j, column))

                grid.get_cells()[j][column] = new_domain

        return variables_assigned, False

    def remove_domain_unit(self, grid, row, column):
        """
        Given a matrix (grid) and a cell on the grid (row and column) whose domain is of size 1 (i.e., the variable has its
        value assigned), this method removes the value of (row, column) from all variables in the same unit. 
        """
        variables_assigned = []

        row_init = (row // 3) * 3
        column_init = (column // 3) * 3

        for i in range(row_init, row_init + 3):
            for j in range(column_init, column_init + 3):
                if i == row and j == column:
                    continue

                new_domain = grid.get_cells()[i][j].replace(grid.get_cells()[row][column], '')

                if len(new_domain) == 0:
                    return None, True

                if len(new_domain) == 1 and len(grid.get_cells()[i][j]) > 1:
                    variables_assigned.append((i, j))

                grid.get_cells()[i][j] = new_domain
        return variables_assigned, False

    def pre_process_consistency(self, grid):
        """
        This method enforces arc consistency for the initial grid of the puzzle.

        The method runs AC3 for the arcs involving the variables whose values are 
        already assigned in the initial grid. 
        """
        domain = grid.get_cells()
        queue = collections.deque()
        for i in range(grid.get_width()):
            for j in range(grid.get_width()):
                if len(domain[i][j]) > 1:
                    continue
                else:
                    queue.append((i,j))

        self.consistency(grid, queue)
                    

    def consistency(self, grid, Q):
        """
        This is a domain-specific implementation of AC3 for Sudoku. 

        It keeps a set of variables to be processed (Q) which is provided as input to the method. 
        Since this is a domain-specific implementation, we don't need to maintain a graph and a set 
        of arcs in memory. We can store in Q the cells of the grid and, when processing a cell, we
        ensure arc consistency of all variables related to this cell by removing the value of
        cell from all variables in its column, row, and unit. 

        For example, if the method is used as a preprocessing step, then Q is initialized with 
        all cells that start with a number on the grid. This method ensures arc consistency by
        removing from the domain of all variables in the row, column, and unit the values of 
        the cells given as input. Like the general implementation of AC3, the method adds to 
        Q all variables that have their values assigned during the propagation of the contraints. 

        The method returns True if AC3 detected that the problem can't be solved with the current
        partial assignment; the method returns False otherwise. 
        """
        
        while Q:
            var = Q.popleft()
            i, j = var
            
            # Process removals in row, column, and unit.
            row_assigned, row_failure = self.remove_domain_row(grid, i, j)
            if row_failure:
                return True  # Failure detected
            
            col_assigned, col_failure = self.remove_domain_column(grid, i, j)
            if col_failure:
                return True  # Failure detected
            
            unit_assigned, unit_failure = self.remove_domain_unit(grid, i, j)
            if unit_failure:
                return True  # Failure detected
            
            # Add neighbors that have been reduced to a single value.
            Q.extend(row_assigned + col_assigned + unit_assigned)
        
        # If we exit the loop, no failure was detected.
        return False
class Backtracking:
    """
    Class that implements backtracking search for solving CSPs. 
    """
    def ac_search(self, grid, var_selector):
        """
        Implements backtracking search with inference. 
        """
        ac = AC3()
        # Completion check
        if grid.is_solved():
            return grid
        # Unpacking tuple
        var = var_selector.select_variable(grid)
        if var is None:
            return None
        
        i,j = var
        domain = grid.get_cells()
        for d in domain[i][j]:
            if grid.is_value_consistent(d, i, j):
                copy_grid = grid.copy()
                copy_grid.get_cells()[i][j] = d
                q = collections.deque()
                q.append((i,j))
                if ac.consistency(copy_grid, q):
                    continue

                result = self.ac_search(copy_grid, var_selector)
                if result is not None:
                    return result
        
        return None
    def search(self, grid, var_selector):
        """
        Implements backtracking search without inference. 
        """

        # Completion check
        if grid.is_solved():
            return grid
        # Unpacking tuple
        var = var_selector.select_variable(grid)
        if var is None:
            return None
        
        i,j = var
        domain = grid.get_cells()
        for d in domain[i][j]:
            if grid.is_value_consistent(d, i, j):
                copy_grid = grid.copy()
                copy_grid.get_cells()[i][j] = d

                result = self.search(copy_grid, var_selector)
                if result is not None:
                    return result
        
        return None


file = open('tutorial_problem.txt', 'r')
problems = file.readlines()

for p in problems:
    # Read problem from string
    g = Grid()
    g.read_file(p)

    back = Backtracking()
    mrv = MRV()
    firsta = FirstAvailable()
    ac = AC3()

    # # Print the grid on the screen
    # print('Puzzle')
    # g.print()

    # # # Print the domains of all variables
    # print('Domains of Variables')
    # g.print_domains()
    # print()

    # # Iterate over domain values
    # for i in range(g.get_width()):
    #     for j in range(g.get_width()):

    #         print('Domain of ', i, j, ': ', g.get_cells()[i][j])

    #         for d in g.get_cells()[i][j]:
    #             print(d, end=' ')
    #         print()

    # # # Make a copy of a grid
    # copy_g = g.copy()

    # print('Copy (copy_g): ')
    # copy_g.print()
    # print()

    # print('Original (g): ')
    # g.print()
    # print()

    # print("Starting recursive backtrack test: ")
    # solution = back.search(copy_g, mrv)
    # if solution:
    #     print("Solved easy instance")
    #     solution.print()
    # else:
    #     print("Failed to solve")

    # Checking hard instance with inference based backtracking
    file = open('top95.txt', 'r')
    problems = file.readlines()
    running_time_mrv = []
    running_time_fa = []

    for p in problems:
        # Read problem from string
        g = Grid()
        g.read_file(p)

        # Print the grid on the screen
        print('Puzzle')
        g.print()

        # # Print the domains of all variables
        print('Domains of Variables')
        g.print_domains()
        print()


        # # Make a copy of a grid
        copy_g1 = g.copy()
        copy_g2 = g.copy()

        # Preprocessing before running search
        start = time.perf_counter()
        ac.pre_process_consistency(copy_g1)
        solution = back.ac_search(copy_g1, firsta)
        end = time.perf_counter()
        timed = end - start
        running_time_fa.append(timed)

        if solution:
            print("Solved hard instance with FA")
            solution.print()
        else:
            print("Failed to solve instance")

        start = time.perf_counter()
        ac.pre_process_consistency(copy_g2)
        solution = back.ac_search(copy_g2, mrv)
        end = time.perf_counter()
        timed = end - start
        running_time_mrv.append(timed)
        if solution:
            print("Solved hard instance with MRV")
            solution.print()
        else:
            print("Failed to solve instance")
    
    plotter = PlotResults()
    plotter.plot_results(running_time_mrv, running_time_fa, "Running Time (MRV)", "Running Time (FA)", "running_time")

    


