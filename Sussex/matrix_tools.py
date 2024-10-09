import os, json
from   functools   import reduce
from   collections import Counter
from   copy        import deepcopy as copy

BLACK, BLUE, RED, GREEN, YELLOW, GRAY, PINK, ORANGE, CYAN, CRIMSON = range(10)

class Matrix(list):

    """Provides methods for manipulating matrices of colors."""
    
    def height(matrix): return len(matrix)
    def width (matrix): return len(matrix[0])
    def size  (matrix): return f'{matrix.width()}x{matrix.height()}'
    
    def row   (matrix, i): return matrix[i]
    def column(matrix, j): return [matrix[i][j] for i in range(matrix.height())]
    
    def flip_vertically  (matrix): return Matrix(matrix[::-1])
    def flip_horizontally(matrix): return Matrix([line[::-1] for line in matrix])
        
    def transpose(matrix):
        return Matrix([[matrix[i][j] for i in range(matrix.height())] for j in range(matrix.width())])
    
    @classmethod
    def empty(_, width, height):
        """Returns an empty matrix with the given width and height."""
        return Matrix([[0]*width for i in range(height)])
    
    def __repr__(matrix):
        """Prints the matrix contents in a human-readable format."""
        return '\n'.join(''.join(map(str,line)) for line in matrix)
    
    def cellwise(matrix, function):
        """Takes a function and returns the current matrix, with that
           function applied to every cell of the matrix.
           
           The function provided should accept three arguments:
           a matrix object,and the i and j coordinates of the cell
           to be processed."""
        matrix = copy(matrix)
        for i in range(matrix.height()):
            for j in range(matrix.width()):
                matrix[i][j] = function(matrix, i, j)
        return matrix
    
    def areas(matrix):
        """A generator for all the Area objects available in the matrix."""
        for i_min in range(matrix.height()):
            for i_max in range(i_min, matrix.height()):
                for j_min in range(matrix.width()):
                    for j_max in range(j_min, matrix.width()):
                        limits = (i_min, i_max, j_min, j_max)
                        yield Area(matrix, limits)
    
    def crop_to(matrix, area):
        """Given an Area object, returns a version of the matrix cropped to that area."""
        return Matrix([l for l in area])
    
    def colors(matrix):
        """Returns a map that gives the frequency of each color in the matrix."""
        return Counter(reduce(lambda a,b: a+b, matrix))
    
    def multiply(matrix1, matrix2):
        """Makes a copy of matrix1 for every truthy cell in matrix2.
           For example, m.multiply([[1,1]]) would return a matrix twice as wide
           as the original and containing two copies of it side by side."""
        output = Matrix.empty(matrix1.width()*matrix2.width(), matrix1.height()*matrix2.height())
        for i1 in range(matrix1.height()):
            for j1 in range(matrix1.width()):
                for i2 in range(matrix2.height()):
                    for j2 in range(matrix2.width()):
                        output[i1+i2*matrix1.height()][j1+j2*matrix1.width()] = \
                            matrix1[i1][j1] if matrix2[i2][j2] else 0
        return output
    
    def magnify(matrix, multiplier):
        """Returns a copy of the matrix where each 1x1 cell has been replaced by a
           multiplier x multiplier square."""
        output = Matrix.empty(matrix.width()*multiplier, matrix.height()*multiplier)
        for i1 in range(matrix.height()):
            for j1 in range(matrix.width()):
                for i2 in range(multiplier):
                    for j2 in range(multiplier):
                        output[i1*multiplier+i2][j1*multiplier+j2] = matrix[i1][j1]
        return output

    def pad(matrix, sides, color):
      """Returns a copy of the matrix with a padding of the given color on the given
         sides."""
      if 'left'   in sides: matrix = Matrix([[color]+line for line in matrix])
      if 'right'  in sides: matrix = Matrix([line+[color] for line in matrix])
      if 'top'    in sides: matrix = Matrix([[color]*matrix.width()] + matrix)
      if 'bottom' in sides: matrix = Matrix(matrix + [[color]*matrix.width()])
      return matrix

    def fill_area(matrix, area, color):
      """Returns a copy of the matrix with the rectangular Area given painted
         in the given color."""
      matrix = copy(matrix)
      for i in range(area.i_min, area.i_max+1):
        for j in range(area.j_min, area.j_max+1):
          matrix[i][j] = color
      return matrix


    def fill(matrix, location, color):
      """Returns a copy of the matrix with the area containing the given location
         flood-filled with the given color."""
      matrix = copy(matrix)
      known_to_color  = set()
      new_to_color    = {location}
      while new_to_color:
        new_new_to_color = set()
        for (i,j) in new_to_color:
          new_new_to_color |= {(a,b) for (a,b) in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)] if 0<=a<matrix.height() and 0<=b<matrix.width() and matrix[a][b]==matrix[i][j]}
        known_to_color |= new_to_color
        new_to_color = new_new_to_color - known_to_color
      for (i,j) in known_to_color:
        matrix[i][j] = color
      return matrix

    def replace_colors(matrix, replacements):
      """Takes a dictionary mapping colors to other colors, returns a copy of
         the matrix with the colors replaced accordingly."""
      return matrix.cellwise(lambda m,i,j: replacements.get(m[i][j],m[i][j]))



class Area:
    """An Area object denotes a rectangular area within a given matrix."""
    
    def __init__(area, matrix, limits):
        area.matrix = matrix
        area.limits = limits
        area.i_min, area.i_max, area.j_min, area.j_max = limits
    
    def __getitem__(area, k):
        return area.matrix[(area.i_min + k) if k>=0 else (area.i_max + 1 + k)][area.j_min:area.j_max+1]
    
    def __iter__(area):
        for i in range(area.i_max-area.i_min+1): yield area[i]
    
    def width(area):  return area.j_max - area.j_min + 1
    def height(area): return area.i_max - area.i_min + 1
    def area(area):   return area.width() * area.height()
    
    def __hash__(area):
        return hash(area.matrix.crop_to(area).__repr__())
    
    def __eq__(area, other_area):
        return hash(other_area) == hash(area)



class Example(dict):
    """An input-output pair in a puzzle."""
    def __init__(example, data):
        super().__init__(data)
        example.input  = Matrix(example['input'])
        example.output = Matrix(example['output'])
    def __repr__(example):
        return f"Input ({example.input.size()}):\n\n{example.input}\n\n" + \
               f"Output ({example.output.size()}):\n\n{example.output}"



class Puzzle(dict):
    """A collection of both training and testing input-output pairs that
       constitute a single puzzle."""
    
    def __init__(puzzle, fname):
        super().__init__(json.load(open(fname)))
        puzzle.id    = fname.split('/')[-1].split('.')[0]
        puzzle.train = [Example(e) for e in puzzle['train']]
        puzzle.test  = [Example(e) for e in puzzle['test']]
    
    def __repr__(puzzle):
        return f'<Puzzle {puzzle.id}>'
    
    def worked_examples(puzzle):
        separator = f"\n\n{'='*15}\n\n"
        body = separator.join(f'Pair #{i}\n\n{e}' for (i,e) in enumerate(puzzle.train, start=1))
        return separator + body + separator



class Solution:
    """A solution for a specific puzzle."""
    
    def __init__(solution, puzzle, func, name=''):
        solution.name   = name
        solution.puzzle = puzzle
        solution.f      = func
    
    def __repr__(solution):
        return f'<Solution {solution.name}>' if solution.name else f'<Unnamed solution>'
    
    def verify(solution, test=False):
        """Checks that the solution's function solves it correctly for
           the training examples.
           
           If test is set to True, this will instead verify the function
           against the test examples."""
        puzzle = solution.puzzle
        for example in (puzzle.test if test else puzzle.train):
            if solution.f(example.input) != example.output:
                return False
        return True


###########################################

training_dir = '../data/training'

training_puzzles = {}

for fname in os.listdir(training_dir):
    puzzle = Puzzle(f'{training_dir}/{fname}')
    training_puzzles[puzzle.id] = puzzle