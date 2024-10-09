from matrix_tools import *


###########################################

# Setting up a list that will contain Solution objects
solutions = []

# Setting up the decorator @solution_for() that will
#   automatically insert solutions to puzzles into the
#   solutions list.

def solution_for(puzzle_id):
  puzzle = training_puzzles[puzzle_id]
  def solution_registrator(f):
    solutions.append(Solution(puzzle, f))
    return f
  return solution_registrator


###########################################

# Helper functions for solving puzzles.
#   When reading this code, it's best to skip to the next
#   section which contains the actual solutions to see how
#   and where these helper functions are used.

# Functions in this area will probably see a lot of change
#   as new solutions are coded.

def crop_while(matrix, condition):
  matrix = copy(matrix)
  i_min, i_max, j_min, j_max = 0, matrix.height()-1, 0, matrix.width()-1
  while i_min <= i_max and condition(matrix[i_min]): i_min += 1
  while i_max >= i_min and condition(matrix[i_max]): i_max -= 1
  matrix = matrix.transpose()
  while j_min <= j_max and condition(matrix[j_min]): j_min += 1
  while j_max >= j_min and condition(matrix[j_max]): j_max -= 1
  matrix = matrix.transpose()
  return Area(matrix, (i_min, i_max, j_min, j_max))

def crop_to_yellow_markers(matrix):
  return crop_while(matrix, lambda r: (4 not in r))

def remove_black_edges(matrix):
  return crop_while(matrix, lambda r: not any(r))

def valid_overlap(matrix1, matrix2, moves_right, moves_down):
  if matrix1.width()  + moves_right > matrix2.width() : return False
  if matrix1.height() + moves_down  > matrix2.height(): return False
  for i in range(matrix1.height()):
    for j in range(matrix1.width()):
      target = matrix2[i+moves_down][j+moves_right]
      if target == 0: continue
      if target == matrix1[i][j]: continue
      return False
  return True

def find_valid_overlap(matrix1, matrix2):
  for moves_right in range(matrix2.width()-matrix1.width()+1):
    for moves_down in range(matrix2.height()-matrix1.height()+1):
      if valid_overlap(matrix1, matrix2, moves_right, moves_down):
        return (moves_right, moves_down)

def actually_overlap(matrix1, matrix2, moves_right, moves_down):
  output = copy(matrix2)
  for i in range(matrix1.height()):
    for j in range(matrix1.width()):
      output[i+moves_down][j+moves_right] = matrix1[i][j]
  return output
  
def has_solid_border(area):
  return len(set(area[0] + area[-1] + [l[0] for l in area] + [l[-1] for l in area]))==1

def has_blue_border(area):
  return has_solid_border(area) and area[0][0]==1

def is_black(area):
  return set(area.matrix.crop_to(area).colors())=={BLACK}

def copy_area(source, destination, location):
  a,b = location
  for i in range(source.height()):
    for j in range(source.width()):
      if not(a+i<destination.height() and b+j<destination.width()): continue
      destination[a+i][b+j] = source[i][j]

###########################################


@solution_for('8403a5d5')
def f(input_matrix):
  colored_cell_x = [k==0 for k in input_matrix[-1]].index(False)
  cell_color     = input_matrix[-1][colored_cell_x]
  output_matrix  = copy(input_matrix)
  for j in range(colored_cell_x, output_matrix.width(), 2):
    for i in range(output_matrix.height()):
      output_matrix[i][j] = cell_color
  for j in range(colored_cell_x + 1, output_matrix.width(), 2):
    output_matrix[0 if (j-colored_cell_x)%4==1 else -1][j] = 5
  return output_matrix


@solution_for('846bdb03')
def f(input_matrix):
  rect1                 = crop_to_yellow_markers(input_matrix)
  output_matrix         = input_matrix.crop_to(rect1)
  modified_input_matrix = input_matrix.fill_area(rect1, BLACK)
  rect2                 = remove_black_edges(modified_input_matrix)
  modified_input_matrix = modified_input_matrix.crop_to(rect2)
  left_side_color       = max(line[0] for line in modified_input_matrix)
  right_side_color      = max(line[-1] for line in modified_input_matrix)
  modified_input_matrix = modified_input_matrix.pad(['left'] ,  left_side_color)
  modified_input_matrix = modified_input_matrix.pad(['right'], right_side_color)
  for mat in [modified_input_matrix, modified_input_matrix.flip_horizontally()]:
    overlap_coords = find_valid_overlap(mat, output_matrix)
    if overlap_coords: return actually_overlap(mat, output_matrix, *overlap_coords)


@solution_for('855e0971')
def f(input_matrix):
  output_matrix = copy(input_matrix)
  horizontal = len(set(output_matrix[0]) - {0}) == 1
  if not horizontal: output_matrix = output_matrix.transpose()
  target = [list(set(line)-{0})[0] for line in output_matrix]
  for j in range(output_matrix.width()):
    changed = set(target[i] for i in range(output_matrix.height()) if output_matrix[i][j]==0)
    for i in range(output_matrix.height()):
      if output_matrix[i][j] in changed: output_matrix[i][j]=0
  if not horizontal: output_matrix = output_matrix.transpose()
  return output_matrix


@solution_for('85c4e7cd')
def f(input_matrix):
  output_matrix = copy(input_matrix)
  color_order = input_matrix[(input_matrix.height()-1)//2][:(input_matrix.height()-1)//2+1]
  color_replacements = {color_order[i]:color_order[-i-1] for i in range(len(color_order))}
  output_matrix = output_matrix.replace_colors(color_replacements)
  return output_matrix


@solution_for('868de0fa')
def f(input_matrix):
  output_matrix = copy(input_matrix)
  squares = [area for area in output_matrix.areas() if area.width()>1 and area.height()>1 and has_blue_border(area)]
  for square in squares:
    location_to_fill = (square.i_min+1, square.j_min+1)
    output_matrix = output_matrix.fill(location_to_fill, ORANGE if square.width()%2==1 else RED)
  return output_matrix


@solution_for('8731374e')
def f(input_matrix):
  rectangle        = max((area for area in input_matrix.areas() if has_solid_border(area)), key=lambda area: area.area())
  output_matrix    = input_matrix.crop_to(rectangle)
  background_color = output_matrix[0][0]
  other_color      = list(set(output_matrix.colors()) - {background_color})[0]
  rows_to_color    = {i for i in range(output_matrix.height()) if other_color in output_matrix.row(i)}
  columns_to_color = {j for j in range(output_matrix.width())  if other_color in output_matrix.column(j)}
  output_matrix    = output_matrix.cellwise(lambda m,i,j: other_color if (i in rows_to_color) or (j in columns_to_color) else background_color)
  return output_matrix


@solution_for('88a10436')
def f(input_matrix):
  output_matrix = copy(input_matrix)
  shape     = crop_while(input_matrix, lambda l: set(l).issubset({0,5}))
  gray_cell = crop_while(input_matrix, lambda l: 5 not in l)
  copy_area(shape, output_matrix, (gray_cell.i_min-1, gray_cell.j_min-1))
  return output_matrix


@solution_for('88a62173')
def f(input_matrix):
  shapes = [Area(input_matrix, (i,i+1,j,j+1)) for i in [0,3] for j in [0,3]]
  least_common_shape = Counter(shapes).most_common()[-1][0]
  output_matrix = input_matrix.crop_to(least_common_shape)
  return output_matrix


@solution_for('890034e9')
def f(input_matrix):
  input_matrix_colors = input_matrix.colors()
  potential_areas = [area for area in input_matrix.areas() if area.width()>2 and area.height()>2 and has_solid_border(area)]
  rectangle  = [area for area in potential_areas if input_matrix.crop_to(area).colors()[area[0][0]]==input_matrix_colors[area[0][0]]][0]
  dark_spots = [area for area in input_matrix.areas() if area.width()==rectangle.width()-2 and area.height()==rectangle.height()-2 and is_black(area)]
  output_matrix = copy(input_matrix)
  for dark_spot in dark_spots:
    copy_area(rectangle, output_matrix, (dark_spot.i_min-1,dark_spot.j_min-1))
  return output_matrix


@solution_for('8a004b2b')
def f(input_matrix):
  # This solution is incomplete
  rect1                 = crop_to_yellow_markers(input_matrix)
  output_matrix         = input_matrix.crop_to(rect1)
  modified_input_matrix = input_matrix.fill_area(rect1, BLACK)
  rect2                 = remove_black_edges(modified_input_matrix)
  modified_input_matrix = modified_input_matrix.crop_to(rect2)
  multiplier = 1
  while True:
    resized_shape = modified_input_matrix.magnify(multiplier)
    if resized_shape.width() > output_matrix.width() or resized_shape.height() > output_matrix.height(): break
    #print(find_valid_overlap(resized_shape, output_matrix))
    multiplier += 1
  return input_matrix


@solution_for('8be77c9e')
def f(input_matrix):
  output_matrix = Matrix.empty(input_matrix.width(), input_matrix.height()*2)
  copy_area(input_matrix, output_matrix, (0,0))
  flipped_input = input_matrix.flip_vertically()
  copy_area(flipped_input, output_matrix, (input_matrix.height(),0))
  return output_matrix


@solution_for('8d5021e8')
def f(input_matrix):
  return input_matrix


@solution_for('8d510a79')
def f(input_matrix):
  return input_matrix


@solution_for('8e1813be')
def f(input_matrix):
  return input_matrix


@solution_for('8e5a5113')
def f(input_matrix):
  return input_matrix


@solution_for('8eb1be9a')
def f(input_matrix):
  return input_matrix


@solution_for('8efcae92')
def f(input_matrix):
  return input_matrix


@solution_for('8f2ea7aa')
def f(input_matrix):
  return input_matrix


@solution_for('90c28cc7')
def f(input_matrix):
  return input_matrix


@solution_for('90f3ed37')
def f(input_matrix):
  return input_matrix



###########################################


for solution in solutions:
  print(solution.puzzle, solution.verify(), solution.verify(test=True))