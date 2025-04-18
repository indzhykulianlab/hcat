"""
BUGS
todo: ADD A CLEAR ALL ACTION!
todo: import 2D images
todo: auto get px size from tiff/CZI metadata if possible...
todo: when you import, the adjustment widget doesnt refresh
todo: reject slider based on distance from cochlear path...

todo: if state has been changed, alert if tried to close without saving...

todo: when the user adjusts the freq path, cell lines to the path are no longer valid,
    additionally, their frequencies are no longer valid... FIX: while editing, stop drawing cell lines
    and upon release, queue up a thread to auto re-assign frequencies if possible. \
    This fixes the issue of stale freq assignments. Only thing is my code is SLOW...
    could fix by caching. Cant just do on mouse release because of esc key hits...wait why not. just make it consistent
    for eval region..

todo: piece info widget cpu_button doesnt refresh disabled state after import. I.e. buttons are still greyed out.

todo: when you delete piece while annotating the eval region, things break because active piece is now none
    all errors after this likely stem from deleting while drawing...
todo: when you clear the freq path... remove all the freq assignments of the cells in that piece...

todo: when you clear the image import widget, it doesnt refresh piece list.

todo: refresh cell closest line...
todo: when you hide the boxes, the cell closeset line dissapears

USER CHECKS
Traceback (most recent call last):
  File "/Users/chrisbuswinka/Documents/Projects/hcat/hcat/gui/canvas.py", line 1066, in mouseReleaseEvent
    self.addGTCellToParent(
  File "/Users/chrisbuswinka/Documents/Projects/hcat/hcat/gui/canvas.py", line 305, in addGTCellToParent
    assign_frequency(new_cell)
  File "/Users/chrisbuswinka/Documents/Projects/hcat/hcat/lib/analyze.py", line 77, in assign_frequency
    paths: List[List[Tuple[float, float]]] = [
  File "/Users/chrisbuswinka/Documents/Projects/hcat/hcat/lib/analyze.py", line 78, in <listcomp>
    interpolate_path(p.freq_path) for p in pieces
  File "/Users/chrisbuswinka/Documents/Projects/hcat/hcat/lib/frequency.py", line 51, in interpolate_path
    curvature = np.concatenate(curvature, axis=0)
  File "<__array_function__ internals>", line 180, in concatenate
ValueError: need at least one array to concatenate

"""



# IN PROGRESS
# todo: channel tool - change, update, and re-color channels
"""
todo: cochlea info widget:
    - show total cells
    - show total number of tissue pieces
    - show info on if all pieces have freq path, eval area, etc...
        - little widget that changes color if good or not...
        -
todo: when you select a huge region to make many cells 'gt',
  then there are still hidden cells which might still appear
  this could be solved by "selecting" candidate children from a region_box
  and deleting them if the gt command is thrown...
"""
# todo: navigator widget
# todo: smart annotate (hold down shift to auto predict?)
# todo: hold down ctrl+option to show all predicitons regardless of detection?
# todo: clear all children button
# todo: change "update" button  -> 'Apply' and add an OK button which closes widget...
# todo: disable update if nothing has been changed in import widget...
# todo: czi file import...

"""
Notes on usability?
    - its kind of annoying to toggle default, as there is no good indicator of which
        default celltype youre in...
    - i felt locked in by the sliders; if i changed them after doing a lot of verification,
        the previous areas were no longer verified.
    - cool to have a smart annotate feature, where we highlight candidates which have a lower percentage

"""

"""
Info Density:
    - Add ruler and scale bar on canvas
    - Add arrows which track mouse on scale bar
    - add loading progress bar in bottom
    - add quick piece info sidebar widget
    - add quick image color channel info viewer
    - simple filtering? Custom convolution filter editer? 
    - navigator screen? 
"""

# FEATURES
# todo: custom style for QComboBox
# todo: add tool tips to every button
# todo: color channel tool... (just hide or show rgb - people can deal)
