from sdforge import box, cylinder, pyramid

stairs = (
    box((12, 0.5, 8)) |
    box((10, 0.5, 6)).translate((0, 0.5, 0)) |
    box((8, 0.5, 4)).translate((0, 1.0, 0))
)

column = cylinder(radius=0.25, height=4)

columns = None

for x in range(-4, 5, 2):
    front = column.translate((x, 3, -2))
    back = column.translate((x, 3,  2))

    pair = front | back
    columns = pair if columns is None else columns | pair

roof = box((9, 0.5, 5)).translate((0, 5.2, 0))

temple = stairs | columns | roof
temple.render()