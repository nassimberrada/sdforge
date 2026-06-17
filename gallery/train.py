from sdforge import box, cylinder, X, Z

body = box((6, 2, 2))

boiler = cylinder(
    radius=1.0,
    height=5
).rotate(Z, 3.14 / 2.0).translate((0, 1.2, 0))

cab = box((2, 2.5, 2)).translate((-2.5, 1.2, 0))

wheel = cylinder(radius=0.8, height=0.4).rotate(X, 3.14 / 2.0)

wheels = None
for x in (-2.5, -1, 1, 2.5):
    left = wheel.translate((x, -1.3, -1.1))
    right = wheel.translate((x, -1.3, 1.1))

    pair = left | right
    wheels = pair if wheels is None else wheels | pair

chimney = cylinder(radius=0.3, height=1.5).translate((2, 2.5, 0))

train = body | boiler | cab | wheels | chimney
train.render()