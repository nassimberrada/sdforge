from sdforge import box, sphere

head = box((1.2, 1.2, 1.2)).translate((0, 2.0, 0))
body = box((2.0, 2.5, 1.2))

arm = box((0.4, 1.8, 0.4))
arms = (
    arm.translate((-1.4, 0.2, 0)) |
    arm.translate(( 1.4, 0.2, 0))
)

leg = box((0.5, 2.0, 0.5))
legs = (
    leg.translate((-0.5, -2.0, 0)) |
    leg.translate(( 0.5, -2.0, 0))
)

eyes = (
    sphere(0.12).translate((-0.25, 2.1, 0.55)) |
    sphere(0.12).translate(( 0.25, 2.1, 0.55))
)

robot = head | body | arms | legs | eyes
robot.render()