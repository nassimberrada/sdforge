from sdforge import box

seat = box((3, 0.8, 3))

back = box((3, 3, 0.6)).translate((0, 1.8, -1.2))

left_arm = box((0.6, 1.4, 3)).translate((-1.2, 0.4, 0))
right_arm = box((0.6, 1.4, 3)).translate((1.2, 0.4, 0))

leg = box((0.4, 1.5, 0.4))

legs = (
    leg.translate((-1.1, -1.1, -1.1)) |
    leg.translate(( 1.1, -1.1, -1.1)) |
    leg.translate((-1.1, -1.1,  1.1)) |
    leg.translate(( 1.1, -1.1,  1.1))
)

chair = seat | back | left_arm | right_arm | legs
chair.render()