from ..core import SDFNode

class Material(SDFNode):
    """Applies a color material to a child object."""
    def __init__(self, child, color):
        """
        Internal constructor for a Material node.

        Note: This class is not typically instantiated directly. Use the
        `.color()` method on an SDFNode object instead.

        Args:
            child (SDFNode): The SDF object to apply the material to.
            color (tuple): An (r, g, b) tuple with values from 0.0 to 1.0.
        """
        super().__init__()
        self.child = child
        self.color = tuple(color) # Ensure color is a tuple for hashing
        self.material_id = -1 # Will be set by the renderer during collection

    def __hash__(self):
        # Hash based on color, not object identity.
        return hash(self.color)

    def __eq__(self, other):
        # Equality is based on color, not object identity.
        return isinstance(other, Material) and self.color == other.color

    def to_glsl(self, ctx) -> str:
        """
        Wraps the child's GLSL result, injecting the material ID into the .y component.
        """
        child_var = self.child.to_glsl(ctx)
        # Result is vec4(distance, material_id, 0, 0)
        result_expr = f"vec4({child_var}.x, {float(self.material_id)}, {child_var}.zw)"
        return ctx.new_variable('vec4', result_expr)

    def to_callable(self):
        # Materials are a render-time concept; for mesh generation, we use the child's shape.
        return self.child.to_callable()

    def _collect_materials(self, materials: list):
        """
        Adds this material to the list if not already present and assigns an ID,
        then continues traversal.
        """
        # Using `__eq__` and `__hash__`, we can now check for presence by value.
        if self not in materials:
            self.material_id = len(materials)
            materials.append(self)
        else:
            # If it's already in the list, find the original and use its ID.
            # This ensures all Material objects with the same color get the same ID.
            self.material_id = materials[materials.index(self)].material_id
            
        self.child._collect_materials(materials)