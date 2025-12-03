from .core import SDFNode, GLSLContext

class Material(SDFNode):
    """
    Applies a color material to a child object.
    Supports masking to apply color only to specific regions.
    """
    def __init__(self, child, color, mask: SDFNode = None):
        """
        Internal constructor for a Material node.

        Args:
            child (SDFNode): The SDF object to apply the material to.
            color (tuple): An (r, g, b) tuple with values from 0.0 to 1.0.
            mask (SDFNode, optional): A mask object. If provided, the material
                                      is only applied where the mask SDF is < 0 (inside).
        """
        super().__init__()
        self.child = child
        self.rgb = tuple(color)
        self.mask = mask
        self.material_id = -1

    def __hash__(self):
        return hash(self.rgb)

    def __eq__(self, other):
        return isinstance(other, Material) and self.rgb == other.rgb

    def _base_to_glsl(self, ctx: GLSLContext, profile_mode: bool) -> str:
        if profile_mode:
            child_var = self.child.to_profile_glsl(ctx)
        else:
            child_var = self.child.to_glsl(ctx)

        new_id = float(self.material_id)
        if self.mask:
            mask_var = self.mask.to_glsl(ctx)
            selector = f"step({mask_var}.x, 0.0)"
            id_expr = f"mix({child_var}.y, {new_id}, {selector})"
        else:
            id_expr = f"{new_id}"
        
        result_expr = f"vec4({child_var}.x, {id_expr}, {child_var}.zw)"
        return ctx.new_variable('vec4', result_expr)

    def to_glsl(self, ctx) -> str:
        return self._base_to_glsl(ctx, profile_mode=False)

    def to_profile_glsl(self, ctx) -> str:
        return self._base_to_glsl(ctx, profile_mode=True)

    def _collect_materials(self, materials: list):
        if self not in materials:
            self.material_id = len(materials)
            materials.append(self)
        else:
            self.material_id = materials[materials.index(self)].material_id
            
        self.child._collect_materials(materials)
        if self.mask:
            self.mask._collect_materials(materials)