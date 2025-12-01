from unittest.mock import MagicMock
from sdforge import sphere, Debug
from sdforge.api.utils import Param
from sdforge.api.scene import NativeRenderer, Scene

def test_param_init():
    p = Param("Size", 1.0, 0.0, 2.0)
    assert p.name == "Size"
    assert p.value == 1.0
    assert p.min_val == 0.0
    assert p.max_val == 2.0
    assert p.uniform_name.startswith("u_param_Size_")

def test_param_to_glsl():
    p = Param("My Param", 0.5, 0, 1)
    assert p.to_glsl() == p.uniform_name
    assert str(p) == p.uniform_name

def test_param_uniform_name_is_unique():
    p1 = Param("Same Name", 1, 0, 2)
    p2 = Param("Same Name", 1, 0, 2)
    assert p1.uniform_name != p2.uniform_name

def test_param_name_sanitization():
    p = Param("Invalid Name!@#$", 1, 0, 2)
    assert "Invalid" in p.uniform_name
    assert "!" not in p.uniform_name
    assert "@" not in p.uniform_name
    assert p.uniform_name.startswith("u_param_Invalid_Name____")

def test_debug_instantiation():
    d = Debug('normals')
    assert d.mode == 'normals'

    d_slice = Debug('slice', plane='XZ', slice_height=1.5, view_scale=10.0)
    assert d_slice.mode == 'slice'
    assert d_slice.plane == 'xz'
    assert d_slice.slice_height == 1.5
    assert d_slice.view_scale == 10.0

def test_render_with_debug_normals():
    s = sphere()
    scene = Scene(s, debug=Debug('normals'))
    renderer = NativeRenderer(scene)
    mock_ctx = MagicMock()
    renderer.ctx = mock_ctx
    renderer._compile_shader()
    call_args, call_kwargs = mock_ctx.program.call_args
    fragment_shader = call_kwargs.get('fragment_shader', '')
    assert "color = debugNormals(normal);" in fragment_shader
    assert "ambientOcclusion(p, normal" not in fragment_shader

def test_render_with_debug_steps():
    s = sphere()
    scene = Scene(s, debug=Debug('steps'))
    renderer = NativeRenderer(scene)
    mock_ctx = MagicMock()
    renderer.ctx = mock_ctx
    renderer._compile_shader()
    call_args, call_kwargs = mock_ctx.program.call_args
    fragment_shader = call_kwargs.get('fragment_shader', '')
    assert "color = debugSteps(hit.z, 100.0);" in fragment_shader

def test_render_with_slice_debug():
    s = sphere()
    debug = Debug('slice', plane='xy', slice_height=0.5, view_scale=4.0)
    scene = Scene(s, debug=debug)
    renderer = NativeRenderer(scene)
    mock_ctx = MagicMock()
    renderer.ctx = mock_ctx
    renderer._compile_shader()
    call_args, call_kwargs = mock_ctx.program.call_args
    fragment_shader = call_kwargs.get('fragment_shader', '')
    # Slice debug logic completely overrides raymarching
    assert "raymarch(ro, rd)" not in fragment_shader
    assert "cameraStatic" not in fragment_shader
    assert "vec3 p = vec3(st * 2.0, 0.5)" in fragment_shader
    assert "debugDistanceField(d)" in fragment_shader

def test_render_with_slice_debug_xz_plane():
    s = sphere()
    debug = Debug('slice', plane='xz', slice_height=1.0, view_scale=2.0)
    scene = Scene(s, debug=debug)
    renderer = NativeRenderer(scene)
    mock_ctx = MagicMock()
    renderer.ctx = mock_ctx
    renderer._compile_shader()
    call_args, call_kwargs = mock_ctx.program.call_args
    fragment_shader = call_kwargs.get('fragment_shader', '')
    assert "vec3 p = vec3(st.x * 1.0, 1.0, st.y * 1.0)" in fragment_shader

def test_render_with_invalid_debug_mode(capsys):
    s = sphere()
    scene = Scene(s, debug=Debug('invalid_mode'))
    renderer = NativeRenderer(scene)
    mock_ctx = MagicMock()
    renderer.ctx = mock_ctx
    renderer._compile_shader()
    call_args, call_kwargs = mock_ctx.program.call_args
    fragment_shader = call_kwargs.get('fragment_shader', '')
    assert "ambientOcclusion(p, normal" in fragment_shader
    assert "debugNormals(normal)" not in fragment_shader